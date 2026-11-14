#!/usr/bin/env python3
"""Checks of the harness that need no RQ1 output (check_rq1.py does the rest once RQ1's files are here).

1. The prompt and system message are LiSSA's, character for character (compared with ReasoningClassifier.java).
2. generate_key() is LiSSA's KeyGenerator: it reproduces the identifiers in the names of LiSSA's own GPT-4o result
   files, which LiSSA computes with the same function from the configuration.
3. Gold standard, link IDs and metrics reproduce LiSSA's published GPT-4o results from their trace-link files.
4. The split has the planned sizes, and dev and test are disjoint and cover every requirement.

    python3 harness/selftest.py
"""

import json
import re
import sys

import evaluate
import lissa_compat as lc

failures = []


def check(condition, message):
    print(("ok   " if condition else "FAIL ") + message)
    if not condition:
        failures.append(message)


# 1. Prompt texts
java = (lc.LISSA / "src/main/java/edu/kit/kastel/sdq/lissa/ratlr/classifier/ReasoningClassifier.java").read_text()
check('"' + lc.COT_PROMPT.replace("\n", "\\n") + '"' in java, "CoT prompt equals REASON_WITH_NAME")
check('"' + lc.SYSTEM_MESSAGE + '"' in java, "system message equals ReasoningClassifier's")

# 2. Key function, via Configuration.getConfigurationIdentifierForFile()
MODULES = [("sourceArtifactProvider", "source_artifact_provider"), ("targetArtifactProvider", "target_artifact_provider"),
           ("sourcePreprocessor", "source_preprocessor"), ("targetPreprocessor", "target_preprocessor"),
           ("embeddingCreator", "embedding_creator"), ("sourceStore", "source_store"), ("targetStore", "target_store"),
           ("classifier", "classifier"), ("resultAggregator", "result_aggregator"),
           ("traceLinkIdPostprocessor", "tracelinkid_postprocessor")]


def configuration_to_string(config):
    """Configuration.toString() with ModuleConfiguration.toString() and Java's Map.toString()."""
    def module(m):
        if m is None:
            return "null"
        arguments = ", ".join(f"{k}={v}" for k, v in m["args"].items())
        return f"ModuleConfiguration[name={m['name']}, arguments={{{arguments}}}]"
    return "Configuration{" + ", ".join(f"{java_name}={module(config.get(key))}" for java_name, key in MODULES) + "}"


result_files = sorted((lc.LISSA / "results-r2c").glob("results-*.md"))
matched = 0
for path in result_files:
    config = json.loads(re.search(r"```json\n(.*?)\n```", path.read_text(), re.S).group(1))
    matched += path.stem.endswith("_" + lc.generate_key(configuration_to_string(config)))
check(matched == len(result_files) > 0, f"generate_key reproduces {matched}/{len(result_files)} LiSSA result-file IDs")

# 3. Gold standard, link IDs and metrics


def corrupt(links_file, project):
    """Statistics.saveTraceLinks opens the file with CREATE but not TRUNCATE_EXISTING: overwriting a longer file
    leaves its tail behind. Such a file has duplicate or unsorted lines or IDs that are not artifacts."""
    lines = links_file.read_text().splitlines()
    reqs = {lc.link_id(n) for n in lc.load_artifacts(lc.DATASETS / project / "UC")}
    code = {lc.link_id(n) for n in lc.load_artifacts(lc.DATASETS / project / "CC")}
    pairs = [tuple(line.split(",")) for line in lines]
    return (len(set(lines)) != len(lines) or pairs != sorted(pairs)
            or any(len(p) != 2 or p[0] not in reqs or p[1] not in code for p in pairs))


compared = matched = explained = 0
for path in result_files:
    text = path.read_text()
    project = re.search(r"datasets/req2code/([^/]+)/", text).group(1)
    if project not in lc.PROJECTS:
        continue  # Dronology: no code, no gold standard here
    links_file = path.with_name(path.name.replace("results-", "traceLinks-", 1)[:-3] + ".csv")
    ours = evaluate.evaluate({project: lc.read_trace_links(links_file)}, "all")["projects"][0]
    theirs = {m: float(re.search(rf"\* {label}: (\S+)", text).group(1))
              for m, label in (("tp", "True Positives"), ("fp", "False Positives"), ("fn", "False Negatives"),
                               ("p", "Precision"), ("r", "Recall"), ("f1", "F1"))}
    compared += 1
    if all(abs(ours[m] - theirs[m]) <= 1e-12 for m in theirs):
        matched += 1
    elif corrupt(links_file, project):
        explained += 1
        print(f"     (trace-link file left over from an earlier run, not comparable: {links_file.name[:70]}...)")
check(compared > 0 and matched + explained == compared,
      f"metrics reproduce LiSSA's for {matched}/{compared} GPT-4o runs; {explained} have corrupt trace-link files")

# 4. Split
if evaluate.SPLIT_FILE.exists():
    split = json.loads(evaluate.SPLIT_FILE.read_text())
    sizes = {p: len(s["dev"]) for p, s in split["projects"].items()}
    check(sizes == {"SMOS": 13, "eTour_en": 12, "iTrust": 26}, f"dev sizes {sizes}")
    for project, parts in split["projects"].items():
        everything = evaluate.requirements_in("all", project)
        check(not set(parts["dev"]) & set(parts["test"]) and set(parts["dev"]) | set(parts["test"]) == everything,
              f"{project}: dev and test are disjoint and cover all {len(everything)} requirements")
else:
    print("skip split checks: run split.py first")

# 5. Java parser: every file gives declarations except SMOS's CSS and JSP files
import javaparse  # noqa: E402  (only needed from here on)

no_types = [f"{p}/{name}" for p in lc.PROJECTS
            for name, text in lc.load_artifacts(lc.DATASETS / p / "CC").items() if not javaparse.parse(text).types]
check(no_types == ["SMOS/display.txt", "SMOS/managerUser.txt"], f"files without Java declarations: {no_types}")
java = javaparse.parse(lc.read_artifact(lc.DATASETS / "eTour_en" / "CC" / "ICulturalHeritageAgencyManager.java"))
check("clearCulturalHeritage(int pCulturalHeritageID)" in (javaparse.element_code(java, "X.clearCulturalHeritage(int)") or ""),
      "element_code finds a method from a qualified identifier")
check(javaparse.element_code(java, "notDeclaredHere") is None, "an undeclared identifier fails the identifier check")

# 6. Variants: the baseline is LiSSA's prompt; the others only add text, LiSSA's question and artifacts stay intact
import variants  # noqa: E402

requirement = lc.read_artifact(lc.DATASETS / "eTour_en" / "UC" / "UC1.txt")
code = lc.read_artifact(lc.DATASETS / "eTour_en" / "CC" / "DBCulturalHeritage.java")
pair = variants.Pair("eTour_en", "UC1.txt", "DBCulturalHeritage.java", requirement, code)
check(variants.baseline(pair) == (lc.SYSTEM_MESSAGE, lc.fill(lc.COT_PROMPT, requirement, code)), "baseline = LiSSA's prompt")
lissa_question = lc.fill(variants.QUESTION, requirement, code)
for name, variant in variants.VARIANTS.items():
    try:
        system, user = variant(pair)
    except FileNotFoundError as missing:
        print(f"skip variant {name}: {missing.filename} missing (run prepare_prompts.py)")
        continue
    check(system == lc.SYSTEM_MESSAGE and user.startswith(variants.HEAD) and lissa_question in user,
          f"variant {name} keeps LiSSA's system message, first sentence, question and artifacts")

# 7. Tolerant parsing and the Holm correction
cases = {"<trace>yes</trace>": True, "Yes, there is a link.": True, "**No**, it is not.": False,
         "Reasoning... <trace>no</trace>": False, "Maybe.": False, "yesterday": False}
check(all(lc.is_related_tolerant(a) == expected for a, expected in cases.items()), "tolerant parsing rules")
import analyze  # noqa: E402

check([round(p, 4) for p in analyze.holm([0.01, 0.04, 0.03])] == [0.03, 0.06, 0.06], "Holm correction")

sys.exit(1 if failures else 0)
