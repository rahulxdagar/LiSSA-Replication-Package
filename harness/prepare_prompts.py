#!/usr/bin/env python3
"""Prepares the inputs of the RQ2 variants and checks that every prompt fits into the context window.

1. Code summaries: data/summaries-<project>.json, {code file: summary}, from javaparse.py (deterministic).
2. Cross-project examples: prompts/examples-<project>.txt, written once and then frozen (--force-examples rewrites).
   For a target project, one linked pair from the first other project and one retrieved-but-unlinked pair from the
   second, drawn with a fixed seed from those projects' dev slices, so no test-slice requirement ever appears in a
   prompt. Only short artifacts qualify, so the examples add little to the prompt. data/examples.json records the pairs.
3. Length check: the longest prompt of each variant over all candidate pairs, in estimated tokens.

    python3 harness/prepare_prompts.py [--candidates data/candidates-rq1.json] [--force-examples]
"""

import argparse
import json
import random
import sys

import evaluate
import javaparse
import lissa_compat as lc
import variants

MAX_REQUIREMENT_CHARS = 1000
MAX_CODE_CHARS = 1500
CHARS_PER_TOKEN = 3.5  # measured 4.0-5.0 on the pilot (prompt_eval_count); 3.5 leaves a margin
ANSWER_RESERVE = 1024  # tokens kept free for the answer (as in RQ1's preflight)


def write_summaries():
    for project in lc.PROJECTS:
        code = lc.load_artifacts(lc.DATASETS / project / "CC")
        result = {name: javaparse.summary(javaparse.parse(text)) for name, text in code.items()}
        (variants.DATA / f"summaries-{project}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
        empty = [name for name, text in result.items() if text.startswith("(no Java")]
        print(f"{project}: {len(result)} summaries; without Java declarations: {empty or 'none'}")


def choose_examples(candidates, split):
    chosen = {}
    for target in lc.PROJECTS:
        others = [p for p in lc.PROJECTS if p != target]
        chosen[target] = []
        for source, linked in ((others[0], True), (others[1], False)):
            requirements = lc.load_artifacts(lc.DATASETS / source / "UC")
            code = lc.load_artifacts(lc.DATASETS / source / "CC")
            gold, dev = lc.load_gold(source), set(split["projects"][source]["dev"])
            pool = [(req, target_file) for req in requirements if lc.link_id(req) in dev
                    for target_file, _ in candidates["projects"][source][req]
                    if ((lc.link_id(req), lc.link_id(target_file)) in gold) == linked
                    and len(requirements[req]) <= MAX_REQUIREMENT_CHARS and len(code[target_file]) <= MAX_CODE_CHARS]
            if not pool:
                sys.exit(f"No {'linked' if linked else 'unlinked'} example in {source} within the length limits.")
            req, target_file = random.Random(f"examples-{target}-{source}").choice(pool)
            chosen[target].append({"project": source, "requirement": req, "code": target_file, "linked": linked,
                                   "requirement_text": requirements[req], "code_text": code[target_file]})
    return chosen


def write_examples(chosen, force):
    record = variants.DATA / "examples.json"
    for target, examples in chosen.items():
        path = variants.PROMPTS / f"examples-{target}.txt"
        if path.exists() and not force:
            print(f"{path.name} exists and is frozen (use --force-examples to rewrite it)")
            continue
        blocks = [f"Example {i}:\nRequirement: '''{e['requirement_text']}'''\nSource code: '''{e['code_text']}'''\n"
                  f"Answer: <trace>{'yes' if e['linked'] else 'no'}</trace>" for i, e in enumerate(examples, 1)]
        path.write_text("\n".join(blocks) + "\n", encoding="utf-8")
        print(f"Wrote {path.name}: " + "; ".join(f"{e['project']} {e['requirement']} / {e['code']} "
                                                 f"({'linked' if e['linked'] else 'not linked'})" for e in examples))
        existing = json.loads(record.read_text()) if record.exists() else {}
        existing[target] = [{k: e[k] for k in ("project", "requirement", "code", "linked")} for e in examples]
        record.write_text(json.dumps(existing, indent=1) + "\n")


def length_check(candidates):
    limit = lc.NUM_CTX - ANSWER_RESERVE
    print(f"\nLongest prompt per variant (estimated at {CHARS_PER_TOKEN} characters per token; limit {limit} tokens):")
    ok = True
    for name, variant in variants.VARIANTS.items():
        longest, where = 0, None
        for project in lc.PROJECTS:
            requirements = lc.load_artifacts(lc.DATASETS / project / "UC")
            code = lc.load_artifacts(lc.DATASETS / project / "CC")
            for req, ranked in candidates["projects"][project].items():
                for target, _ in ranked:
                    system, user = variant(variants.Pair(project, req, target, requirements[req], code[target]))
                    if len(system) + len(user) > longest:
                        longest, where = len(system) + len(user), f"{project} {req} / {target}"
        tokens = round(longest / CHARS_PER_TOKEN)
        ok &= tokens <= limit
        print(f"  {name:11} {longest:6} characters, about {tokens:5} tokens ({where}){'' if tokens <= limit else '  TOO LONG'}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", default=None, help="default: data/candidates-rq1.json, else -local")
    parser.add_argument("--force-examples", action="store_true")
    args = parser.parse_args()
    candidates_file = lc.pathlib.Path(args.candidates) if args.candidates else (
        variants.DATA / "candidates-rq1.json" if (variants.DATA / "candidates-rq1.json").exists()
        else variants.DATA / "candidates-local.json")
    candidates = json.loads(candidates_file.read_text())
    split = json.loads(evaluate.SPLIT_FILE.read_text())
    print(f"Candidates: {candidates_file.name}")
    write_summaries()
    write_examples(choose_examples(candidates, split), args.force_examples)
    if not length_check(candidates):
        sys.exit("Some prompts would not fit into the context window; shorten the added information.")


if __name__ == "__main__":
    main()
