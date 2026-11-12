#!/usr/bin/env python3
"""RQ1 of "Verify Before You Link": LiSSA with local Ollama models, compared with the published results.

Scope, taken from the proposal (RQ1, Figure 1, Analysis Procedure) and nothing beyond it:
  * LLMs:        qwen2.5-coder:14b and llama3.1:8b, each as its own series, no fallback; temperature 0
  * Embeddings:  bge-m3
  * Projects:    SMOS, eTour, iTrust; whole artifacts on both sides, top-20 retrieval
  * Classifier:  retrieval-only (recall ceiling), KISS prompt, CoT prompt
  * Metrics:     P, R, F1, F2 per project, average and link-weighted average,
                 next to the published GPT-4o, GPT-4o-mini, VSM, LSI and FTLR results

Repetitions follow the LiSSA paper: KISS and retrieval-only run once, CoT runs with the 5 significance seeds.
Runs are ordered so that the first pass (one seed) is already a complete result set; later seeds only add repetitions.

Usage (from anywhere; only the standard library is needed):
    python3 run_rq1_benchmark.py --dry-run      # plan only: calls neither Ollama nor LiSSA, writes nothing
    python3 run_rq1_benchmark.py --smoke        # a few requirements per project: checks the wiring, prints an ETA
    python3 run_rq1_benchmark.py                # full RQ1; run the same command again to resume
    python3 run_rq1_benchmark.py --seeds 1      # stop after the first pass
    python3 run_rq1_benchmark.py --hours 6      # run in batches: stop after 6 hours, the same command continues
    python3 run_rq1_benchmark.py --report-only  # rebuild the report from the results so far, and show the progress

Outputs go to LiSSA-RATLR-V2/lissa/results-local/rq1/ (rq1-smoke/ for --smoke); rq1-report.md is rebuilt after every
config. The answers are cached in cache-local-r2c/ (cache-local-smoke/). The GPT-4o results are only read.

Batches do not change the results: every LLM call is independent of the others, and a config that is stopped halfway
is finished from its cached answers. Results only appear in the report once a config is complete.
"""

import argparse
import atexit
import copy
import csv
import datetime
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

sys.dont_write_bytecode = True  # keeps local/ free of __pycache__ when its modules are imported below

ROOT = pathlib.Path(__file__).resolve().parent
LISSA = ROOT / "LiSSA-RATLR-V2" / "lissa"
UPSTREAM = ROOT.parent / "Replication-Package-ICSE25_LiSSA-Toward-Generic-Traceability-Link-Recovery-through-RAG"
JAR = "target/ratlr-0.2.0-SNAPSHOT-jar-with-dependencies.jar"
PUBLISHED_WORKBOOK = ROOT / "Evaluation-Req2Code.xlsx"

LLMS = ["qwen2.5-coder:14b", "llama3.1:8b"]
EMBEDDING = "bge-m3"
NUM_CTX = 16384
EMBEDDING_NUM_CTX = 8192
TOP_K = 20
# Dataset -> (requirements, code files, gold-standard links), as stated in the proposal
PROJECTS = {"SMOS": (67, 100, 1044), "eTour_en": (58, 116, 308), "iTrust": (131, 226, 286)}
PROMPT_LABELS = {"simple": "KISS", "reasoning": "CoT"}
CACHE_ROOTS = {False: "cache-local-r2c", True: "cache-local-smoke"}


def load_local_module(name):
    spec = importlib.util.spec_from_file_location(name, LISSA / "local" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = load_local_module("generate_configs_local")
summarize = load_local_module("summarize")
tag = generator.model_tag

ollama_server = None
driver_log = None


def echo(message=""):
    print(message, flush=True)
    if driver_log:
        driver_log.write(message + "\n")
        driver_log.flush()


def stream(command, log, deadline=None, **kwargs):
    """Runs a command, showing its output and appending it to log. Returns the exit code.
    At the deadline (time.monotonic) the command gets SIGTERM; LiSSA saves its cache on that, as it does on Ctrl-C."""
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", **kwargs
    )
    timer = None
    if deadline is not None:
        timer = threading.Timer(max(0, deadline - time.monotonic()), process.terminate)
        timer.daemon = True
        timer.start()
    try:
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
    except KeyboardInterrupt:
        # Ctrl-C reaches the command as well; it has to finish saving its cache before the report reads it
        process.wait()
        raise
    finally:
        if timer:
            timer.cancel()
        log.flush()
    return process.wait()


# --- Environment ---------------------------------------------------------------------------------------------------


def java_major(home):
    try:
        output = subprocess.run([f"{home}/bin/java", "-version"], capture_output=True, text=True).stderr
    except OSError:
        return None
    match = re.search(r'version "(\d+)', output)
    return int(match.group(1)) if match else None


def find_java_21():
    registered = subprocess.run(["/usr/libexec/java_home", "-v", "21"], capture_output=True, text=True).stdout.strip()
    for candidate in (
        os.environ.get("JAVA_HOME"),
        registered,
        "/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home",
        "/usr/local/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home",
    ):
        if candidate and java_major(candidate) == 21:
            return candidate
    return None


def jar_is_current():
    jar = LISSA / JAR
    if not jar.exists():
        return False
    sources = [LISSA / "pom.xml", *(p for p in (LISSA / "src").rglob("*") if p.is_file())]
    return jar.stat().st_mtime >= max(p.stat().st_mtime for p in sources)


def model_is_pulled(model):
    name, _, version = model.partition(":")
    manifests = pathlib.Path.home() / ".ollama/models/manifests/registry.ollama.ai/library"
    return (manifests / name / (version or "latest")).exists()


def hidden_dataset_files():
    # LiSSA loads every regular file of UC/ and CC/ as an artifact, so a Finder .DS_Store would become a requirement
    return [
        path
        for project in PROJECTS
        for folder in ("UC", "CC")
        for path in (LISSA / "datasets" / "req2code" / project / folder).iterdir()
        if path.name.startswith(".")
    ]


def on_battery():
    try:
        return "Battery Power" in subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    except OSError:
        return False


def keep_awake():
    """Holds off sleep until this script exits. LiSSA is not wrapped in caffeinate, so signals reach it directly."""
    if shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-is", "-w", str(os.getpid())])


def ollama_request(url, path, timeout=10):
    with urllib.request.urlopen(url.rstrip("/") + path, timeout=timeout) as response:
        return json.loads(response.read().decode())


def ollama_is_up(url):
    try:
        ollama_request(url, "/api/version", timeout=5)
        return True
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False


def stop_ollama_server():
    if ollama_server and ollama_server.poll() is None:
        ollama_server.terminate()


def ensure_ollama(url, run_dir):
    """Uses a running Ollama, otherwise starts 'ollama serve' with its log in the run folder."""
    global ollama_server
    if ollama_is_up(url):
        return
    if not shutil.which("ollama"):
        sys.exit("The ollama command was not found. Install Ollama or start it manually.")
    if not ollama_server or ollama_server.poll() is not None:
        echo(f">> Ollama is not reachable at {url}: starting 'ollama serve'")
        log = open(run_dir / "ollama-serve.log", "ab")
        env = {**os.environ, "OLLAMA_HOST": url}
        ollama_server = subprocess.Popen(["ollama", "serve"], env=env, stdout=log, stderr=subprocess.STDOUT)
        atexit.register(stop_ollama_server)
    for _ in range(60):
        time.sleep(1)
        if ollama_is_up(url):
            return
    sys.exit(f"Ollama did not come up at {url}; see {run_dir / 'ollama-serve.log'}.")


def pull_missing_models(url, llms):
    local = {model["name"] for model in ollama_request(url, "/api/tags").get("models", [])}
    for model in [*llms, EMBEDDING]:
        if model not in local and f"{model}:latest" not in local:
            echo(f">> ollama pull {model}")
            subprocess.run(["ollama", "pull", model], env={**os.environ, "OLLAMA_HOST": url}, check=True)


def build_jar(java_home):
    if jar_is_current():
        return
    echo(">> Building LiSSA (mvn -DskipTests package)")
    if not shutil.which("mvn"):
        sys.exit("mvn was not found. Install it with: brew install maven")
    env = {**os.environ, "JAVA_HOME": java_home}
    if stream(["mvn", "-B", "-q", "-DskipTests", "package"], driver_log, cwd=LISSA, env=env) != 0:
        sys.exit("The Maven build failed, nothing was run.")


def preflight(url, llm, run_dir, allow_partial_gpu):
    """local/preflight.py with the fallbacks switched off: the proposal fixes the models of each series."""
    probe = run_dir / f".preflight-{tag(llm)}.json"
    command = [
        sys.executable, "local/preflight.py", "--ollama-url", url, "--num-ctx", str(NUM_CTX), "--out", str(probe),
        "--llm", llm, "--llm-fallback", "", "--embedding", EMBEDDING, "--embedding-fallback", "",
        "--probe-source-dirs", *[f"datasets/req2code/{p}/UC" for p in PROJECTS],
        "--probe-target-dirs", *[f"datasets/req2code/{p}/CC" for p in PROJECTS],
    ]
    if allow_partial_gpu:
        command.append("--allow-partial-gpu")
    echo(f">> Preflight for {llm} (loads the models with num_ctx {NUM_CTX} and checks GPU placement)")
    if stream(command, driver_log, cwd=LISSA) != 0:
        sys.exit(f"Preflight failed for {llm}, see the reasons above. Nothing further was run.")

    card = json.loads(probe.read_text())
    quantization = card["llm"].get("details", {}).get("quantization_level", "")
    if not quantization.startswith("Q4"):
        sys.exit(f"{llm} is quantised as '{quantization}', but the proposal specifies 4-bit models.")
    target = run_dir / f"preflight-{tag(llm)}.json"
    if target.exists():
        first = json.loads(target.read_text())
        chosen = (first["llm"]["digest"], first["embedding"]["digest"])
        if chosen != (card["llm"]["digest"], card["embedding"]["digest"]):
            sys.exit(f"{run_dir} was started with other model digests than the ones pulled now. Use a new --run-id.")
        probe.replace(run_dir / "logs" / f"preflight-{tag(llm)}-{datetime.datetime.now():%Y%m%d-%H%M%S}.json")
    else:
        probe.replace(target)
    return card


def write_provenance(run_dir, java_home):
    jar_hash = hashlib.sha256((LISSA / JAR).read_bytes()).hexdigest()
    git = ["git", "-C", str(ROOT)]
    version = subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip()
    java = subprocess.run([f"{java_home}/bin/java", "-version"], capture_output=True, text=True).stderr.splitlines()[0]
    status = subprocess.run([*git, "status", "--short"], capture_output=True, text=True).stdout
    commit = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (run_dir / "code-version.txt").write_text(
        f"commit: {commit}\njar sha256: {jar_hash}\njava: {java}\nollama: {version}\n\ngit status:\n{status}"
    )
    # The local changes are committed in this repository, so 'git diff' cannot show them; compare with the original
    upstream = UPSTREAM / "LiSSA-RATLR-V2" / "lissa"
    if upstream.is_dir():
        with open(run_dir / "lissa-vs-upstream.diff", "w") as diff:
            for part in ("src", "pom.xml"):
                subprocess.run(["diff", "-ru", str(upstream / part), str(LISSA / part)], stdout=diff)


# --- Plan ----------------------------------------------------------------------------------------------------------


def plan_runs(llms, seed_count, llm_ctx=None, embedding_ctx=EMBEDDING_NUM_CTX):
    """Returns the configs in run order; the first pass covers every approach, later seeds only repeat CoT."""
    seeds = generator.SEEDS[:seed_count]
    retrieval, cot = {}, {}
    for llm in llms:
        ctx = (llm_ctx or {}).get(llm, NUM_CTX)
        for name, config, project, seed in generator.generate("req2code", llm, EMBEDDING, ctx, embedding_ctx, True):
            calls = PROJECTS[project][0] * TOP_K
            if config["classifier"]["name"] == "mock":
                retrieval[name] = {"name": name, "config": config, "llm": None, "prompt": "retrieval only",
                                   "seed": None, "project": project, "calls": 0}
            else:
                cot[llm, project, seed] = {"name": name, "config": config, "llm": llm, "prompt": "CoT",
                                           "seed": seed, "project": project, "calls": calls}

    def kiss(llm, project):
        run = copy.deepcopy(cot[llm, project, seeds[0]])
        run["config"]["classifier"]["name"] = "simple_ollama"
        run["name"] = f"{project}_artifact_artifact_simple_ollama_{tag(llm)}_{tag(EMBEDDING)}.json"
        run["prompt"] = "KISS"
        return run

    runs = list(retrieval.values())
    for llm in llms:
        runs += [cot[llm, project, seeds[0]] for project in PROJECTS]
        runs += [kiss(llm, project) for project in PROJECTS]
    for seed in seeds[1:]:
        for llm in llms:
            runs += [cot[llm, project, seed] for project in PROJECTS]
    return runs


def shrink_to_smoke(runs, run_dir, requirements):
    """Points the configs to a few linked requirements per project; the code side and retrieval stay as they are."""
    for project in PROJECTS:
        dataset = LISSA / "datasets" / "req2code" / project
        target = run_dir / "datasets" / "req2code" / project
        (target / "UC").mkdir(parents=True, exist_ok=True)
        gold = [line.strip() for line in (dataset / "answer.csv").read_text().splitlines() if line.strip()]
        linked = {line.split(",")[0] for line in gold}
        chosen = [p for p in sorted((dataset / "UC").iterdir()) if p.stem in linked][:requirements]
        for path in chosen:
            shutil.copyfile(path, target / "UC" / path.name)
        stems = {path.stem for path in chosen}
        (target / "answer.csv").write_text("\n".join(l for l in gold if l.split(",")[0] in stems) + "\n")
    for run in runs:
        config, project = run["config"], run["project"]
        relative = f"./{(run_dir / 'datasets' / 'req2code' / project).relative_to(LISSA)}"
        config["source_artifact_provider"]["args"]["path"] = f"{relative}/UC"
        config["gold_standard_configuration"]["path"] = f"{relative}/answer.csv"
        config["cache_dir"] = f"./{CACHE_ROOTS[True]}/{pathlib.PurePath(config['cache_dir']).name}"
        run["calls"] = requirements * TOP_K if run["llm"] else 0


def print_plan(runs):
    for index, run in enumerate(runs, 1):
        seed = f"seed {run['seed']}" if run["seed"] else ""
        echo(f"  {index:>3}. {run['project']:<9} {run['llm'] or EMBEDDING:<18} {run['prompt']:<15} {seed}")
    calls = sum(run["calls"] for run in runs)
    first_pass = sum(run["calls"] for run in runs if run["prompt"] == "KISS" or run["seed"] == generator.SEEDS[0])
    # Pairs with identical text share one cached answer, so the actual number of calls is slightly lower
    echo(f"  {len(runs)} configs, at most {calls:,} LLM calls ({first_pass:,} in the first pass)")


# --- Running -------------------------------------------------------------------------------------------------------


def is_done(run_dir, name):
    return any(run_dir.glob(f"results-{name}_*.md"))


def run_config(run, index, total, run_dir, java_home, url, retries, deadline=None):
    """Returns DONE (nothing to do), OK, FAILED or PAUSED (out of time; the answers so far are cached)."""
    name = run["name"]
    config_path = run_dir / "configs" / name
    config_path.write_text(json.dumps(run["config"], indent=2) + "\n")
    if is_done(run_dir, name):
        echo(f"== [{index}/{total}] {name} already done")
        return "DONE"
    if deadline is not None and time.monotonic() >= deadline:
        return "PAUSED"

    env = {**os.environ, "JAVA_HOME": java_home, "OLLAMA_HOST": url, "OLLAMA_EMBEDDING_HOST": url}
    command = [f"{java_home}/bin/java", "-jar", JAR, "eval", "-c", str(config_path.relative_to(LISSA))]
    start = time.time()
    status, attempt = "FAILED", 0
    while attempt <= retries and status != "OK":
        attempt += 1
        ensure_ollama(url, run_dir)
        # LiSSA writes into its working directory and does not truncate existing files: move leftovers away
        for leftover in [*LISSA.glob(f"results-{name}_*.md"), *LISSA.glob(f"traceLinks-{name}_*.csv")]:
            leftover.replace(run_dir / "logs" / f"stale-{int(time.time())}-{leftover.name}")
        echo(f">> [{index}/{total}] {name} (attempt {attempt})")
        with open(run_dir / "logs" / f"{config_path.stem}.log", "a") as log:
            exit_code = stream(command, log, deadline, cwd=LISSA, env=env)
        if exit_code != 0 and deadline is not None and time.monotonic() >= deadline:
            # Stopped by the time limit. A result file written at that moment may be incomplete, so it is not trusted:
            # the next batch moves it away and rebuilds it from the cache
            status = "PAUSED"
            break
        # LiSSA only logs exceptions and exits with 0, so the result file is what tells success from failure
        for result in [*LISSA.glob(f"results-{name}_*.md"), *LISSA.glob(f"traceLinks-{name}_*.csv")]:
            if result.name.startswith("results-"):
                status = "OK"
            result.replace(run_dir / result.name)
        if status != "OK":
            echo(f"!! {name} produced no result, see {run_dir / 'logs' / config_path.stem}.log")

    with open(run_dir / "runs.tsv", "a") as runs_file:
        runs_file.write(f"{name}\t{status}\t{attempt}\t{int(time.time() - start)}\n")
    try:
        with open(run_dir / "ollama-ps.jsonl", "a") as ps_file:
            ps_file.write(json.dumps({"config": name, **ollama_request(url, "/api/ps")}) + "\n")
    except (urllib.error.URLError, TimeoutError, ValueError):
        pass
    return status


def answered_pairs(runs, run_dir):
    """Candidate pairs with an answer: all of a finished config, otherwise what its cache holds so far."""
    answered = 0
    for run in runs:
        if not run["llm"]:
            continue
        if is_done(run_dir, run["name"]):
            answered += run["calls"]
            continue
        classifier = "ReasoningClassifier" if run["prompt"] == "CoT" else "SimpleClassifier"
        cache = LISSA / run["config"]["cache_dir"] / f"{classifier}_{run['llm'].replace(':', '__')}.json"
        if cache.exists():
            answered += min(run["calls"], len(json.loads(cache.read_text())))
    return answered


def progress_line(runs, run_dir):
    done = sum(1 for run in runs if is_done(run_dir, run["name"]))
    total = sum(run["calls"] for run in runs)
    answered = answered_pairs(runs, run_dir)
    return f"Configs finished: {done} of {len(runs)}. Candidate pairs answered: {answered:,} of {total:,} ({answered / total:.0%})."


def pace_line(runs, run_dir, answered_before, started):
    """Remaining running time at this batch's pace. KISS is faster than CoT, so this is only a rough guide."""
    answered = answered_pairs(runs, run_dir)
    if answered <= answered_before:
        return "No new answers in this batch, so there is no time estimate."
    hours_per_pair = (time.monotonic() - started) / 3600 / (answered - answered_before)
    remaining = sum(run["calls"] for run in runs) - answered
    return f"At this batch's pace, about {remaining * hours_per_pair:.1f} hours of running remain."


# --- Report --------------------------------------------------------------------------------------------------------


def read_sheet(workbook, sheet):
    """Stored cell values of one worksheet as {"C9": "0.325"}; the workbook is only read."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_id = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    with zipfile.ZipFile(workbook) as archive:
        shared = [
            "".join(t.text or "" for t in si.iter(f"{{{ns['m']}}}t"))
            for si in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("m:si", ns)
        ]
        targets = {r.get("Id"): r.get("Target") for r in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))}
        sheets = ET.fromstring(archive.read("xl/workbook.xml")).find("m:sheets", ns)
        target = next(targets[s.get(rel_id)] for s in sheets if s.get("name") == sheet).lstrip("/")
        root = ET.fromstring(archive.read(target if target.startswith("xl/") else f"xl/{target}"))
    cells = {}
    for cell in root.iter(f"{{{ns['m']}}}c"):
        value = cell.find("m:v", ns)
        if value is not None and value.text is not None:
            cells[cell.get("r")] = shared[int(value.text)] if cell.get("t") == "s" else value.text
    return cells


def published_results():
    """The published rows as single runs. Their averages are recomputed over the 3 projects (the paper's include
    the Dronology projects, which cannot be run here)."""
    gpt4o, mini = "Req2Code (GPT4o)", "Req2Code (GPT4o mini)"
    # (sheet, row, label column, expected label, name in the report)
    rows = [
        (gpt4o, 9, "B", "No LLM", "retrieval only (text-embedding-3-large)"),
        (gpt4o, 10, "B", "KISS", "GPT-4o KISS"),
        (gpt4o, 11, "B", "Chain of Thought", "GPT-4o CoT (original run)"),
        (mini, 10, "B", "KISS", "GPT-4o-mini KISS"),
        (mini, 11, "B", "Chain of Thought", "GPT-4o-mini CoT"),
        (gpt4o, 3, "A", "VSM", "VSM (opt.)"),
        (gpt4o, 4, "A", "LSI", "LSI (opt.)"),
        (gpt4o, 6, "A", "FTLR (Best)", "FTLR"),
        (gpt4o, 7, "A", "FTLR (Opt)", "FTLR (opt.)"),
    ]
    columns = {"SMOS": "CDEF", "eTour_en": "GHIJ", "iTrust": "KLMN"}
    sheets = {sheet: read_sheet(PUBLISHED_WORKBOOK, sheet) for sheet in (gpt4o, mini)}
    results = []
    for sheet, row, label_column, expected, name in rows:
        label = sheets[sheet].get(f"{label_column}{row}")
        if label != expected:
            raise ValueError(f"{PUBLISHED_WORKBOOK.name}, {sheet}, row {row}: expected '{expected}', found '{label}'")
        for project, cols in columns.items():
            values = [float(sheets[sheet][f"{col}{row}"]) for col in cols]
            metrics = dict(zip(("p", "r", "f1", "f2"), values))
            results.append({**metrics, "approach": name, "project": project, "gs": PROJECTS[project][2]})
        if name == "GPT-4o CoT (original run)":
            # The 5-seed runs of the significance tests, from LiSSA's own result files
            pattern = "results-*_artifact_artifact_reasoning_gpt_gpt-4o-2024-05-13*.md"
            for path in sorted((LISSA / "results-r2c").glob(pattern)):
                result = summarize.parse(path)
                if result["project"] in PROJECTS:
                    results.append({**result, "approach": "GPT-4o CoT (5 seeds)"})
    return results


def local_results(run_dir):
    results = []
    for path in sorted(run_dir.glob("results-*.md")):
        result = summarize.parse(path)
        if result["mode"] != "mock":
            # summarize.py groups by model only, which would average the KISS and CoT runs of an LLM into one row
            result["approach"] = f"{result['approach'].split(' + ')[0]} {PROMPT_LABELS[result['mode']]}"
        results.append(result)
    return results


def format_table(rows, scopes):
    names = {"eTour_en": "eTour"}
    index = {(row["approach"], row["project"]): row for row in rows}
    approaches = list(dict.fromkeys(row["approach"] for row in rows))
    lines = [
        "| Approach | " + " | ".join(f"{names.get(s, s)} {m}" for s in scopes for m in ("P", "R", "F1", "F2")) + " |",
        "|---|" + "---|" * (4 * len(scopes)),
    ]
    for approach in approaches:
        cells = []
        for scope in scopes:
            row = index.get((approach, scope))
            cells += [f"{row[m]:.3f}" if row else "—" for m in ("p", "r", "f1", "f2")]
        lines.append(f"| {approach} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_counts(rows):
    lines = ["| Approach | SMOS | eTour | iTrust | F1 sd (SMOS / eTour / iTrust) |", "|---|---|---|---|---|"]
    index = {(row["approach"], row["project"]): row for row in rows}
    for approach in dict.fromkeys(row["approach"] for row in rows):
        project_rows = [index.get((approach, project)) for project in PROJECTS]
        counts = " | ".join(str(row["runs"]) if row else "0" for row in project_rows)
        spread = " / ".join(f"{row['f1_sd']:.4f}" if row and not math.isnan(row["f1_sd"]) else "—" for row in project_rows)
        lines.append(f"| {approach} | {counts} | {spread} |")
    return "\n".join(lines)


def integrity_checks(run_dir, smoke):
    """Compares what LiSSA reports for each run with the numbers stated in the proposal."""
    if smoke:
        return ["- Dataset counts are not checked in a smoke run."]
    patterns = (r"# Source Artifacts: (\d+)", r"# Target Artifacts: (\d+)", r"# TraceLinks \(GS\): (\d+)")
    problems, candidates = [], {}
    for path in sorted(run_dir.glob("results-*.md")):
        text = path.read_text()
        result = summarize.parse(path)
        counts = tuple(int(re.search(pattern, text).group(1)) for pattern in patterns)
        if counts != PROJECTS[result["project"]]:
            problems.append(f"- Mismatch in {path.name}: {counts}, expected {PROJECTS[result['project']]}")
        if result["mode"] == "mock":
            found = (re.search(pattern, text).group(1) for pattern in (r"True Positives: (\d+)", r"False Positives: (\d+)"))
            candidates[result["project"]] = sum(map(int, found))
    lines = problems or ["- Requirements, code files and gold links of every run match the proposal."]
    if candidates:
        expected = {project: PROJECTS[project][0] * TOP_K for project in candidates}
        verdict = "as expected" if candidates == expected else f"expected {expected}"
        lines.append(f"- Retrieval-only candidate pairs: {sum(candidates.values()):,} {candidates} ({verdict}).")
    return lines


def answer_checks(llms, cache_root):
    """Counts answers that LiSSA's unchanged parsing decides silently, which could otherwise be mistaken for the
    model's own judgement: CoT looks for the <trace> tag, KISS only looks for the substring 'yes'."""
    lines = ["| LLM | Prompt | Answers | Not parseable as intended |", "|---|---|---|---|"]
    for llm in llms:
        for classifier, prompt in (("ReasoningClassifier", "CoT"), ("SimpleClassifier", "KISS")):
            answers = []
            for path in cache_root.glob(f"*/{classifier}_{llm.replace(':', '__')}.json"):
                answers += json.loads(path.read_text()).values()
            if not answers:
                continue
            if prompt == "CoT":
                missing = sum(1 for a in answers if not re.search(r"<trace>(.*?)</trace>", a, re.IGNORECASE))
                note = f"{missing:,} without a `<trace>` answer (counted as no)"
            else:
                lowered = [a.lower() for a in answers]
                neither = sum(1 for a in lowered if "yes" not in a and not re.search(r"\bno\b", a))
                both = sum(1 for a in lowered if "yes" in a and re.search(r"\bno\b", a))
                note = f"{neither:,} with neither yes nor no (counted as no); {both:,} with both (counted as yes)"
            # Normal answers stay far below this; 4,096 tokens are roughly 12,000-16,000 characters
            cut = sum(1 for a in answers if len(a) > 8000)
            if cut:
                note += f"; {cut:,} over 8,000 characters, probably cut at the 4,096-token answer limit"
            lines.append(f"| {llm} | {prompt} | {len(answers):,} | {note} |")
    return lines if len(lines) > 2 else ["No answers cached yet."]


def model_table(run_dir):
    lines = ["| LLM | Digest | Quantisation | num_ctx | GPU | Longest prompt (tokens) | Fits | Embedding (num_ctx, GPU) | Ollama |",
             "|---|---|---|---|---|---|---|---|---|"]
    for path in sorted(run_dir.glob("preflight-*.json")):
        card = json.loads(path.read_text())
        llm, embedding = card["llm"], card["embedding"]
        lines.append(
            f"| {llm['model']} | {str(llm.get('digest'))[:12]} | {llm['details'].get('quantization_level')} "
            f"| {llm['num_ctx']} | {llm['gpu_fraction']:.0%} | {llm['probe']['longest_prompt_tokens']} "
            f"| {llm['probe']['fits']} | {embedding['model']} ({embedding['num_ctx']}, {embedding['gpu_fraction']:.0%}) "
            f"| {card['ollama_version']} |"
        )
    # Every batch runs its own preflight, so this also shows an Ollama update between batches
    checks = [*run_dir.glob("preflight-*.json"), *run_dir.glob("logs/preflight-*.json")]
    versions = sorted({json.loads(path.read_text())["ollama_version"] for path in checks})
    if len(versions) > 1:
        lines.append(f"\n**The batches of this run used different Ollama versions: {', '.join(versions)}.**")
    return "\n".join(lines) if len(lines) > 2 else "No preflight recorded yet."


def seconds_per_call(run_dir, runs):
    """Measured wall-clock seconds per LLM call for each series, from runs.tsv."""
    seconds, finished = {}, set()
    tsv = run_dir / "runs.tsv"
    if tsv.exists():
        for line in tsv.read_text().splitlines():
            name, status, _, duration = line.split("\t")
            # A config stopped by --hours has one row per batch
            if status in ("OK", "PAUSED"):
                seconds[name] = seconds.get(name, 0) + int(duration)
            if status == "OK":
                finished.add(name)
    measured = {}
    for run in runs:
        if run["llm"] and run["name"] in finished:
            total = measured.setdefault((run["llm"], run["prompt"]), [0, 0])
            total[0] += seconds[run["name"]]
            total[1] += run["calls"]
    return {key: spent / calls for key, (spent, calls) in measured.items() if calls}


def eta_lines(run_dir, runs, llms):
    """Scales the smoke run's timing to the full benchmark."""
    measured = seconds_per_call(run_dir, runs)
    full_runs = plan_runs(llms, len(generator.SEEDS))
    lines = ["| LLM | Prompt | s/call (measured) | Full calls | Hours |", "|---|---|---|---|---|"]
    first, total = 0.0, 0.0
    for (llm, prompt), rate in measured.items():
        series = [run for run in full_runs if run["llm"] == llm and run["prompt"] == prompt]
        calls = sum(run["calls"] for run in series)
        lines.append(f"| {llm} | {prompt} | {rate:.1f} | {calls:,} | {rate * calls / 3600:.1f} |")
        total += rate * calls / 3600
        first += rate * sum(r["calls"] for r in series if r["prompt"] == "KISS" or r["seed"] == generator.SEEDS[0]) / 3600
    if len(lines) > 2:
        lines.append(f"\nEstimated total: **{total:.0f} h** for 5 CoT seeds, **{first:.0f} h** with `--seeds 1`.")
    return lines


def write_report(run_dir, runs, llms, smoke):
    local = local_results(run_dir)
    published = published_results()
    projects = list(PROJECTS)
    rows = summarize.aggregate(local, projects) + summarize.aggregate(published, projects)
    # Only the ordering of the rows: retrieval only, then per LLM KISS before CoT, as in the paper's tables
    def order(row):
        name = row["approach"]
        for position, llm in enumerate(LLMS):
            if name.startswith(llm):
                return (1 + position, 0 if name.endswith("KISS") else 1)
        return (0, 0) if name == f"retrieval only ({EMBEDDING})" else (len(LLMS) + 1, 0)

    rows = sorted(rows, key=order)
    parts = ["# RQ1: LiSSA with local LLMs vs. the published results", ""]
    if smoke:
        parts += ["> **Smoke test on a few requirements. These numbers only show that the pipeline works; they are not"
                  " results.**", ""]
    parts += [
        f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M}. {progress_line(runs, run_dir)}",
        "",
        "Pipeline: LiSSA V2 unchanged in prompts, retrieval (top 20), aggregation and metrics; whole artifacts; bge-m3;"
        " temperature 0; SMOS, eTour and iTrust. Values with several runs are means over seeds. Ollama client settings"
        " (see `lissa-vs-upstream.diff`): context 16,384 tokens, embedding inputs cut at 8,192 tokens, answers capped at"
        " 4,096 tokens.",
        "",
        "## Averages over the three projects",
        "",
        format_table(rows, ["Avg.", "w. Avg."]),
        "",
        "The paper's averages also include the two Dronology projects (for example the 0.32 average F1 of GPT-4o CoT in"
        " the proposal), so they are not comparable with these. All published rows here are recomputed over SMOS, eTour"
        " and iTrust with the paper's formulas: mean per project, and mean weighted by gold-standard links. A dash means"
        " that not all three projects have results yet.",
        "",
        "The retrieval-only rows give the recall ceiling of each embedding model.",
        "",
        "## Per project",
        "",
        format_table(rows, projects),
        "",
        "## Runs per approach",
        "",
        run_counts(rows),
        "",
        "## Validity checks",
        "",
        *integrity_checks(run_dir, smoke),
        "",
        *answer_checks(llms, LISSA / CACHE_ROOTS[smoke]),
        "",
        "## Models and hardware evidence",
        "",
        model_table(run_dir),
        "",
        "`ollama-ps.jsonl` has the GPU placement after every config, `lissa-vs-upstream.diff` the code differences from"
        " the original package, `code-version.txt` the commit, jar hash and tool versions.",
        "",
        "Sources of the published rows: `Evaluation-Req2Code.xlsx` (read only) and, for the 5-seed runs,"
        " `LiSSA-RATLR-V2/lissa/results-r2c/`.",
        "",
    ]
    if smoke:
        parts[4:4] = ["## Estimated duration of the full benchmark", "", *eta_lines(run_dir, runs, llms), ""]
    (run_dir / "rq1-report.md").write_text("\n".join(parts))
    with open(run_dir / "rq1-results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["approach", "project", "runs", "p", "r", "f1", "f1_sd", "f2"],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# --- Main ----------------------------------------------------------------------------------------------------------


def dry_run(runs, run_dir, smoke, args):
    home = find_java_21()
    echo(f"Java 21:   {home or 'NOT FOUND (brew install openjdk@21)'}")
    echo(f"Jar:       {'up to date' if jar_is_current() else 'will be rebuilt (mvn -DskipTests package)'}")
    echo(f"Ollama:    {shutil.which('ollama') or 'NOT FOUND'} (not contacted)")
    for model in [*args.llm, EMBEDDING]:
        echo(f"Model:     {model}: {'pulled' if model_is_pulled(model) else 'will be pulled'}")
    hidden = hidden_dataset_files()
    echo(f"Datasets:  {f'hidden files that LiSSA would load: {hidden}' if hidden else 'no hidden files in UC/ or CC/'}")
    echo(f"Power:     {'ON BATTERY: plug in before a real run' if on_battery() else 'on AC'}")
    echo(f"Output:    {run_dir}")
    echo(f"Cache:     {LISSA / CACHE_ROOTS[smoke]}")
    echo(f"Batches:   {f'stops after {args.hours:g} h; the same command continues' if args.hours else 'runs until finished'}")
    if smoke:
        for run in runs:
            run["calls"] = args.smoke_requirements * TOP_K if run["llm"] else 0
    echo("Plan:")
    print_plan(runs)


def main():
    global driver_log
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="show the plan; calls neither Ollama nor LiSSA")
    parser.add_argument("--smoke", action="store_true", help="only a few requirements per project, with an ETA")
    parser.add_argument("--smoke-requirements", type=int, default=2, help="requirements per project in --smoke")
    parser.add_argument("--report-only", action="store_true", help="rebuild rq1-report.md from the results so far")
    parser.add_argument("--llm", action="append", choices=LLMS, help="only this LLM, e.g. one LLM per laptop (default: both)")
    parser.add_argument("--seeds", type=int, default=5, choices=range(1, 6), help="CoT seeds (default: 5, as in LiSSA)")
    parser.add_argument("--hours", type=float, help="stop after this many hours, even within a config; the same command continues")
    parser.add_argument("--retries", type=int, default=2, help="re-runs of a failed config, resuming from the cache")
    parser.add_argument("--allow-partial-gpu", action="store_true", help="accept models not fully on the GPU")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--run-id", help="output folder name (default: rq1, or rq1-smoke with --smoke)")
    args = parser.parse_args()
    args.llm = [llm for llm in LLMS if llm in (args.llm or LLMS)]
    if args.hours is not None and args.hours <= 0:
        parser.error("--hours must be greater than 0")
    deadline = time.monotonic() + args.hours * 3600 if args.hours else None

    run_dir = LISSA / "results-local" / (args.run_id or ("rq1-smoke" if args.smoke else "rq1"))
    seed_count = 1 if args.smoke else args.seeds
    runs = plan_runs(args.llm, seed_count)

    if args.dry_run:
        dry_run(runs, run_dir, args.smoke, args)
        return
    if args.report_only:
        if not run_dir.is_dir():
            sys.exit(f"{run_dir} does not exist yet.")
        if args.smoke:
            shrink_to_smoke(runs, run_dir, args.smoke_requirements)
        write_report(run_dir, runs, args.llm, args.smoke)
        echo(progress_line(runs, run_dir))
        echo(f"Report: {run_dir / 'rq1-report.md'}")
        return

    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "configs").mkdir(exist_ok=True)
    (LISSA / CACHE_ROOTS[args.smoke]).mkdir(exist_ok=True)
    driver_log = open(run_dir / "driver.log", "a")
    echo(f"\n=== {datetime.datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(sys.argv[1:]) or '(full RQ1)'}")
    keep_awake()

    hidden = hidden_dataset_files()
    if hidden:
        sys.exit("LiSSA would load these hidden files as artifacts, delete them first:\n  " +
                 "\n  ".join(str(path) for path in hidden))
    java_home = find_java_21()
    if not java_home:
        sys.exit("JDK 21 was not found. Install it with: brew install openjdk@21")
    if (LISSA / ".env").exists() and "OLLAMA" in (LISSA / ".env").read_text():
        echo(f"Note: {LISSA / '.env'} sets OLLAMA_* variables, and LiSSA prefers them over {args.ollama_url}.")
    if on_battery():
        echo("Note: this Mac is on battery. caffeinate only prevents system sleep on AC; plug it in and keep the lid open.")

    build_jar(java_home)
    ensure_ollama(args.ollama_url, run_dir)
    pull_missing_models(args.ollama_url, args.llm)
    cards = {llm: preflight(args.ollama_url, llm, run_dir, args.allow_partial_gpu) for llm in args.llm}
    write_provenance(run_dir, java_home)

    llm_ctx = {llm: card["llm"]["num_ctx"] for llm, card in cards.items()}
    runs = plan_runs(args.llm, seed_count, llm_ctx, next(iter(cards.values()))["embedding"]["num_ctx"])
    if args.smoke:
        shrink_to_smoke(runs, run_dir, args.smoke_requirements)
    echo("Plan:")
    print_plan(runs)

    failed, paused = [], False
    answered_before, started = answered_pairs(runs, run_dir), time.monotonic()
    try:
        for index, run in enumerate(runs, 1):
            status = run_config(run, index, len(runs), run_dir, java_home, args.ollama_url, args.retries, deadline)
            if status == "PAUSED":
                paused = True
                break
            if status == "FAILED":
                failed.append(run["name"])
            if status != "DONE":
                write_report(run_dir, runs, args.llm, args.smoke)
    except KeyboardInterrupt:
        write_report(run_dir, runs, args.llm, args.smoke)
        echo(f"\nInterrupted. {progress_line(runs, run_dir)}\n{pace_line(runs, run_dir, answered_before, started)}")
        sys.exit(f"Answers so far are cached; run the same command again to resume. Run folder: {run_dir}")

    write_report(run_dir, runs, args.llm, args.smoke)
    echo(f"\nReport: {run_dir / 'rq1-report.md'}")
    echo(progress_line(runs, run_dir))
    if paused:
        echo(f"The {args.hours:g} h limit is reached and the answers so far are saved. It is safe to shut down.")
        echo(pace_line(runs, run_dir, answered_before, started))
        echo("Run the same command again to continue.")
    if args.smoke and not paused:
        echo("\n".join(eta_lines(run_dir, runs, args.llm)))
    if failed:
        sys.exit(f"{len(failed)} config(s) failed: {', '.join(failed)}\nRun the same command again to retry them.")


if __name__ == "__main__":
    main()
