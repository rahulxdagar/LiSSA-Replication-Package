#!/usr/bin/env python3
"""RQ2: asks a local LLM about every candidate pair of a slice with one prompt variant, then scores the links.

The candidates are LiSSA's (retrieval.py), the prompt comes from variants.py, and the Ollama request, answer parsing
and scoring copy LiSSA's (lissa_compat.py). Every run is scored twice: with LiSSA's strict parsing (a link only if
the <trace> tag says yes) and with the tolerant parsing (else a leading yes/no). Unlike LiSSA, every call's Ollama
timing and token counts are kept.

    python3 harness/run.py --variant baseline --model qwen2.5-coder:14b --slice dev
    python3 harness/run.py ... --lissa-cache PATH/cache-local-r2c   # reuse RQ1's answers (same chip as RQ1 only)
    python3 harness/run.py ... --limit 2       # pilot: first 2 requirements of the slice per project
    python3 harness/run.py ... --hours 6       # stop after 6 hours; the same command continues
    python3 harness/run.py ... --dry-run       # counts only; calls nothing

Answers are cached per model and chip in harness/cache/. Outputs: harness/runs/<run id>/ with decisions.csv (one
row per pair), links-<parser>-<project>.csv, metrics.json, metrics.md and run.json.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error

import evaluate
import lissa_compat as lc
import llm
from variants import VARIANTS, Pair, files_used


def lissa_answers(cache_root, model, project):
    """RQ1's answers for the first seed: {key: answer}. Empty if that cache file does not exist."""
    if not cache_root:
        return {}
    path = lc.lissa_cache_file(pathlib.Path(cache_root) / f"{project}-{lc.SEED}", model)
    return lc.read_lissa_cache(path) if path.exists() else {}


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def default_candidates():
    rq1 = lc.HARNESS / "data" / "candidates-rq1.json"
    return rq1 if rq1.exists() else lc.HARNESS / "data" / "candidates-local.json"


def plan(projects, slice_name, limit, candidates, split):
    """The pairs to judge, in LiSSA's order: requirement by file name, then candidates by rank."""
    pairs = []
    for project in projects:
        requirements = lc.load_artifacts(lc.DATASETS / project / "UC")
        code = lc.load_artifacts(lc.DATASETS / project / "CC")
        wanted = evaluate.requirements_in(slice_name, project, split)
        for req in [name for name in requirements if lc.link_id(name) in wanted][:limit]:
            for rank, (target, cosine) in enumerate(candidates["projects"][project][req], 1):
                pairs.append((Pair(project, req, target, requirements[req], code[target]), rank, cosine))
    return pairs


def judged_split(rows):
    """The requirements actually judged (the whole slice, or the first N with --limit), as a one-slice split."""
    judged = {}
    for row in rows:
        judged.setdefault(row["project"], set()).add(lc.link_id(row["requirement"]))
    return {"projects": {p: {"judged": sorted(reqs)} for p, reqs in judged.items()}}


def predicted_links(rows, parser):
    links = {}
    for row in rows:
        links.setdefault(row["project"], set())
        if row[f"related_{parser}"] in (True, "True"):
            links[row["project"]].add((lc.link_id(row["requirement"]), lc.link_id(row["code"])))
    return links


def time_summary(rows):
    """Median and IQR of Ollama's total_duration per call and per requirement (seconds), over timed calls."""
    per_call = [float(row["total_seconds"]) for row in rows if row["total_seconds"] != ""]
    per_req = {}
    for row in rows:
        per_req.setdefault((row["project"], row["requirement"]), []).append(row["total_seconds"])
    per_req = [sum(map(float, values)) for values in per_req.values() if "" not in values]

    def describe(values):
        if len(values) < 2:
            return "not enough timed calls"
        q1, median, q3 = statistics.quantiles(values, n=4)
        return f"median {median:.1f} s (IQR {q1:.1f}-{q3:.1f} s, n = {len(values)})"

    return [f"- Per call: {describe(per_call)}", f"- Per requirement: {describe(per_req)}"]


def write_outputs(run_dir, rows, header, args, device, candidates_file, candidates, split, url):
    with open(run_dir / "decisions.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    judged = judged_split(rows)
    results, report = {}, [header, ""]
    for parser in lc.PARSERS:
        links = predicted_links(rows, parser)
        for project, project_links in links.items():
            (run_dir / f"links-{parser}-{project}.csv").write_text(
                "\n".join(f"{s},{t}" for s, t in sorted(project_links)))
        results[parser] = evaluate.evaluate(links, "judged", judged)
        report += [f"## {parser.capitalize()} parsing", "", evaluate.table(results[parser]), ""]
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=1) + "\n")
    sources = {s: sum(row["source"] == s for row in rows) for s in ("ollama", "harness-cache", "lissa-rq1")}
    report += [f"- Pairs: {len(rows)}; answers from {sources}",
               f"- Without a <trace> tag: {sum(not row['trace_tag'] for row in rows)}; "
               f"cut at {lc.NUM_PREDICT} answer tokens: {sum(row['done_reason'] == 'length' for row in rows)}",
               *time_summary(rows), ""]
    (run_dir / "metrics.md").write_text("\n".join(report))
    try:
        version = llm.ollama(url, "/api/version", timeout=10).get("version")
        digest = {m["name"]: m.get("digest") for m in llm.ollama(url, "/api/tags", timeout=30)["models"]}.get(args.model)
    except (urllib.error.URLError, TimeoutError, OSError):
        version = digest = None
    (run_dir / "run.json").write_text(json.dumps({
        "args": {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items()},
        "options": llm.OPTIONS, "ollama_version": version, "model_digest": digest, "device": device,
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
        "candidates": {"file": str(candidates_file), "sha256": sha256(candidates_file), "source": candidates["source"]},
        "split_sha256": sha256(evaluate.SPLIT_FILE) if split else None,
        "variant_files_sha256": {f: sha256(lc.HARNESS / f) for f in files_used(args.variant, args.projects)},
        "code_sha256": {name: sha256(lc.HARNESS / name)
                        for name in ("lissa_compat.py", "llm.py", "variants.py", "run.py", "javaparse.py")},
    }, indent=1) + "\n")
    print("\n".join(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=list(VARIANTS), default="baseline")
    parser.add_argument("--model", required=True, help="e.g. qwen2.5-coder:14b or llama3.1:8b")
    parser.add_argument("--slice", choices=["dev", "test", "all"], required=True)
    parser.add_argument("--projects", nargs="+", choices=lc.PROJECTS, default=list(lc.PROJECTS))
    parser.add_argument("--candidates", type=pathlib.Path, help="default: data/candidates-rq1.json, else -local")
    parser.add_argument("--lissa-cache", help="RQ1's cache-local-r2c folder, to reuse its answers")
    parser.add_argument("--limit", type=int, help="only the first N requirements of the slice per project")
    parser.add_argument("--hours", type=float, help="stop after this many hours; the same command continues")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--device", help="name of the chip for the cache file (default: detected)")
    parser.add_argument("--run-id", help="output folder name (default: <variant>__<model>__<device>__<slice>)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    candidates_file = args.candidates or default_candidates()
    candidates = json.loads(candidates_file.read_text())
    split = json.loads(evaluate.SPLIT_FILE.read_text()) if args.slice != "all" else None
    if candidates["source"]["kind"] != "lissa-cache":
        print(f"Note: {candidates_file.name} was not computed from RQ1's embeddings; use it for pilots only.")
    device = args.device or llm.device_name()
    cache = llm.AnswerCache(args.model, device)
    variant = VARIANTS[args.variant]
    rq1 = {project: lissa_answers(args.lissa_cache, args.model, project) for project in args.projects}
    keyed = [(pair, rank, cosine, *variant(pair)) for pair, rank, cosine in plan(
        args.projects, args.slice, args.limit, candidates, split)]
    keyed = [(pair, rank, cosine, system, user, llm.call_key(system, user)) for pair, rank, cosine, system, user in keyed]
    missing = sum(1 for pair, *_, key in keyed if key not in rq1[pair.project] and not cache.get(key))
    print(f"Device: {device}. {len(keyed)} pairs; {len(keyed) - missing} already answered; "
          f"{missing} to ask {args.model}.")
    if args.dry_run or not keyed:
        return

    run_dir = lc.HARNESS / "runs" / (args.run_id or f"{args.variant}__{lc.model_tag(args.model)}__{device}__"
                                     f"{args.slice}" + (f"-first{args.limit}" if args.limit else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    if missing:
        pulled = {m["name"] for m in llm.ollama(args.ollama_url, "/api/tags", timeout=30)["models"]}
        if args.model not in pulled:
            sys.exit(f"{args.model} is not pulled (ollama pull {args.model}).")
        if shutil.which("caffeinate"):
            subprocess.Popen(["caffeinate", "-is", "-w", str(os.getpid())])
    deadline = time.monotonic() + args.hours * 3600 if args.hours else None

    rows, asked, started = [], 0, time.monotonic()
    try:
        for index, (pair, rank, cosine, system, user, key) in enumerate(keyed, 1):
            # RQ1's own answer first: for the baseline prompt it is LiSSA's answer
            if key in rq1[pair.project]:
                record, source = {"answer": rq1[pair.project][key]}, "lissa-rq1"
            else:
                if deadline and not cache.get(key) and time.monotonic() >= deadline:
                    print(f"Time limit reached after {asked} calls; run the same command to continue.")
                    return
                record = cache.ask(args.ollama_url, system, user, kind="classify", variant=args.variant,
                                   project=pair.project, requirement=pair.requirement, code=pair.code)
                source = "ollama" if record.get("fresh") else "harness-cache"
            if source == "ollama":
                asked += 1
                left = sum(1 for p, *_, k in keyed[index:] if k not in rq1[p.project] and not cache.get(k))
                eta = left * (time.monotonic() - started) / asked / 3600
                print(f"[{index}/{len(keyed)}] {pair.project} {pair.requirement} #{rank} {pair.code}: "
                      f"{'yes' if lc.is_related_tolerant(record['answer']) else 'no'}"
                      f"{'' if lc.has_trace_tag(record['answer']) else ' (no tag)'} "
                      f"({record['wall_seconds']:.1f} s; about {eta:.1f} h left)", flush=True)
            total = record.get("total_duration")
            rows.append({"project": pair.project, "requirement": pair.requirement, "code": pair.code, "rank": rank,
                         "cosine": cosine, "key": key,
                         **{f"related_{name}": rule(record["answer"]) for name, rule in lc.PARSERS.items()},
                         "trace_tag": lc.has_trace_tag(record["answer"]), "source": source,
                         "done_reason": record.get("done_reason", ""),
                         "prompt_eval_count": record.get("prompt_eval_count", ""),
                         "eval_count": record.get("eval_count", ""),
                         "total_seconds": "" if total is None else total / 1e9})
    except KeyboardInterrupt:
        sys.exit(f"\nInterrupted after {asked} calls; answers so far are cached. Run the same command to continue.")

    header = (f"# {args.variant} with {args.model} on {device}, {args.slice} slice"
              + (f" (first {args.limit} per project)" if args.limit else ""))
    write_outputs(run_dir, rows, header, args, device, candidates_file, candidates, split, args.ollama_url)
    print(f"Run folder: {run_dir}")


if __name__ == "__main__":
    main()
