#!/usr/bin/env python3
"""Checks that the harness reproduces LiSSA's RQ1 runs exactly, using Rahul's RQ1 outputs.

1. Candidates: LiSSA's retrieval recomputed from RQ1's cached embeddings must give exactly the pairs of RQ1's
   retrieval-only runs. If so, harness/data/candidates-rq1.json is written (the candidates for RQ2 and RQ3).
2. Prompts: for every candidate pair, the baseline prompt's cache key must be found in RQ1's CoT answer cache.
   A hit proves that the harness builds LiSSA's prompt byte for byte.
3. Decisions: parsing those cached answers must give exactly RQ1's CoT trace links and LiSSA's P/R/F1.
4. Optional (--reask N): N cached pairs are asked again on this machine; reports how often the answer text and the
   yes/no decision are the same (temperature 0 can still differ slightly between machines).

    python3 harness/check_rq1.py --rq1-run PATH/results-local/rq1 --lissa-cache PATH/cache-local-r2c [--reask 20]
"""

import argparse
import json
import pathlib
import random
import re

import evaluate
import lissa_compat as lc
import llm
import retrieval
import run as runner
from variants import Pair, baseline

MODELS = ("qwen2.5-coder:14b", "llama3.1:8b")


def lissa_output(rq1_run, prefix, config_stem):
    """LiSSA's result or trace-link file of one config (name: <prefix>-<config file name>_<uuid>.<ext>)."""
    ext = "md" if prefix == "results" else "csv"
    found = sorted(pathlib.Path(rq1_run).glob(f"{prefix}-{config_stem}.json_*.{ext}"))
    return found[0] if found else None


def lissa_metrics(path):
    text = path.read_text()
    return {name: float(re.search(rf"\* {label}: (\S+)", text).group(1))
            for name, label in (("p", "Precision"), ("r", "Recall"), ("f1", "F1"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rq1-run", required=True, help="RQ1 run folder with results-*.md and traceLinks-*.csv")
    parser.add_argument("--lissa-cache", required=True, help="RQ1's cache-local-r2c folder")
    parser.add_argument("--reask", type=int, default=0, help="ask N cached pairs again on this machine")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    report, all_ok = [], True

    def say(line, ok=True):
        nonlocal all_ok
        all_ok &= ok
        report.append(("" if ok else "!! ") + line)
        print(report[-1], flush=True)

    # 1. Candidates
    source = {"kind": "lissa-cache", "path": str(pathlib.Path(args.lissa_cache).resolve())}
    candidates = {"embedding": retrieval.EMBEDDING, "top_k": retrieval.TOP_K, "source": source, "projects": {}}
    for project in lc.PROJECTS:
        candidates["projects"][project] = retrieval.candidates_for_project(project, source)
        ours = {(lc.link_id(req), lc.link_id(code))
                for req, ranked in candidates["projects"][project].items() for code, _ in ranked}
        links_file = lissa_output(args.rq1_run, "traceLinks", f"{project}_artifact_artifact_no_llm_{retrieval.EMBEDDING}")
        if not links_file:
            say(f"{project}: no retrieval-only trace links in {args.rq1_run}", False)
            continue
        theirs = lc.read_trace_links(links_file)
        say(f"{project}: candidates {'identical' if ours == theirs else 'DIFFER'} ({len(ours)} pairs here, "
            f"{len(theirs)} in RQ1; {len(ours - theirs)} only here, {len(theirs - ours)} only in RQ1)", ours == theirs)
    out = lc.HARNESS / "data" / "candidates-rq1.json"
    if all_ok:
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(candidates, indent=1) + "\n")
        say(f"Wrote {out}")

    # 2 and 3. Prompts and decisions, per model
    sample = []
    for model in MODELS:
        for project in lc.PROJECTS:
            answers = runner.lissa_answers(args.lissa_cache, model, project)
            if not answers:
                say(f"{model} {project}: no CoT cache for seed {lc.SEED} (not run yet?)")
                continue
            requirements = lc.load_artifacts(lc.DATASETS / project / "UC")
            code = lc.load_artifacts(lc.DATASETS / project / "CC")
            hits, predicted = 0, set()
            for req, ranked in candidates["projects"][project].items():
                for target, _ in ranked:
                    pair = Pair(project, req, target, requirements[req], code[target])
                    key = lc.message_key(*baseline(pair))
                    if key in answers:
                        hits += 1
                        sample.append((model, pair, answers[key]))
                        if lc.is_related_cot(answers[key]):
                            predicted.add((lc.link_id(req), lc.link_id(target)))
            total = sum(len(ranked) for ranked in candidates["projects"][project].values())
            say(f"{model} {project}: {hits}/{total} prompts found in RQ1's cache", hits == total)
            stem = f"{project}_{lc.SEED}_artifact_artifact_reasoning_ollama_{lc.model_tag(model)}_{retrieval.EMBEDDING}"
            links_file, results_file = lissa_output(args.rq1_run, "traceLinks", stem), lissa_output(args.rq1_run, "results", stem)
            if hits == total and links_file and results_file:
                theirs = lc.read_trace_links(links_file)
                ours = evaluate.evaluate({project: predicted}, "all")["projects"][0]
                expected = lissa_metrics(results_file)
                same = predicted == theirs and all(abs(ours[m] - expected[m]) < 1e-12 for m in expected)
                say(f"{model} {project}: links {'identical' if predicted == theirs else 'DIFFER'}; P/R/F1 here "
                    f"{ours['p']:.4f}/{ours['r']:.4f}/{ours['f1']:.4f}, LiSSA {expected['p']:.4f}/{expected['r']:.4f}/"
                    f"{expected['f1']:.4f}", same)
            elif hits == total:
                say(f"{model} {project}: config not finished in RQ1 yet, links not compared")

    # 4. Same answers on this machine?
    if args.reask and sample:
        same_text = same_decision = 0
        chosen = random.Random(lc.SEED).sample(sample, min(args.reask, len(sample)))
        for model, pair, cached in chosen:
            fresh = llm.ask(args.ollama_url, model, *baseline(pair))["answer"]
            same_text += fresh == cached
            same_decision += lc.is_related_cot(fresh) == lc.is_related_cot(cached)
            print(f"   {model} {pair.project} {pair.requirement} {pair.code}: "
                  f"{'same text' if fresh == cached else 'text differs'}", flush=True)
        say(f"Re-asked {len(chosen)} pairs here: identical text {same_text}, identical yes/no {same_decision}")

    say("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED, see the lines marked !!", all_ok)
    (lc.HARNESS / "data" / "check_rq1.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
