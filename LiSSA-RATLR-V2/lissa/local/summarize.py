#!/usr/bin/env python3
"""Summarises the LiSSA result files of a local run next to the original GPT-4o results (read only).

Per project: precision, recall, F1 and F2 (mean over seeds); then the plain average over the projects and
the average weighted by the number of gold-standard links, as reported in the LiSSA paper.
Writes summary.md and summary.csv into the run directory.
"""

import argparse
import csv
import json
import math
import pathlib
import re
import statistics

FIELDS = {
    "gs": r"# TraceLinks \(GS\): (\d+)",
    "tp": r"True Positives: (\d+)",
    "fp": r"False Positives: (\d+)",
    "fn": r"False Negatives: (\d+)",
}


def parse(path):
    text = path.read_text()
    config = json.loads(re.search(r"```json\n(.*?)\n```", text, re.S).group(1))
    values = {key: int(re.search(pattern, text).group(1)) for key, pattern in FIELDS.items()}
    tp, fp, fn = values["tp"], values["fp"], values["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    def f_beta(beta):
        denominator = beta**2 * precision + recall
        return (1 + beta**2) * precision * recall / denominator if denominator else 0.0

    gs_parts = pathlib.PurePosixPath(config["gold_standard_configuration"]["path"]).parts
    classifier = config["classifier"]
    embedding = config["embedding_creator"]["args"].get("model", config["embedding_creator"]["name"])
    if classifier["name"] == "mock":
        approach = f"retrieval only ({embedding})"
    else:
        approach = f"{classifier['args'].get('model', classifier['name'])} + {embedding}"
    return {
        "approach": approach,
        "mode": classifier["name"].split("_")[0],
        "preprocessors": (config["source_preprocessor"]["name"], config["target_preprocessor"]["name"]),
        "project": gs_parts[gs_parts.index("datasets") + 2],
        "seed": classifier["args"].get("seed", "-"),
        "gs": values["gs"],
        "p": precision,
        "r": recall,
        "f1": f_beta(1),
        "f2": f_beta(2),
    }


def aggregate(results, projects):
    rows = []
    for approach in dict.fromkeys(r["approach"] for r in results):
        per_project = {}
        for project in projects:
            runs = [r for r in results if r["approach"] == approach and r["project"] == project]
            if runs:
                metrics = {m: statistics.mean(r[m] for r in runs) for m in ("p", "r", "f1", "f2")}
                spread = statistics.stdev([r["f1"] for r in runs]) if len(runs) > 1 else float("nan")
                per_project[project] = {**metrics, "runs": len(runs), "f1_sd": spread, "gs": runs[0]["gs"]}
                rows.append({"approach": approach, "project": project, **per_project[project]})
        if len(per_project) == len(projects) and len(projects) > 1:
            links = sum(v["gs"] for v in per_project.values())
            for label, weight in (("Avg.", lambda v: 1 / len(per_project)), ("w. Avg.", lambda v: v["gs"] / links)):
                metrics = {m: sum(weight(v) * v[m] for v in per_project.values()) for m in ("p", "r", "f1", "f2")}
                rows.append({"approach": approach, "project": label, **metrics, "runs": "", "f1_sd": float("nan")})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=pathlib.Path)
    parser.add_argument("--baseline-dir", type=pathlib.Path, help="original results, e.g. results-r2c (read only)")
    args = parser.parse_args()

    local = [parse(p) for p in sorted(args.run_dir.glob("results-*.md"))]
    if not local:
        print(f"No result files in {args.run_dir}")
        return
    projects = list(dict.fromkeys(r["project"] for r in local))
    # Baseline runs with the same pipeline as the local runs (same preprocessing, classifier mode and projects)
    setups = {(r["preprocessors"], r["mode"]) for r in local}
    baseline = []
    if args.baseline_dir and args.baseline_dir.is_dir():
        baseline = [r for r in map(parse, sorted(args.baseline_dir.glob("results-*.md")))
                    if (r["preprocessors"], r["mode"]) in setups and r["project"] in projects]

    rows = aggregate(local, projects) + aggregate(baseline, projects)
    lines = ["| Approach | Project | Runs | Precision | Recall | F1 (sd) | F2 |", "|---|---|---|---|---|---|---|"]
    for row in rows:
        sd = "" if math.isnan(row["f1_sd"]) else f" ({row['f1_sd']:.3f})"
        lines.append(f"| {row['approach']} | {row['project']} | {row['runs']} | {row['p']:.3f} | {row['r']:.3f} "
                     f"| {row['f1']:.3f}{sd} | {row['f2']:.3f} |")
    note = (f"\nLocal results: `{args.run_dir}`." +
            (f" Baseline (original GPT-4o runs, read only): `{args.baseline_dir}`." if baseline else "") +
            " Avg./w. Avg. are only shown when an approach has results for all projects.\n")
    summary = "\n".join(lines) + "\n" + note
    (args.run_dir / "summary.md").write_text(summary)
    with open(args.run_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["approach", "project", "runs", "p", "r", "f1", "f1_sd", "f2"],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(summary)


if __name__ == "__main__":
    main()
