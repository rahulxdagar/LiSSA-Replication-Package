#!/usr/bin/env python3
"""Scores predicted links on a slice of the requirements, with LiSSA's formulas.

Per project: precision, recall, F1, F2 over the links whose requirement is in the slice; then the plain average over
the projects and the average weighted by gold links (as LiSSA reports them). Per requirement: TP, FP, FN and F1 for
the Wilcoxon tests. F1 is left undefined (NaN) for a requirement with no gold link and no predicted link; how such
requirements enter the tests is decided in the analysis, not here.

Score LiSSA's own RQ1 output on a slice:
    python3 harness/evaluate.py --slice test --links SMOS=path/traceLinks-SMOS_...csv eTour_en=... iTrust=...
"""

import argparse
import json
import math
import pathlib

import lissa_compat as lc

SPLIT_FILE = lc.HARNESS / "data" / "split.json"


def requirements_in(slice_name, project, split=None):
    if slice_name == "all":
        return {lc.link_id(name) for name in lc.load_artifacts(lc.DATASETS / project / "UC")}
    split = split or json.loads(SPLIT_FILE.read_text())
    return set(split["projects"][project][slice_name])


def evaluate(predicted, slice_name, split=None):
    """predicted: {project: {(requirement id, code id), ...}} -> {"projects": [...], "requirements": [...]}"""
    projects, per_requirement = [], []
    for project, links in predicted.items():
        reqs = requirements_in(slice_name, project, split)
        gold = {link for link in lc.load_gold(project) if link[0] in reqs}
        links = {link for link in links if link[0] in reqs}
        tp, fp, fn = lc.counts(links, gold)
        projects.append({"project": project, "requirements": len(reqs), "gold": len(gold), "predicted": len(links),
                         "tp": tp, "fp": fp, "fn": fn, **lc.scores(tp, fp, fn)})
        for req in sorted(reqs):
            r_tp, r_fp, r_fn = lc.counts({l for l in links if l[0] == req}, {l for l in gold if l[0] == req})
            f1 = 2 * r_tp / (2 * r_tp + r_fp + r_fn) if r_tp + r_fp + r_fn else math.nan
            per_requirement.append({"project": project, "requirement": req, "tp": r_tp, "fp": r_fp, "fn": r_fn,
                                    "f1": f1})
    if len(projects) > 1:
        links_total = sum(row["gold"] for row in projects)
        for label, weight in (("Avg.", lambda row: 1 / len(projects)), ("w. Avg.", lambda row: row["gold"] / links_total)):
            averaged = {m: sum(weight(row) * row[m] for row in projects) for m in ("p", "r", "f1", "f2")}
            projects.append({"project": label, **averaged})
    return {"slice": slice_name, "projects": projects, "requirements": per_requirement}


def table(result):
    lines = ["| Project | Reqs | Gold | Pred. | TP | FP | FN | P | R | F1 | F2 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in result["projects"]:
        n = [str(row.get(k, "")) for k in ("requirements", "gold", "predicted", "tp", "fp", "fn")]
        lines.append(f"| {row['project']} | " + " | ".join(n) + " | " +
                     " | ".join(f"{row[m]:.3f}" for m in ("p", "r", "f1", "f2")) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slice", choices=["dev", "test", "all"], default="test")
    parser.add_argument("--links", nargs="+", required=True, metavar="PROJECT=traceLinks.csv")
    parser.add_argument("--json", help="also write the full result (with per-requirement rows) to this file")
    args = parser.parse_args()
    predicted = {}
    for item in args.links:
        project, path = item.split("=", 1)
        predicted[project] = lc.read_trace_links(path)
    result = evaluate(predicted, args.slice)
    print(table(result))
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
