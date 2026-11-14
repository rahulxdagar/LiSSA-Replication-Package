#!/usr/bin/env python3
"""Analysis for RQ2 and RQ3: choices on the dev slice, comparisons on the test slice.

A RESULT is a run folder (links as run.py accepted them) or a run folder plus a verification threshold,
written RUN@tau=0.6 (links kept by verify.py at that tau).

    python3 harness/analyze.py table   RESULT...                  # P, R, F1, F2 per project and averages
    python3 harness/analyze.py variant RUN...                     # dev: the variant with the highest average F1
    python3 harness/analyze.py tau     RUN                        # dev: the tau with the highest average F1
    python3 harness/analyze.py test    BASE RESULT...             # test: per-requirement F1, Wilcoxon, Holm

All commands take --parser strict|tolerant (default: strict, LiSSA's rule).
Wilcoxon: two-sided signed-rank test on per-requirement F1 (scipy, zero differences dropped), alpha 0.05, Holm-corrected
over the comparisons of one call; effect size is the matched-pairs rank-biserial correlation. By default, requirements
without gold links are excluded (their F1 is undefined or 0); --unlinked score gives them F1 1 without and 0 with
predicted links instead.
"""

import argparse
import csv
import json
import pathlib

import numpy as np
from scipy import stats

import evaluate
import lissa_compat as lc

ALPHA = 0.05


def load(result, parser):
    """(name, {project: links}, judged split) for a run folder or 'RUN@tau=X'."""
    folder, _, tau = result.partition("@tau=")
    run = pathlib.Path(folder)
    with open(run / "decisions.csv", newline="") as file:
        rows = list(csv.DictReader(file))
    judged = {}
    for row in rows:
        judged.setdefault(row["project"], set()).add(lc.link_id(row["requirement"]))
    links = {p: set() for p in judged}
    if tau:
        for line in (run / f"verify-{parser}" / "links.jsonl").read_text().splitlines():
            record = json.loads(line)
            if record["share"] >= float(tau):
                links[record["project"]].add((lc.link_id(record["requirement"]), lc.link_id(record["code"])))
    else:
        for row in rows:
            if row[f"related_{parser}"] == "True":
                links[row["project"]].add((lc.link_id(row["requirement"]), lc.link_id(row["code"])))
    split = {"projects": {p: {"judged": sorted(reqs)} for p, reqs in judged.items()}}
    return run.name + (f" @ tau={tau}" if tau else ""), links, split


def average_f1(result):
    """Plain mean of the per-project F1 (LiSSA's 'Avg.'); also defined for a run with a single project."""
    rows = [row for row in result["projects"] if row["project"] not in ("Avg.", "w. Avg.")]
    return sum(row["f1"] for row in rows) / len(rows)


def cmd_table(args):
    for result in args.results:
        name, links, split = load(result, args.parser)
        print(f"### {name} ({args.parser} parsing)\n\n{evaluate.table(evaluate.evaluate(links, 'judged', split))}\n")


def cmd_variant(args):
    scores = []
    for run in args.runs:
        name, links, split = load(run, args.parser)
        scores.append((average_f1(evaluate.evaluate(links, "judged", split)), name))
    for f1, name in sorted(scores, reverse=True):
        print(f"{f1:.4f}  {name}")
    print(f"\nHighest average F1 ({args.parser} parsing): {max(scores)[1]}")


def cmd_tau(args):
    metrics = json.loads((pathlib.Path(args.run) / f"verify-{args.parser}" / "metrics.json").read_text())
    scores = {tau: average_f1(result) for tau, result in metrics.items()}
    for tau, f1 in scores.items():
        print(f"tau {tau:>4}: average F1 {f1:.4f}")
    best = max((tau for tau in scores if tau != "none"), key=lambda t: (scores[t], -float(t)))
    print(f"\nChosen tau ({args.parser} parsing): {best} (ties go to the smaller tau, which removes fewer links)")


def per_requirement(links, split, unlinked):
    values = {}
    for row in evaluate.evaluate(links, "judged", split)["requirements"]:
        has_gold = row["tp"] + row["fn"] > 0
        if has_gold:
            values[(row["project"], row["requirement"])] = row["f1"]
        elif unlinked == "score":
            values[(row["project"], row["requirement"])] = 1.0 if row["fp"] == 0 else 0.0
    return values


def rank_biserial(differences):
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    return float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())


def holm(p_values):
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted, running = [0.0] * len(p_values), 0.0
    for position, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - position) * p_values[i]))
        adjusted[i] = running
    return adjusted


def cmd_test(args):
    base_name, base_links, base_split = load(args.base, args.parser)
    base = per_requirement(base_links, base_split, args.unlinked)
    print(f"Base: {base_name}\n\n{evaluate.table(evaluate.evaluate(base_links, 'judged', base_split))}\n")
    rows = []
    for result in args.results:
        name, links, split = load(result, args.parser)
        if split != base_split:
            raise SystemExit(f"{name} judged other requirements than the base; compare runs of the same slice.")
        other = per_requirement(links, split, args.unlinked)
        keys = sorted(base)
        differences = np.array([other[k] - base[k] for k in keys])
        if np.any(differences != 0):
            p = float(stats.wilcoxon([other[k] for k in keys], [base[k] for k in keys],
                                     alternative="two-sided").pvalue)
        else:
            p = 1.0
        print(f"{name}\n\n{evaluate.table(evaluate.evaluate(links, 'judged', split))}\n")
        rows.append({"comparison": f"{name} vs {base_name}", "n": len(keys),
                     "changed": int(np.count_nonzero(differences)), "mean_diff": float(differences.mean()),
                     "median_diff": float(np.median(differences)), "p": p, "r": rank_biserial(differences)})
    for row, adjusted in zip(rows, holm([row["p"] for row in rows])):
        row["p_holm"] = adjusted
    print(f"Per-requirement F1 ({args.parser} parsing; unlinked requirements: {args.unlinked}); "
          f"two-sided Wilcoxon, Holm over {len(rows)} comparison(s)\n")
    print("| Comparison | n | changed | mean ΔF1 | median ΔF1 | p | p (Holm) | r | significant |")
    print("|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        print(f"| {row['comparison']} | {row['n']} | {row['changed']} | {row['mean_diff']:+.3f} | "
              f"{row['median_diff']:+.3f} | {row['p']:.4f} | {row['p_holm']:.4f} | {row['r']:+.2f} | "
              f"{'yes' if row['p_holm'] < ALPHA else 'no'} |")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parser", choices=list(lc.PARSERS), default="strict")
    commands = parser.add_subparsers(dest="command", required=True)
    table = commands.add_parser("table")
    table.add_argument("results", nargs="+")
    variant = commands.add_parser("variant")
    variant.add_argument("runs", nargs="+")
    tau = commands.add_parser("tau")
    tau.add_argument("run")
    test = commands.add_parser("test")
    test.add_argument("base")
    test.add_argument("results", nargs="+")
    test.add_argument("--unlinked", choices=["exclude", "score"], default="exclude")
    test.add_argument("--json", help="also write the comparison rows to this file")
    args = parser.parse_args()
    {"table": cmd_table, "variant": cmd_variant, "tau": cmd_tau, "test": cmd_test}[args.command](args)


if __name__ == "__main__":
    main()
