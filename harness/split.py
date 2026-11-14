#!/usr/bin/env python3
"""The 20% development / 80% test split of the requirements (proposal, Analysis Procedure for RQ2-RQ3).

Per project, 20% of the requirements (rounded) go to the development slice. Within a project the slice is drawn
separately from requirements with and without gold links (largest-remainder allocation), so both slices keep the
project's share of unlinked requirements (iTrust: 26 of 131). The seed is fixed; the split is written once and then
frozen: prompt variants and the threshold tau are chosen on 'dev' only, all reported comparisons use 'test'.

    python3 harness/split.py            # writes harness/data/split.json, refuses to overwrite it
"""

import argparse
import json
import pathlib
import random
import sys

import lissa_compat as lc

SEED = 5187  # fixed before any RQ2/RQ3 result was seen
DEV_SHARE = 0.2


def allocate(total, sizes):
    """Largest-remainder allocation of 'total' over groups proportional to 'sizes'."""
    exact = [total * size / sum(sizes) for size in sizes]
    shares = [int(x) for x in exact]
    by_remainder = sorted(range(len(sizes)), key=lambda i: exact[i] - shares[i], reverse=True)
    for i in by_remainder[: total - sum(shares)]:
        shares[i] += 1
    return shares


def make_split(seed=SEED):
    rng = random.Random(seed)
    split = {"seed": seed, "dev_share": DEV_SHARE, "projects": {}}
    for project in lc.PROJECTS:
        requirements = sorted(lc.link_id(name) for name in lc.load_artifacts(lc.DATASETS / project / "UC"))
        linked = {source for source, _ in lc.load_gold(project)}
        groups = [[r for r in requirements if r in linked], [r for r in requirements if r not in linked]]
        dev_total = round(DEV_SHARE * len(requirements))
        dev = []
        for group, size in zip(groups, allocate(dev_total, [len(g) for g in groups])):
            dev += rng.sample(group, size)
        dev = sorted(dev)
        test = [r for r in requirements if r not in set(dev)]
        split["projects"][project] = {"dev": dev, "test": test}
    return split


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(lc.HARNESS / "data" / "split.json"))
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; the split is frozen. Delete it only if no RQ2/RQ3 run has used it yet.")
    split = make_split()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=1) + "\n")
    for project, parts in split["projects"].items():
        linked = {source for source, _ in lc.load_gold(project)}
        unlinked_dev = sum(r not in linked for r in parts["dev"])
        print(f"{project:9} dev {len(parts['dev']):3} ({unlinked_dev} without gold links)  test {len(parts['test']):3}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
