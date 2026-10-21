#!/usr/bin/env python3
"""Derive local-Ollama LiSSA configurations from the original GPT-4o configurations.

Only three parts of each original configuration change:
  * classifier:        reasoning_openai (gpt-4o / gpt-4o-mini) -> reasoning_ollama (<llm>, num_ctx)
  * embedding_creator: openai (text-embedding-3-large)         -> ollama (<embedding>, num_ctx)
  * cache_dir:         ./cache-r2c/...                         -> ./cache-local-r2c/...  (GPT-4o caches stay untouched)
Datasets, preprocessing, top-20 retrieval, aggregation, postprocessing, seeds and the prompt are copied as they are.

For req2code, one retrieval-only configuration per project is added (mock classifier, like the original
*_artifact_artifact_no_llm.json runs); its recall is the ceiling of the retrieval step.

Run from LiSSA-RATLR-V2/lissa:
    python3 local/generate_configs_local.py --llm qwen2.5-coder:14b --embedding bge-m3
"""

import argparse
import copy
import json
import pathlib
import sys

LISSA_DIR = pathlib.Path(__file__).resolve().parent.parent

TASKS = {
    # task: (original GPT-4o configs, original cache root, local cache root)
    "req2code": ("configs/req2code-significance", "./cache-r2c/", "./cache-local-r2c/"),
    "doc2code": ("configs/doc2code-significance", "./cache-d2c/", "./cache-local-d2c/"),
}
# As in run-experiments.sh: the Dronology code may not be redistributed, so its configurations cannot run.
EXCLUDED_DATASETS = ("dronology-dd", "dronology-re")
# Seed order of the original significance runs (generate_configs_*_significance.py).
SEEDS = ["133742243", "35418170", "83939039", "-102959883", "-294857830"]
# The retrieval-only configs reuse the cache directory (and thus the embeddings) of this seed.
RETRIEVAL_ONLY_SEED = SEEDS[0]


def model_tag(model):
    return model.removesuffix(":latest").replace(":", "-").replace("/", "-")


def dataset_of(config):
    parts = pathlib.PurePosixPath(config["gold_standard_configuration"]["path"]).parts
    return parts[parts.index("datasets") + 2]


def localize(config, llm, embedding, llm_num_ctx, embedding_num_ctx, original_cache_root, local_cache_root):
    classifier = config["classifier"]
    if not classifier["name"].endswith("_openai") or config["embedding_creator"]["name"] != "openai":
        raise ValueError(f"unexpected original configuration: {classifier['name']} / {config['embedding_creator']}")
    if not config["cache_dir"].startswith(original_cache_root):
        raise ValueError(f"unexpected cache_dir: {config['cache_dir']}")

    local = copy.deepcopy(config)
    local["cache_dir"] = local_cache_root + config["cache_dir"][len(original_cache_root) :]
    local["embedding_creator"] = {"name": "ollama", "args": {"model": embedding, "num_ctx": str(embedding_num_ctx)}}
    local["classifier"] = {
        "name": classifier["name"].removesuffix("_openai") + "_ollama",
        "args": {**classifier["args"], "model": llm, "num_ctx": str(llm_num_ctx)},
    }
    return local


def generate(task, llm, embedding, llm_num_ctx, embedding_num_ctx, retrieval_only):
    """Returns [(file name, config)] in run order: retrieval-only configs first, then per dataset and seed."""
    original_dir, original_cache_root, local_cache_root = TASKS[task]
    suffix = f"ollama_{model_tag(llm)}_{model_tag(embedding)}"
    reasoning, retrieval = [], []
    for path in sorted((LISSA_DIR / original_dir).glob("*.json")):
        original = json.loads(path.read_text())
        dataset = dataset_of(original)
        if dataset in EXCLUDED_DATASETS:
            continue
        local = localize(
            original, llm, embedding, llm_num_ctx, embedding_num_ctx, original_cache_root, local_cache_root
        )
        seed = local["classifier"]["args"].get("seed", "")
        stem = path.stem.split("_gpt_")[0]
        reasoning.append(((dataset, SEEDS.index(seed) if seed in SEEDS else len(SEEDS)), f"{stem}_{suffix}.json", local))

        if retrieval_only and task == "req2code" and seed == RETRIEVAL_ONLY_SEED:
            mock = copy.deepcopy(local)
            mock["classifier"] = {"name": "mock", "args": {}}
            preprocessors = f"{local['source_preprocessor']['name']}_{local['target_preprocessor']['name']}"
            retrieval.append(((dataset,), f"{dataset}_{preprocessors}_no_llm_{model_tag(embedding)}.json", mock))

    ordered = sorted(retrieval, key=lambda it: it[0]) + sorted(reasoning, key=lambda it: it[0])
    return [(name, config, key[0], config["classifier"]["args"].get("seed")) for key, name, config in ordered]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=[*TASKS, "all"], default="all")
    parser.add_argument("--llm", default="qwen2.5-coder:14b")
    parser.add_argument("--embedding", default="bge-m3")
    parser.add_argument("--llm-num-ctx", type=int, default=16384)
    parser.add_argument(
        "--embedding-num-ctx", type=int, default=8192, help="default: 8192, the input limit of bge-m3"
    )
    parser.add_argument("--projects", nargs="*", help="only these datasets, e.g. SMOS eTour_en iTrust")
    parser.add_argument("--seeds", nargs="*", help="only these seeds (retrieval-only configs are always kept)")
    parser.add_argument("--no-retrieval-only", action="store_true")
    parser.add_argument("--list", action="store_true", help="print the selected config paths in run order")
    parser.add_argument("--dry-run", action="store_true", help="like --list, but write nothing")
    args = parser.parse_args()

    out_dir = pathlib.Path("configs-local") / f"{model_tag(args.llm)}__{model_tag(args.embedding)}"
    selected = []
    for task in TASKS if args.task == "all" else [args.task]:
        configs = generate(
            task, args.llm, args.embedding, args.llm_num_ctx, args.embedding_num_ctx, not args.no_retrieval_only
        )
        datasets = sorted({dataset for _, _, dataset, _ in configs})
        if args.projects:
            unknown = [p for p in args.projects if p not in datasets]
            if unknown and args.task != "all":
                sys.exit(f"Unknown project(s) {unknown} for {task}; available: {' '.join(datasets)}")

        if not args.dry_run:
            (LISSA_DIR / out_dir / task).mkdir(parents=True, exist_ok=True)
            (LISSA_DIR / TASKS[task][2]).mkdir(exist_ok=True)
        for name, config, dataset, seed in configs:
            if not args.dry_run:
                (LISSA_DIR / out_dir / task / name).write_text(json.dumps(config, indent=2) + "\n")
            is_retrieval_only = config["classifier"]["name"] == "mock"
            if args.projects and dataset not in args.projects:
                continue
            if args.seeds and not is_retrieval_only and seed not in args.seeds:
                continue
            selected.append(out_dir / task / name)

    if args.list or args.dry_run:
        print("\n".join(str(path) for path in selected))
    else:
        print(f"Wrote configs to {out_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
