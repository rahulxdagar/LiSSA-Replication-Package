# Harness for RQ2–RQ4

A Python harness that judges LiSSA's candidate pairs with enriched prompts (RQ2) and verifies accepted links claim by claim (RQ3). LiSSA itself stays exactly as RQ1 ran it. The harness copies the LiSSA behaviour it depends on:

- **Candidates:** LiSSA's top-20 bge-m3 retrieval.
- **Prompt and cache key:** the baseline prompt is LiSSA's chain-of-thought prompt, character for character, so RQ1's cached answers are reused.
- **Ollama request:** same temperature, seed, `num_ctx` and `num_predict`.
- **Answer parsing and metrics:** `<trace>` parsing and LiSSA's metric formulas.

Every Ollama call's timing and token counts are kept (RQ4).

**Setup:** Python 3 with `pip install -r harness/requirements.txt` (numpy, scipy, and tree-sitter for Java).

| File | Purpose |
|---|---|
| `lissa_compat.py` | Exact copies of LiSSA's file loading, prompt, cache key, `<trace>` parsing and metrics, plus the tolerant parsing |
| `llm.py` | Ollama requests, chip detection, and the answer cache (one file per model and chip) |
| `retrieval.py` | LiSSA's top-20 retrieval, from RQ1's cached embeddings or from bge-m3 on this machine |
| `split.py` | The frozen 20% dev / 80% test split (`data/split.json`, seed 5187) |
| `javaparse.py` | Java structure: summaries (RQ2), declared identifiers and the code of a cited element (RQ3) |
| `prepare_prompts.py` | Writes the code summaries and the cross-project examples, and checks that every prompt fits the context |
| `variants.py` | RQ2 prompts: `baseline` (LiSSA's), `definition`, `summary`, `project`, `examples` |
| `prompts/` | The added texts and the RQ3 prompts (**drafts**, see `prompts/README.md`) |
| `run.py` | RQ2: one variant on one slice. Writes decisions, links and metrics under both parsings |
| `verify.py` | RQ3: claims, identifier check and factored check for every accepted link. Metrics for each τ |
| `analyze.py` | Chooses the variant and τ on dev; on test, per-requirement F1 with Wilcoxon and Holm |
| `check_rq1.py` | Checks the harness against RQ1's outputs (candidates, cache keys, links, metrics) |
| `selftest.py` | Checks that need no RQ1 output |
| `RQ2_RQ3_INSTRUCTIONS.md` | Full instructions and technical details for running RQ2 and RQ3 on Rahul's laptop |

## Workflow

```bash
cd LiSSA-Replication-Package
python3 harness/selftest.py

# 0. Once RQ1's results-local/rq1/ and cache-local-r2c/ are here: exact candidates (data/candidates-rq1.json)
python3 harness/check_rq1.py --rq1-run PATH/results-local/rq1 --lissa-cache PATH/cache-local-r2c --reask 20
python3 harness/prepare_prompts.py                     # summaries, examples, length check

# RQ2 on dev: the baseline (from RQ1's cache if on RQ1's chip) and the four variants, then choose
python3 harness/run.py --variant baseline   --model qwen2.5-coder:14b --slice dev --lissa-cache PATH/cache-local-r2c
python3 harness/run.py --variant definition --model qwen2.5-coder:14b --slice dev --hours 6   # same for summary, project, examples
python3 harness/analyze.py variant harness/runs/*__dev

# RQ3 on dev: verify the baseline's and the best variant's accepted links, then choose tau
python3 harness/verify.py --run harness/runs/baseline__qwen2.5-coder-14b__apple-m5__dev --parser strict
python3 harness/analyze.py tau harness/runs/baseline__qwen2.5-coder-14b__apple-m5__dev

# Test slice: the 2x2 (baseline or best prompt, with or without verification at the frozen tau)
python3 harness/run.py    --variant <best> --model qwen2.5-coder:14b --slice test
python3 harness/verify.py --run harness/runs/<best>__...__test --parser strict      # and for the baseline's test run
python3 harness/analyze.py test harness/runs/baseline__...__test \
    harness/runs/<best>__...__test  harness/runs/baseline__...__test@tau=0.6  harness/runs/<best>__...__test@tau=0.6
```

**Outputs:**
- `harness/cache/<model>__<chip>.jsonl` holds every answer, with Ollama's durations.
- `harness/runs/<variant>__<model>__<chip>__<slice>/` holds:
  - `decisions.csv`: one row per pair, with the rank, cosine, both decisions, answer source and timing;
  - `links-<parser>-<project>.csv`;
  - `metrics.md` and `metrics.json`;
  - `run.json`: provenance (model digest, Ollama version, hashes of the code, prompts, candidates and split);
  - `verify-<parser>/`: RQ3 output for this run.

## Rules built into the code (and where to change them)

- **Answers are never mixed across chips.** Temperature 0 is not bit-identical across chips, so each cache file belongs to one chip. RQ1's answers are used only through `--lissa-cache`. Use that option only when running on RQ1's chip.
- **Both parsings are always reported.** Strict is LiSSA's rule; tolerant also accepts a leading "Yes"/"No" when the `<trace>` tag is missing. Which one is primary is still to be decided.
- **RQ3 settings** (all in `verify.py`):
  - at most 5 claims per link;
  - `unclear` is not support (`SUPPORTED`);
  - a link without claims has share 0;
  - τ ∈ {0.2, 0.4, 0.6, 0.8, 1.0} (`TAUS`);
  - the verifier is the run's own model unless `--model` is given;
  - ties in choosing τ go to the smaller τ.
- **Undefined F1 in the Wilcoxon tests.** Requirements without gold links are excluded by default; `--unlinked score` includes them instead.
- **The cross-project examples** come only from the other projects' dev slices, so no test-slice requirement ever appears in a prompt.
- **Pilots and reported runs:** `data/candidates-local.json` was computed on Prabhjot's Mac and is for pilots only. Reported runs use `data/candidates-rq1.json`.
- **LiSSA's trace-link CSVs** can be corrupted: LiSSA overwrites the file without truncating it, so the tail of an older, longer file can remain. Four of the published GPT-4o CSVs have this problem (`selftest.py` lists them). LiSSA's `results-*.md` files are correct.
