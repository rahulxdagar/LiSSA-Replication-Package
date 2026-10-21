# LiSSA with local Ollama models (CSI5187 research copy)

This folder is an isolated copy of the LiSSA ICSE'25 replication package (upstream commit `a8e652f2`, branch `local-ollama`).
It runs LiSSA's requirements-to-code evaluation (RQ1) with locally hosted open models instead of GPT-4o.
The original package in `../Replication-Package-ICSE25_LiSSA-Toward-Generic-Traceability-Link-Recovery-through-RAG` is never modified.
All later research (RQ2, RQ3) happens in this copy.

## Models and settings

| Role | GPT-4o package | Local primary | Local fallback |
|---|---|---|---|
| Classifier LLM | `gpt-4o-2024-05-13` | `qwen2.5-coder:14b` (Q4_K_M, 9.0 GB) | `gemma4:12b` (Q4_K_M, 7.4 GB) |
| Embeddings | `text-embedding-3-large` | `bge-m3` (F16, 1.2 GB) | `llama3.1:8b` (see the caveat below) |

- **Context:** every LLM request asks for `num_ctx` 16384. Embedding requests ask for the same, but bge-m3's window is 8192 tokens, so Ollama caps it there. That is the same window as OpenAI's embedding model in the original runs.
- **Decoding:** temperature 0 with LiSSA's seed, exactly as in the original code.
- **GPU:** a model counts as usable only if Ollama loads it 100% on the GPU (Metal). The runner logs `/api/ps` after every config as evidence.
- **Configuration:** all of the above lives in [`LiSSA-RATLR-V2/lissa/local/models.conf`](LiSSA-RATLR-V2/lissa/local/models.conf).

**Fallback rule.** The runner picks the models once per run, before LiSSA starts. For each role it uses the primary model unless that model is not pulled, fails to load, or does not fit completely on the GPU; in that case it uses the fallback for the entire run. A run never switches models halfway:
- Mixing two LLMs would make the results uninterpretable.
- Mixing embedding models would break retrieval: bge-m3 vectors have 1024 dimensions, llama3.1 vectors have 4096.

The models chosen, and the reason any primary was rejected, are recorded in `preflight.json` in the run folder.

> **Caveat on `llama3.1:8b` as embedding fallback.** In Ollama 0.40.x only models whose GGUF file declares a pooling type (embedding models such as bge-m3) are started in embedding mode. llama3.1:8b is a chat model, so Ollama will most likely refuse embedding requests for it. The preflight then rejects it with the reason. If you want a working embedding fallback, set `EMBEDDING_FALLBACK` to a real embedding model (for example `nomic-embed-text`, `mxbai-embed-large` or `snowflake-arctic-embed2`).

## What changed compared with the original package

Only the code that talks to the LLM changed. Everything else — artifact loading, preprocessing, top-20 retrieval, prompts, link aggregation, postprocessing and metric computation — is untouched. `git diff` shows the Java changes; `git status` shows the new files.

| File | Change |
|---|---|
| `LiSSA-RATLR-V2/lissa/src/.../classifier/ChatLanguageModelProvider.java` | Ollama chat requests now send `num_ctx` (config argument, default 16384). Without it, Ollama uses its server default (4k on a 24 GB Mac) and silently cuts longer prompts. The OpenAI branch is unchanged. |
| `LiSSA-RATLR-V2/lissa/src/.../embeddingcreator/OllamaEmbeddingCreator.java` | Embeddings now come from Ollama's `/api/embed` endpoint, with `truncate: true`, `num_ctx` and `num_batch = num_ctx` (config argument, default 8192). The old langchain4j 0.33 client used the legacy `/api/embeddings`, which fails for any input longer than the batch size (2048 tokens), i.e. for most code files, and LiSSA's length-fix fallback only knows OpenAI tokenizers. Long inputs are now cut to their first 8192 tokens, as the original code does for OpenAI. |
| `LiSSA-RATLR-V2/lissa/local/` (new) | `models.conf`, `generate_configs_local.py`, `preflight.py`, `summarize.py` |
| `LiSSA-RATLR-V2/lissa/configs-local/` (new, generated) | Local configs. Each is the original GPT-4o config with only the classifier, the embedding creator and `cache_dir` replaced. |
| `LiSSA-RATLR-V2/lissa/env-template-ollama` (new) | `.env` template for manual runs |
| `run-local-benchmark.sh` (new) | One-command runner |

**Not changed:** the GPT-4o configs, the datasets, the GPT-4o results (V1 `evaluation/*/results-gpt4o*`, V2 `results-r2c/` and `results-d2c/`, `statistical-evaluation/`, the Excel sheets), and the original `run-experiments.sh`. Local runs only write to `LiSSA-RATLR-V2/lissa/results-local/` and `LiSSA-RATLR-V2/lissa/cache-local-r2c/` (or `cache-local-d2c/`).

### Benchmark protocol (unchanged from the GPT-4o significance runs)

- **Projects:** SMOS, eTour and iTrust. Dronology is skipped because its code is not distributed, as in `run-experiments.sh`.
- **Pipeline:** whole artifacts on both sides, top-20 retrieval, chain-of-thought prompt.
- **Seeds:** the 5 seeds of the original significance runs.
- **Retrieval-only runs:** in addition, one run per project without the LLM, mirroring the original `*_artifact_artifact_no_llm` runs. It gives the recall ceiling of the retrieval step.

## Setup (once per laptop)

1. **JDK 21:** run `brew install openjdk@21`. LiSSA targets Java 21; if no JDK 21 is found, the runner uses a newer JDK and prints a note.
2. **Ollama:** version 0.30.9 or newer, which gemma4 requires. Ollama 0.40.7 was used while preparing this copy.
3. **Models:** run `bash run-local-benchmark.sh --setup`. It pulls the four models from `models.conf`, about 23 GB.
4. **Optional:** to make 16384 the server default for other Ollama use, run `launchctl setenv OLLAMA_CONTEXT_LENGTH 16384` and restart Ollama. LiSSA does not need this, because every request carries its own `num_ctx`.

## Running a benchmark

```bash
cd LiSSA-Local-Research

# See what would run (calls neither Ollama nor LiSSA)
bash run-local-benchmark.sh --dry-run

# Smoke test: SMOS, one seed (1 retrieval-only run + 1,340 LLM calls)
bash run-local-benchmark.sh --projects SMOS --seeds 133742243

# Full RQ1 benchmark: 3 retrieval-only runs + 3 projects x 5 seeds x top-20 = 25,600 LLM calls
bash run-local-benchmark.sh

# The second model family as its own run (no fallback involved)
bash run-local-benchmark.sh --llm gemma4:12b
```

What a run does:
1. Builds the jar if the sources changed.
2. Runs the preflight. It loads the models with the run's settings and checks GPU placement. It also sends the longest requirement–code pair to check that the longest prompt fits into 16384 tokens.
3. Generates the configs for the chosen models.
4. Runs each config in its own JVM, retrying a failed config up to twice. `caffeinate` keeps the Mac awake; keep it plugged in with the lid open.
5. Prints a summary table next to the GPT-4o results.

**Resuming.** LiSSA caches every answer. If a run is interrupted, start it again with `--run-id <folder name>`: configs that already finished are skipped, and the others continue from the cache. The runner refuses to resume a folder with different models.

**Outputs** go to `LiSSA-RATLR-V2/lissa/results-local/<run-id>/`:

| File | Content |
|---|---|
| `results-*.md` | LiSSA's own result files: config, TP/FP/FN, precision, recall, F1 |
| `traceLinks-*.csv` | Predicted links |
| `summary.md`, `summary.csv` | P, R, F1, F2 per project (mean over seeds), average and link-weighted average; GPT-4o rows recomputed over the same projects |
| `preflight.json` | Chosen models with digests, quantisation, context, GPU share, longest-prompt token count, rejected models with reasons |
| `runs.tsv` | Status, attempts and seconds per config |
| `logs/` | Full LiSSA log per config |
| `ollama-ps.jsonl` | GPU placement after each config |
| `configs/`, `code-version.txt`, `code.diff` | Exact configs, commit and Java changes used |

**Manual run** (no fallback and no GPU check; outputs land in `LiSSA-RATLR-V2/lissa/`):

```bash
cd LiSSA-RATLR-V2/lissa
cp env-template-ollama .env && mkdir -p cache-local-r2c
/opt/homebrew/opt/openjdk@21/bin/java -jar target/ratlr-0.2.0-SNAPSHOT-jar-with-dependencies.jar eval \
  -c configs-local/qwen2.5-coder-14b__bge-m3/req2code/SMOS_133742243_artifact_artifact_reasoning_ollama_qwen2.5-coder-14b_bge-m3.json
```

## Interpreting results

- **Seeds:** with temperature 0, Ollama decodes greedily, so the five seeds should give practically identical answers. GPT-4o is non-deterministic even at temperature 0, which is why the original used five. Running all five reproduces the protocol at five times the cost; the smoke run's time (in `runs.tsv`) lets you estimate a full run.
- **Averages:** the paper's averages include the two Dronology projects. Compare per project, or use the 3-project GPT-4o averages in `summary.md`.
- **Paper's single-run numbers:** the published GPT-4o, GPT-4o-mini, VSM, LSI and FTLR results are in `Evaluation-Req2Code.xlsx` (produced with V1); the 5-seed GPT-4o runs are in `LiSSA-RATLR-V2/lissa/results-r2c/`.
- **Truncation:** code files longer than 8192 tokens are embedded from their first 8192 tokens (LiSSA logs `Input truncated ...`). The GPT-4o runs truncated at about 8000 tokens. If the preflight warns that the longest prompt does not fit, the affected LLM calls were truncated by Ollama; report that as a threat to validity.
- **Thinking models:** if the LLM is a thinking model (the preflight reports this), Ollama's default thinking behaviour applies, because LiSSA's client does not set `think`.
