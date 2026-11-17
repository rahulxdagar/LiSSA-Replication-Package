# RQ1: LiSSA with local LLMs vs. the published results

Generated 2026-11-17 00:53. Configs finished: 15 of 15. Candidate pairs answered: 20,480 of 20,480 (100%).

Pipeline: LiSSA V2 unchanged in prompts, retrieval (top 20), aggregation and metrics; whole artifacts; bge-m3; temperature 0; SMOS, eTour and iTrust. Values with several runs are means over seeds. Ollama client settings (see `lissa-vs-upstream.diff`): context 16,384 tokens, embedding inputs cut at 8,192 tokens, answers capped at 4,096 tokens.

## Averages over the three projects

| Approach | Avg. P | Avg. R | Avg. F1 | Avg. F2 | w. Avg. P | w. Avg. R | w. Avg. F1 | w. Avg. F2 |
|---|---|---|---|---|---|---|---|---|
| retrieval only (bge-m3) | 0.190 | 0.506 | 0.250 | 0.335 | 0.263 | 0.481 | 0.316 | 0.377 |
| qwen2.5-coder:14b KISS | 0.448 | 0.358 | 0.346 | 0.347 | 0.569 | 0.283 | 0.326 | 0.291 |
| qwen2.5-coder:14b CoT | 0.404 | 0.211 | 0.243 | 0.219 | 0.510 | 0.169 | 0.224 | 0.184 |
| llama3.1:8b KISS | 0.210 | 0.498 | 0.265 | 0.346 | 0.293 | 0.469 | 0.333 | 0.383 |
| llama3.1:8b CoT | 0.205 | 0.443 | 0.257 | 0.327 | 0.281 | 0.433 | 0.320 | 0.365 |
| retrieval only (text-embedding-3-large) | 0.200 | 0.588 | 0.271 | 0.374 | 0.258 | 0.512 | 0.316 | 0.386 |
| GPT-4o KISS | 0.405 | 0.463 | 0.356 | 0.401 | 0.509 | 0.337 | 0.325 | 0.317 |
| GPT-4o CoT (original run) | 0.399 | 0.460 | 0.365 | 0.406 | 0.488 | 0.341 | 0.334 | 0.326 |
| GPT-4o CoT (5 seeds) | 0.399 | 0.457 | 0.362 | 0.403 | 0.488 | 0.336 | 0.328 | 0.320 |
| GPT-4o-mini KISS | 0.227 | 0.571 | 0.291 | 0.389 | 0.291 | 0.481 | 0.327 | 0.382 |
| GPT-4o-mini CoT | 0.361 | 0.479 | 0.342 | 0.396 | 0.459 | 0.361 | 0.331 | 0.330 |
| VSM (opt.) | 0.398 | 0.356 | 0.374 | 0.363 | 0.415 | 0.384 | 0.398 | 0.389 |
| LSI (opt.) | 0.373 | 0.379 | 0.376 | 0.378 | 0.393 | 0.404 | 0.398 | 0.402 |
| FTLR | 0.329 | 0.434 | 0.359 | 0.396 | 0.383 | 0.389 | 0.370 | 0.376 |
| FTLR (opt.) | 0.351 | 0.475 | 0.398 | 0.439 | 0.336 | 0.529 | 0.405 | 0.469 |

The paper's averages also include the two Dronology projects (for example the 0.32 average F1 of GPT-4o CoT in the proposal), so they are not comparable with these. All published rows here are recomputed over SMOS, eTour and iTrust with the paper's formulas: mean per project, and mean weighted by gold-standard links. A dash means that not all three projects have results yet.

The retrieval-only rows give the recall ceiling of each embedding model.

## Per project

| Approach | SMOS P | SMOS R | SMOS F1 | SMOS F2 | eTour P | eTour R | eTour F1 | eTour F2 | iTrust P | iTrust R | iTrust F1 | iTrust F2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| retrieval only (bge-m3) | 0.349 | 0.447 | 0.392 | 0.423 | 0.177 | 0.666 | 0.279 | 0.429 | 0.044 | 0.406 | 0.080 | 0.154 |
| qwen2.5-coder:14b KISS | 0.711 | 0.189 | 0.298 | 0.221 | 0.411 | 0.581 | 0.481 | 0.537 | 0.223 | 0.304 | 0.257 | 0.284 |
| qwen2.5-coder:14b CoT | 0.632 | 0.117 | 0.197 | 0.140 | 0.457 | 0.328 | 0.382 | 0.348 | 0.123 | 0.189 | 0.149 | 0.171 |
| llama3.1:8b KISS | 0.391 | 0.431 | 0.410 | 0.422 | 0.181 | 0.662 | 0.284 | 0.432 | 0.058 | 0.402 | 0.102 | 0.185 |
| llama3.1:8b CoT | 0.369 | 0.417 | 0.391 | 0.406 | 0.196 | 0.604 | 0.296 | 0.427 | 0.049 | 0.308 | 0.084 | 0.149 |
| retrieval only (text-embedding-3-large) | 0.325 | 0.418 | 0.366 | 0.395 | 0.216 | 0.815 | 0.342 | 0.525 | 0.058 | 0.531 | 0.105 | 0.202 |
| GPT-4o KISS | 0.632 | 0.184 | 0.285 | 0.214 | 0.378 | 0.711 | 0.493 | 0.604 | 0.206 | 0.493 | 0.290 | 0.385 |
| GPT-4o CoT (original run) | 0.590 | 0.195 | 0.294 | 0.226 | 0.409 | 0.734 | 0.526 | 0.633 | 0.199 | 0.451 | 0.276 | 0.360 |
| GPT-4o CoT (5 seeds) | 0.592 | 0.188 | 0.285 | 0.217 | 0.400 | 0.728 | 0.516 | 0.625 | 0.206 | 0.457 | 0.284 | 0.367 |
| GPT-4o-mini KISS | 0.364 | 0.370 | 0.367 | 0.369 | 0.235 | 0.815 | 0.364 | 0.545 | 0.083 | 0.528 | 0.143 | 0.254 |
| GPT-4o-mini CoT | 0.572 | 0.216 | 0.314 | 0.247 | 0.374 | 0.740 | 0.497 | 0.619 | 0.138 | 0.479 | 0.214 | 0.321 |
| VSM (opt.) | 0.430 | 0.414 | 0.422 | 0.417 | 0.557 | 0.427 | 0.483 | 0.448 | 0.208 | 0.227 | 0.217 | 0.223 |
| LSI (opt.) | 0.415 | 0.430 | 0.422 | 0.427 | 0.452 | 0.453 | 0.453 | 0.453 | 0.251 | 0.255 | 0.253 | 0.254 |
| FTLR | 0.444 | 0.331 | 0.380 | 0.349 | 0.379 | 0.633 | 0.474 | 0.558 | 0.165 | 0.339 | 0.222 | 0.280 |
| FTLR (opt.) | 0.314 | 0.588 | 0.409 | 0.501 | 0.505 | 0.597 | 0.548 | 0.576 | 0.234 | 0.241 | 0.238 | 0.240 |

## Runs per approach

| Approach | SMOS | eTour | iTrust | F1 sd (SMOS / eTour / iTrust) |
|---|---|---|---|---|
| retrieval only (bge-m3) | 1 | 1 | 1 | — / — / — |
| qwen2.5-coder:14b KISS | 1 | 1 | 1 | — / — / — |
| qwen2.5-coder:14b CoT | 1 | 1 | 1 | — / — / — |
| llama3.1:8b KISS | 1 | 1 | 1 | — / — / — |
| llama3.1:8b CoT | 1 | 1 | 1 | — / — / — |
| retrieval only (text-embedding-3-large) | 1 | 1 | 1 | — / — / — |
| GPT-4o KISS | 1 | 1 | 1 | — / — / — |
| GPT-4o CoT (original run) | 1 | 1 | 1 | — / — / — |
| GPT-4o CoT (5 seeds) | 5 | 5 | 5 | 0.0032 / 0.0052 / 0.0033 |
| GPT-4o-mini KISS | 1 | 1 | 1 | — / — / — |
| GPT-4o-mini CoT | 1 | 1 | 1 | — / — / — |
| VSM (opt.) | 1 | 1 | 1 | — / — / — |
| LSI (opt.) | 1 | 1 | 1 | — / — / — |
| FTLR | 1 | 1 | 1 | — / — / — |
| FTLR (opt.) | 1 | 1 | 1 | — / — / — |

## Validity checks

- Requirements, code files and gold links of every run match the proposal.
- Retrieval-only candidate pairs: 5,120 {'SMOS': 1340, 'eTour_en': 1160, 'iTrust': 2620} (as expected).

| LLM | Prompt | Answers | Not parseable as intended |
|---|---|---|---|
| qwen2.5-coder:14b | CoT | 5,040 | 3,352 without a `<trace>` answer (counted as no) |
| qwen2.5-coder:14b | KISS | 5,040 | 0 with neither yes nor no (counted as no); 0 with both (counted as yes) |
| llama3.1:8b | CoT | 5,040 | 949 without a `<trace>` answer (counted as no); 62 over 8,000 characters, probably cut at the 4,096-token answer limit |
| llama3.1:8b | KISS | 5,040 | 0 with neither yes nor no (counted as no); 0 with both (counted as yes) |

## Models and hardware evidence

| LLM | Digest | Quantisation | num_ctx | GPU | Longest prompt (tokens) | Fits | Embedding (num_ctx, GPU) | Ollama |
|---|---|---|---|---|---|---|---|---|
| llama3.1:8b | 46e0c10c039e | Q4_K_M | 16384 | 100% | 9947 | True | bge-m3 (8192, 100%) | 0.40.14 |
| qwen2.5-coder:14b | 9ec8897f747e | Q4_K_M | 16384 | 100% | 10224 | True | bge-m3 (8192, 100%) | 0.40.14 |

`ollama-ps.jsonl` has the GPU placement after every config, `lissa-vs-upstream.diff` the code differences from the original package, `code-version.txt` the commit, jar hash and tool versions.

Sources of the published rows: `Evaluation-Req2Code.xlsx` (read only) and, for the 5-seed runs, `LiSSA-RATLR-V2/lissa/results-r2c/`.
