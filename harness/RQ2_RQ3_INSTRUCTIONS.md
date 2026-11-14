# RQ2 and RQ3 on Rahul's laptop: instructions and technical reference

Everything for RQ2 (prompt enrichment) and RQ3 (claim-level verification) runs on Rahul's MacBook Pro, the machine that ran RQ1.

- **Part A** is the procedure: setup, the decisions to settle first, and the commands in order.
- **Part B** explains how every piece works, so results can be checked and the report's method section written from it.

The code is the `harness/` folder; `harness/README.md` lists its files.

---

# Part A: procedure

## A1. What has to exist first

1. **RQ1's first pass has finished for each model you'll use.** That means the retrieval-only runs plus the CoT run with seed `133742243` for SMOS, eTour_en and iTrust. The harness needs two folders from it:
   - `LiSSA-RATLR-V2/lissa/results-local/rq1/`: LiSSA's `results-*.md` and `traceLinks-*.csv` files.
   - `LiSSA-RATLR-V2/lissa/cache-local-r2c/<project>-133742243/`: the cached bge-m3 embeddings (`OllamaEmbeddingCreator_bge-m3.json`) and the cached CoT answers (`ReasoningClassifier_<model>.json`).
2. **RQ1 is not running at the same time.** Both use Ollama, so they would compete for the GPU and memory, and the timings (RQ4) would be wrong.
3. **Ollama and the models stay exactly as in RQ1.** Same Ollama version, same `qwen2.5-coder:14b` and `llama3.1:8b`. Don't update Ollama or re-pull models until RQ2 and RQ3 are done: a change can alter answers, and the harness records the model digest in every run.

## A2. Setup (once)

```bash
cd LiSSA-Local-Research                    # the folder that contains LiSSA-RATLR-V2/
# Put the harness/ folder here (next to LiSSA-RATLR-V2/), from git or by copying it.
python3 -m venv .venv-harness              # keeps the packages away from the system Python
source .venv-harness/bin/activate          # run this again in every new terminal
pip install -r harness/requirements.txt    # numpy, scipy, tree-sitter 0.23.2, tree-sitter-java 0.23.5
python3 harness/selftest.py                # all lines must say ok
python3 -c "import sys; sys.path.insert(0, 'harness'); import llm; print(llm.device_name())"
```

The last command prints the chip name used in cache and run folder names. On the Pro this should be `apple-m5-pro`. All commands below are run from `LiSSA-Local-Research/` with the environment active.

## A3. Decisions to settle before any reported run

Prabhjot leads RQ2 and RQ3. The code has a default for each decision, so nothing blocks a pilot, but each one should be confirmed and noted for the report.

| # | Decision | Default in the code | Where to change it |
|---|---|---|---|
| 1 | Primary answer parsing: strict (LiSSA: only `<trace>`) or tolerant (else a leading Yes/No) | Both are always computed; analyses default to `--parser strict` | `--parser` |
| 2 | Test design: every variant on the test slice, or only the best one | Either works; the cost differs four-fold (A6) | Which `run.py` commands you run |
| 3 | Models: both for everything, or Qwen fully and Llama only for the final 2×2 | Both are supported | `--model` |
| 4 | Prompt texts (definition, project descriptions, RQ3 prompts) | Drafts in `harness/prompts/` | Edit the `.txt` files, then **freeze** them |
| 5 | RQ3 rules: does `unclear` count as support? A link without claims? The τ grid? | `unclear` = not supported; no claims = share 0; τ ∈ {0.2, 0.4, 0.6, 0.8, 1.0} | `SUPPORTED`, `TAUS` in `verify.py` |
| 6 | Which dev run τ is tuned on | The baseline's dev verification (`analyze.py tau`) | Command choice |
| 7 | Requirements without gold links in the Wilcoxon tests | Excluded | `--unlinked score` |

**Freezing:** once the dev runs start, the prompt files, the split and the code must not change. Any change alters the prompts, so the cached answers no longer match and comparisons break. Every run records SHA-256 hashes of the code, prompts, candidates and split in its `run.json`.

## A4. Commands, in order (shown for Qwen; repeat with `--model llama3.1:8b` if decision 3 says so)

Set two shortcuts first:

```bash
RQ1=LiSSA-RATLR-V2/lissa/results-local/rq1
CACHE=LiSSA-RATLR-V2/lissa/cache-local-r2c
```

**Step 1: check the harness against RQ1, and get the exact candidates.**

```bash
python3 harness/check_rq1.py --rq1-run $RQ1 --lissa-cache $CACHE --reask 20
```

Expected:
- every line ok, ending in `ALL CHECKS PASSED`;
- `harness/data/candidates-rq1.json` is written;
- the re-asked pairs report `identical text 20` (same chip, temperature 0).

The output is also saved to `harness/data/check_rq1.md`. If anything fails, stop and see B12.

**Step 2: inputs of the variants.**

```bash
python3 harness/prepare_prompts.py --force-examples
```

This regenerates the code summaries and redraws the cross-project examples from RQ1's exact candidates. `--force-examples` is needed once, because the current examples were drawn from the pilot candidates. It ends with a length check, and every variant must fit. Then review `harness/prompts/examples-*.txt` with the other prompt texts, and freeze them.

**Step 3: RQ2 on the dev slice** (51 requirements, 1,020 pairs per run).

```bash
python3 harness/run.py --variant baseline --model qwen2.5-coder:14b --slice dev --lissa-cache $CACHE   # instant: RQ1's answers
for v in definition summary project examples; do
  python3 harness/run.py --variant $v --model qwen2.5-coder:14b --slice dev --hours 8
done
python3 harness/analyze.py variant harness/runs/*__qwen2.5-coder-14b__*__dev
```

- **Batches:** with `--hours`, a run stops at the time limit. Repeat the same command to continue; cached answers are not asked again.
- **Choosing the variant:** `analyze.py variant` names the one with the highest average F1. Run it with `--parser strict` and with `--parser tolerant`, and check whether the choice agrees.

**Step 4: RQ3 on the dev slice.** Verify the accepted links of the baseline and the best variant, then choose τ.

```bash
python3 harness/verify.py --run harness/runs/baseline__qwen2.5-coder-14b__apple-m5-pro__dev --parser strict
python3 harness/verify.py --run harness/runs/<best>__qwen2.5-coder-14b__apple-m5-pro__dev --parser strict
python3 harness/analyze.py --parser strict tau harness/runs/baseline__qwen2.5-coder-14b__apple-m5-pro__dev
```

**Step 5: the test slice** (205 requirements, 4,100 pairs per run).

```bash
python3 harness/run.py --variant baseline --model qwen2.5-coder:14b --slice test --lissa-cache $CACHE  # instant
python3 harness/run.py --variant <best>   --model qwen2.5-coder:14b --slice test --hours 8           # repeat until done
python3 harness/verify.py --run harness/runs/baseline__qwen2.5-coder-14b__apple-m5-pro__test --parser strict
python3 harness/verify.py --run harness/runs/<best>__qwen2.5-coder-14b__apple-m5-pro__test --parser strict
```

If decision 2 says every variant goes on test, also run the other three variants with `--slice test`.

**Step 6: the comparisons.** Write τ as chosen in step 4. Each call applies Holm over its own comparisons.

```bash
B=harness/runs/baseline__qwen2.5-coder-14b__apple-m5-pro__test
P=harness/runs/<best>__qwen2.5-coder-14b__apple-m5-pro__test
python3 harness/analyze.py --parser strict test $B $P $B@tau=<τ> $P@tau=<τ> --json harness/runs/rq2_rq3_test_qwen.json
python3 harness/analyze.py --parser strict test $P $P@tau=<τ>   # does verification still help on top of the best prompt?
```

The first call is the 2×2:
- best prompt vs baseline (RQ2);
- baseline with verification vs baseline (RQ3);
- both vs baseline.

The second call answers the complementarity question directly. Repeat both with `--parser tolerant` as a sensitivity analysis.

**Step 7: send back** `harness/runs/` and `harness/cache/`. They contain every answer, timing, decision and metric.

## A5. Running safely for days

- **Interrupting is always safe.** Ctrl-C, a crash, a reboot or the `--hours` limit all lose nothing: each answer is appended to `harness/cache/<model>__<chip>.jsonl` as it arrives, and repeating the command continues. A line cut off by a crash is skipped, and that call is simply made again.
- **Sleep:** the scripts start `caffeinate`, so the Mac stays awake. Keep it on power with the lid open.
- **Prompt length:** if a prompt ever comes close to the 16,384-token context, `prompt_eval_count` in `decisions.csv` shows it. The length check in step 2 makes this unlikely.

## A6. How many calls, and how long

Per model. Baseline answers come free from RQ1's cache.

| Run | New LLM calls |
|---|---|
| RQ2 dev, 4 variants | 4 × 1,020 = 4,080 |
| RQ2 test, best variant only | 4,100 |
| RQ2 test, every variant | 16,400 |
| RQ3, per verified run | 1 claims call per accepted link + up to 5 short checks per link |

**Speed:** on Prabhjot's MacBook Air (M5), a baseline call took a median of 19.3 s (IQR 11.4–38.3 s; 20 calls; eTour). The Pro should be faster; `metrics.md` of the first dev run shows the real median.

**RQ3 volume:**
- Claims calls read the whole requirement and code file and write JSON, so they're slower than a classification call: median 53.8 s on the Air.
- Checks read one method and one claim: median 9.0 s.
- With 5 claims per link, that's about 100 s per verified link on the Air.
- The number of accepted links depends on the model's yes-rate: GPT-4o accepted about 1,540 links over all requirements, and the test slice holds 80% of them.

## A7. Pilot evidence so far (Prabhjot's Air; pilot candidates; eTour UC12; baseline prompt; Qwen)

- **Format compliance:** 14 of 20 answers had no `<trace>` tag. All 14 began with a plain "Yes, …" or "No, …". Strict parsing found 1 of the 4 gold links (P 0.50, R 0.25); tolerant parsing found 3 of 4 (P 0.60, R 0.75). That's why both parsings are reported.
- **RQ3:** see the verification pilot in A8.

## A8. Verification pilot (Prabhjot's Air; the 5 links accepted under tolerant parsing)

eTour UC12 is ViewTouristCard; its gold links are Tourist, TouristTableModel, BeanTourist and TouristCommonManager.

| Link | Gold | Verdicts of the 5 claims | Share |
|---|---|---|---|
| TouristTableModel | yes | unsupported, supported, unclear, unclear, unclear | 0.20 |
| TouristCard | no | 5 × supported | 1.00 |
| IDBTourist | no | 5 × supported | 1.00 |
| BeanTourist | yes | supported, unclear, supported, unclear, unclear | 0.40 |
| Tourist | yes | unsupported, unsupported, unsupported, supported, supported | 0.40 |

- **Mechanics worked:** all 5 claims answers were valid JSON, every link got the maximum of 5 claims, and all 25 cited identifiers exist in their files (no identifier-check failures).
- **Timing on the Air:** claims call median 53.8 s (max 117 s); check call median 9.0 s; so about 100 s per verified link.
- **Result:** the shares run *opposite* to correctness. Both wrong links got 1.00, the three correct ones 0.20–0.40, so every τ ≥ 0.4 lowers F1 here (τ 0.2: F1 0.667; 0.4: 0.500; 0.6 and above: 0.000). Five links prove nothing, but two mechanisms are visible:
  - The factored check confirms true facts about the code. It doesn't test whether those facts connect the code to the requirement, because by design it never sees the requirement.
  - "unclear" (6 of 25 verdicts) counts as not supported and pulls correct links down.

  Look at the dev-slice verification before freezing decision 5 (and possibly the claims prompt).
- **Possible gold gap:** TouristCard looks like a genuine implementation of "view tourist card" but is not a gold link. That's a candidate "gold-standard gap" case for the RQ4 error coding.

---

# Part B: how it works

## B1. Why a separate harness

LiSSA's Java pipeline cannot express two of the things RQ2 and RQ3 need:
- **A per-file code summary.** A config change can't add one to the prompt without also changing what gets embedded, and so which 20 candidates are retrieved.
- **A multi-call verification of each accepted link.** LiSSA's classifier only returns yes/no.

So LiSSA stays exactly as in RQ1, and a Python harness repeats LiSSA's behaviour wherever the comparison depends on it. Every such copy is tested against LiSSA's own files: `selftest.py` without RQ1, `check_rq1.py` with RQ1.

## B2. Candidates (`retrieval.py`)

For each requirement, the candidates are the 20 code files most similar to it, exactly as LiSSA retrieves them:
- **Content:** the whole artifact, read as UTF-8 with `\r\n` turned into `\n` (LiSSA's `Knowledge.getContent()`). A byte-order mark stays in, as in Java.
- **Vectors:** bge-m3 vectors, read from RQ1's embedding cache. The key is LiSSA's `KeyGenerator` of the content (B4), and each value is a JSON float array. Using RQ1's vectors guarantees RQ1's exact candidates.
- **Similarity:** cosine computed as `ElementStore.cosineSimilarity` does it. Float products are summed in double precision in index order, the squared norms are summed in double, and the quotient is cast to float.
- **Ranking:** a stable sort by descending similarity over the files in name order, then the first 20, as `findSimilarWithDistancesByCosineSimilarity` does.

`check_rq1.py` compares the result with RQ1's retrieval-only `traceLinks` files and writes `data/candidates-rq1.json` only if all three projects match. `data/candidates-local.json` was computed with bge-m3 on Prabhjot's Air and is only for pilots; `run.py` prints a note when it's used.

Recall@20 of those pilot candidates: SMOS 0.447, eTour 0.666, iTrust 0.406. GPT-4o's OpenAI embeddings gave 0.420, 0.818 and 0.531. bge-m3 therefore lowers the recall ceiling on eTour and iTrust, which matters for interpreting RQ1.

## B3. Split (`split.py`, `data/split.json`)

- **Sizes:** per project, round(20%) of the requirements go to dev and the rest to test: SMOS 13/54, eTour 12/46, iTrust 26/105. That's 51 dev and 205 test requirements in total.
- **How it's drawn:** within each project, requirements with and without gold links are drawn separately (largest-remainder allocation). iTrust's 26 unlinked requirements are therefore spread proportionally (5 in dev).
- **Frozen:** the seed is 5187, fixed before any RQ2/RQ3 result existed. The script refuses to overwrite the file.

Variants and τ are chosen on dev; all reported comparisons use test.

## B4. Prompt, cache key and Ollama request (`lissa_compat.py`, `llm.py`)

- **Baseline prompt:** LiSSA's `ReasoningClassifier` with its system message, *"Your job is to determine if there is a traceability link between two artifacts of a system."*, and the `REASON_WITH_NAME` template.
  - The four placeholders are filled with the same `String.replace` calls in the same order.
  - Artifact types are "requirement" and "source code".
  - `selftest.py` checks the strings against the Java source.
- **Cache key:** LiSSA keys its answer cache with `KeyGenerator.generateKey(messages.toString())`, a version-3 UUID (MD5) of langchain4j 0.33's message list printed as text, for example `[SystemMessage { text = "…" }, UserMessage { name = null contents = [TextContent { text = "…" }] }]`.
  - The format was taken from the langchain4j 0.33.0 sources (`SystemMessage`, `UserMessage`, `TextContent`, `Utils.quoted`).
  - The hash function reproduces all 43 configuration identifiers in LiSSA's published GPT-4o result file names.
  - Because the keys match, the baseline's answers are read straight from RQ1's cache, and `check_rq1.py` confirms that every candidate pair's key exists there.
- **Ollama request:** `/api/chat` with options `temperature 0.0`, `seed 133742243`, `num_ctx 16384` and `num_predict 4096`. These are exactly the options of Rahul's `ChatLanguageModelProvider`. It uses 3 attempts and a 15-minute timeout, like langchain4j's client.
- **Answer cache:** one append-only JSONL file per model and chip. Each record holds the answer and Ollama's `total_duration`, `load_duration`, `prompt_eval_count`, `prompt_eval_duration`, `eval_count`, `eval_duration` and `done_reason`. A record is reused only when the key and the Ollama options match.
- **Never mixed across chips:** temperature 0 is not guaranteed to give bit-identical answers on different chips. `--lissa-cache` (reusing RQ1's answers) is therefore correct only on the laptop that ran RQ1. That's one reason to run everything on Rahul's Pro.

## B5. Answer parsing

- **Strict** (LiSSA's `isRelated`): the first `<trace>…</trace>` on one line, case-insensitive. It's a link if that text contains "yes"; a missing tag counts as no.
- **Tolerant:** the tag if there is one, otherwise a "yes" or "no" that opens the answer (ignoring leading punctuation such as `**`). Anything else counts as no.

Every run reports both, so effects of the added information can be separated from changes in format compliance. The `examples` variant shows `<trace>` answers, which may itself raise compliance.

## B6. The RQ2 variants (`variants.py`)

Each variant adds one piece of information. LiSSA's system message, first sentence, question, answer format and both artifacts stay byte-identical, which `selftest.py` checks for every variant. The layouts:

```
baseline    Below are two artifacts from the same software system. Is there a traceability link between (1) and (2)? Give your reasoning and then answer with 'yes' or 'no' enclosed in <trace> </trace>.
             (1) requirement: '''…''' 
             (2) source code: '''…''' 

definition  Below are two artifacts from the same software system.
            Definition: <prompts/definition.txt>
            Is there a traceability link between (1) and (2)? … (rest identical)

project     … About this system: <prompts/project-<project>.txt> …
examples    … Labelled examples from other software systems:
            <prompts/examples-<project>.txt> …

summary     <baseline prompt, unchanged>
             Structural summary of (2), extracted by a parser: '''<data/summaries-<project>.json[file]>''' 
```

- **definition:** one text for all projects.
- **project:** one text of about 45 words per project, written from the dataset READMEs and the use-case names. The SMOS text says that its requirements and identifiers are Italian.
- **examples:** two real labelled pairs per target project.
  - One linked pair, drawn from the first other project.
  - One retrieved-but-unlinked pair (a hard negative), drawn from the second other project.
  - Both come from those projects' dev slices, with requirements of at most 1,000 and code of at most 1,500 characters.
  - The seed is fixed per target and source project; `data/examples.json` records which pairs were chosen.
  - Each example shows the requirement, the code and `Answer: <trace>yes|no</trace>`.
- **summary:** from `javaparse.py` (B7), added on top of the full code, never instead of it.

**Length check** (`prepare_prompts.py`): the longest prompt of each variant is estimated at 3.5 characters per token. On the pilot, measured prompts had 4.0–5.0 characters per token, so 3.5 is a safety margin. The worst case is about 15,200 of the 15,360 available tokens (16,384 minus 1,024 for the answer), for the summary variant on eTour's largest file (`CardBC.java`, about 48,000 characters). At the measured ratio it is about 11,000–13,000 tokens.

## B7. Java parsing (`javaparse.py`)

The parser is tree-sitter-java (pinned versions). It recovers declarations even from files with syntax errors, and parses LiSSA's content of each file without the byte-order mark.
- **Coverage:** all 442 code files give declarations, except SMOS's `display.txt` (CSS) and `managerUser.txt` (JSP). eTour's `NewsTableModelTest.java` has syntax errors but its declarations are still found.
- **Summary:** for each class, interface or enum, its header (name, `extends`, `implements`) and the first sentence of its Javadoc. Then one line per field, constructor and method: its signature as written, plus the first sentence of its Javadoc (the description up to the first `@` tag).
  - Capped at 4,000 characters, ending with "... (N more lines)" when cut.
  - Files without declarations get "(no Java declarations found)".
- **Declared identifiers (RQ3):** the names of all types, methods, constructors and fields in the file.
- **Code of a cited element (RQ3):** the identifier is normalised first, dropping parameters and qualifiers (`Manager.delete(int)` becomes `delete`). Then the code shown to the checker is:
  - for methods and constructors: every one with that name (overloads included), with their Javadoc;
  - otherwise for a field: its declaration;
  - otherwise for a type: its header, Javadoc and member signatures.

## B8. RQ3 verification (`verify.py`)

This is an adaptation of the factored variant of Chain-of-Verification. It runs on the links a run accepted under the chosen parsing, and only on those, so verification can only remove links.

1. **Claims** (one call):
   - System message: *"You justify traceability links between requirements and source code."*
   - The prompt is `prompts/claims.txt` with the requirement and the full code. Ollama's structured output forces JSON `{"claims": [{"claim", "identifier"}]}`.
   - At most 5 claims are used. An answer that isn't valid JSON gives no claims, and is counted in `metrics.md`.
2. **Identifier check** (no LLM): if the cited identifier isn't declared in the file, the claim gets the verdict `identifier-not-declared`, which counts as not supported. This catches invented identifiers cheaply.
3. **Factored check** (one call per remaining claim):
   - System message: *"You check claims about source code."*
   - The prompt is `prompts/check.txt`, showing only the cited element's code (B7) and the claim. The requirement is never shown, so the check can't be swayed by the link it is meant to test.
   - JSON `{"reason", "verdict"}` with verdict ∈ {supported, unsupported, unclear}. The reason comes first, so the model reasons before it decides.
4. **Decision:** share = supported claims ÷ claims (0 if there are no claims). A link is kept if share ≥ τ.

Claims and checks are cached in the same answer cache. Their key adds the JSON schema to the hashed text, so they can never collide with classification answers.

`verify-<parser>/links.jsonl` keeps every link's claims, identifiers, verdicts, reasons and share, which the RQ4 error analysis needs. `metrics.json` holds the results without verification and for every τ.

## B9. Metrics (`evaluate.py`) and statistics (`analyze.py`)

- **Scoring set:** links are scored on the requirements a run judged (its slice). Gold links count only if their requirement is in that set.
- **Per project:** TP, FP, FN, precision, recall, F1 and F2, with LiSSA's formulas. The averages are the plain mean over projects and the mean weighted by gold links, as in LiSSA's paper.
  - `selftest.py` reproduces LiSSA's metrics from its own trace-link files for 29 of 33 GPT-4o runs.
  - The other 4 have corrupted CSVs. LiSSA's `saveTraceLinks` overwrites without truncating, so the tail of an older, longer file remains. LiSSA's metrics in `results-*.md` are correct, and RQ1's driver moves old files away, so RQ1's CSVs are clean.
- **Per requirement:** TP, FP, FN and F1. F1 is undefined (NaN) when a requirement has no gold links and no predictions.
- **Tests** (`analyze.py test`):
  - Each comparison pairs the per-requirement F1 of two results on the same requirements: two-sided Wilcoxon signed-rank (scipy defaults, zero differences dropped), α = 0.05.
  - Holm correction over the comparisons of one call.
  - Effect size: the matched-pairs rank-biserial correlation.
  - Requirements without gold links are excluded by default.
- **Choices on dev:**
  - `analyze.py variant` takes the highest average F1.
  - `analyze.py tau` takes the τ with the highest average F1; a tie goes to the smaller τ, which removes fewer links.

## B10. Provenance

Each run folder's `run.json` stores:
- the arguments and the Ollama options;
- the Ollama version and the model digest;
- the chip;
- SHA-256 hashes of the candidates file, the split, the variant's prompt and data files, and the harness code.

Two runs are comparable only if these match, except for the variant's own files.

## B11. RQ4 material produced along the way

- **Runtime and tokens:** `decisions.csv` has, for every pair, `total_seconds`, `prompt_eval_count`, `eval_count`, `done_reason`, the candidate's rank and cosine, and both decisions. The answer cache has every duration Ollama reports. `metrics.md` already gives the median and IQR per call and per requirement.
- **Error sample:** the three pools (false link kept, false link removed, correct link lost) come from comparing a run's links with its verified links and the gold standard. `links.jsonl` gives the claims and verdicts behind each removal.

## B12. Troubleshooting

| Symptom | Meaning / fix |
|---|---|
| `check_rq1`: candidates DIFFER | The embeddings or files differ from RQ1's: check that `--lissa-cache` points at RQ1's cache and that the datasets are unchanged (no `.DS_Store` in `UC/` or `CC/`) |
| `check_rq1`: fewer prompts found than pairs | Prompt text or cache key differs: check that the datasets and `lissa_compat.py` are unchanged |
| `check_rq1`: re-asked text differs | Same chip, but the answers changed: Ollama or the model changed since RQ1 (compare the digests in RQ1's `preflight-*.json`) |
| `run.py`: "not computed from RQ1's embeddings" | `data/candidates-rq1.json` is missing: run step 1 |
| `prepare_prompts.py`: TOO LONG | Shorten the added text, or lower `MAX_SUMMARY_CHARS` in `javaparse.py` |
| `<model> is not pulled` | `ollama pull <model>`, but only the same model as RQ1 (check its digest) |
| Many `no tag` answers in the progress lines | Expected for Qwen; that's why both parsings are reported |
| `verify.py`: many `identifier-not-declared` | The model cites names that aren't in the file (often from other files). That is the hallucination the check exists for, so report it as a result, not as a bug |
