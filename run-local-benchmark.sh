#!/usr/bin/env bash
# Runs the LiSSA (V2) evaluation with local Ollama models instead of GPT-4o (see LOCAL-OLLAMA.md).
# Outputs go to LiSSA-RATLR-V2/lissa/results-local/<run-id>/; the GPT-4o results are only read.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LISSA="$ROOT/LiSSA-RATLR-V2/lissa"
JAR="target/ratlr-0.2.0-SNAPSHOT-jar-with-dependencies.jar"

# shellcheck source=LiSSA-RATLR-V2/lissa/local/models.conf
source "$LISSA/local/models.conf"

TASK=req2code
PROJECTS=""
SEEDS="133742243 35418170 83939039 -102959883 -294857830"
RETRIEVAL_ONLY=1
RETRIES=2
RUN_ID=""
FORCE_LLM=""
FORCE_EMBEDDING=""
NO_FALLBACK=0
DRY_RUN=0
SETUP=0

usage() {
    cat <<EOF
Usage: bash run-local-benchmark.sh [options]

  --setup               Pull the models listed in models.conf ('ollama pull'), then exit.
  --dry-run             Show models, configs and output folder without calling Ollama or LiSSA.
  --task T              req2code (default, RQ1) or doc2code.
  --projects "A B"      Subset of projects (req2code: SMOS eTour_en iTrust).
  --seeds "S1 S2"       Subset of the five LiSSA seeds (default: all five).
  --no-retrieval-only   Skip the retrieval-only runs (no LLM, recall ceiling).
  --llm MODEL           Use exactly this LLM, without fallback (e.g. gemma4:12b).
  --embedding MODEL     Use exactly this embedding model, without fallback.
  --no-fallback         Fail instead of falling back when a primary model is unusable.
  --allow-partial-gpu   Accept models that Ollama only partly offloads to the GPU.
  --retries N           Re-run a failed config up to N times, resuming from the cache (default: $RETRIES).
  --run-id ID           Output folder name; pass an earlier ID to resume an interrupted run.

Models, Ollama URL and context length: LiSSA-RATLR-V2/lissa/local/models.conf
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --setup) SETUP=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --task) TASK="$2"; shift 2 ;;
        --projects) PROJECTS="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --no-retrieval-only) RETRIEVAL_ONLY=0; shift ;;
        --llm) FORCE_LLM="$2"; shift 2 ;;
        --embedding) FORCE_EMBEDDING="$2"; shift 2 ;;
        --no-fallback) NO_FALLBACK=1; shift ;;
        --allow-partial-gpu) REQUIRE_FULL_GPU=0; shift ;;
        --retries) RETRIES="$2"; shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

case "$TASK" in
    req2code) [ -n "$PROJECTS" ] || PROJECTS="SMOS eTour_en iTrust"; BASELINE_DIR="results-r2c" ;;
    doc2code) [ -n "$PROJECTS" ] || PROJECTS="bigbluebutton mediastore teastore"; BASELINE_DIR="results-d2c"; RETRIEVAL_ONLY=0 ;;
    *) echo "Unknown task: $TASK" >&2; exit 1 ;;
esac

if [ "$SETUP" = 1 ]; then
    for model in "$LLM_PRIMARY" "$LLM_FALLBACK" "$EMBEDDING_PRIMARY" "$EMBEDDING_FALLBACK"; do
        echo ">> ollama pull $model"
        OLLAMA_HOST="$OLLAMA_URL" ollama pull "$model"
    done
    exit 0
fi

tag() { local t="${1%:latest}"; t="${t//:/-}"; echo "${t//\//-}"; }
json_get() {
    python3 -c 'import json, sys
value = json.load(open(sys.argv[1]))
for key in sys.argv[2:]:
    value = value[key]
print(value)' "$@"
}
java_major() { "$1/bin/java" -version 2>&1 | awk -F'"' '/version/ { split($2, v, "."); print (v[1] == "1" ? v[2] : v[1]); exit }'; }

# LiSSA targets JDK 21; a newer JDK is used only if no JDK 21 is found
select_java_home() {
    local candidate major newer=""
    for candidate in "${JAVA_HOME:-}" \
        "$(/usr/libexec/java_home -v 21 2>/dev/null || true)" \
        /opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home \
        /usr/local/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home \
        /opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home \
        /usr/local/opt/openjdk/libexec/openjdk.jdk/Contents/Home; do
        [ -n "$candidate" ] && [ -x "$candidate/bin/java" ] || continue
        major="$(java_major "$candidate")"
        if [ "$major" = "21" ]; then echo "$candidate"; return 0; fi
        if [ -z "$newer" ] && [ "${major:-0}" -gt 21 ] 2>/dev/null; then newer="$candidate"; fi
    done
    [ -n "$newer" ] && echo "$newer"
}

cd "$LISSA"

JAVA_HOME="$(select_java_home || true)"
if [ -z "$JAVA_HOME" ]; then
    echo "No JDK >= 21 found. Install one with: brew install openjdk@21" >&2
    exit 1
fi
export JAVA_HOME
[ "$(java_major "$JAVA_HOME")" = "21" ] || echo "Note: no JDK 21 found, using JDK $(java_major "$JAVA_HOME") ($JAVA_HOME)."

if [ -n "$FORCE_LLM" ]; then LLM_PRIMARY="$FORCE_LLM"; LLM_FALLBACK=""; fi
if [ -n "$FORCE_EMBEDDING" ]; then EMBEDDING_PRIMARY="$FORCE_EMBEDDING"; EMBEDDING_FALLBACK=""; fi
if [ "$NO_FALLBACK" = 1 ]; then LLM_FALLBACK=""; EMBEDDING_FALLBACK=""; fi

GENERATOR_FILTERS=(--task "$TASK" --projects $PROJECTS --seeds $SEEDS)
[ "$RETRIEVAL_ONLY" = 1 ] || GENERATOR_FILTERS+=(--no-retrieval-only)

if [ "$DRY_RUN" = 1 ]; then
    echo "Java:       $JAVA_HOME ($( [ -f "$JAR" ] && echo "jar built" || echo "jar will be built"))"
    echo "Ollama:     $OLLAMA_URL (not contacted)"
    echo "LLM:        $LLM_PRIMARY${LLM_FALLBACK:+ (fallback: $LLM_FALLBACK)}, num_ctx $NUM_CTX"
    echo "Embedding:  $EMBEDDING_PRIMARY${EMBEDDING_FALLBACK:+ (fallback: $EMBEDDING_FALLBACK)}"
    echo "Full GPU:   $( [ "$REQUIRE_FULL_GPU" = 1 ] && echo required || echo "not required")"
    echo "Output:     results-local/${RUN_ID:-<timestamp>_${TASK}_$(tag "$LLM_PRIMARY")__$(tag "$EMBEDDING_PRIMARY")}/"
    echo "Configs (assuming the primary models are usable):"
    python3 local/generate_configs_local.py --dry-run --llm "$LLM_PRIMARY" --embedding "$EMBEDDING_PRIMARY" \
        "${GENERATOR_FILTERS[@]}" | sed 's/^/  /'
    exit 0
fi

if [ ! -f "$JAR" ] || [ -n "$(find src pom.xml -newer "$JAR" -print 2>/dev/null | head -n 1)" ]; then
    echo ">> Building LiSSA (mvn -DskipTests package)"
    mvn -B -q -DskipTests package
fi

# 1. Choose the models for the whole run (primary, or fallback if the primary is unusable)
PROBE_ARGS=()
if [ "$TASK" = req2code ]; then
    PROBE_ARGS+=(--probe-source-dirs)
    for p in $PROJECTS; do PROBE_ARGS+=("datasets/req2code/$p/UC"); done
    PROBE_ARGS+=(--probe-target-dirs)
    for p in $PROJECTS; do PROBE_ARGS+=("datasets/req2code/$p/CC"); done
fi
[ "$REQUIRE_FULL_GPU" = 1 ] || PROBE_ARGS+=(--allow-partial-gpu)
PREFLIGHT="$(mktemp -t lissa-preflight)"
echo ">> Preflight (loads the models with num_ctx $NUM_CTX and checks GPU placement)"
python3 local/preflight.py --ollama-url "$OLLAMA_URL" --num-ctx "$NUM_CTX" --out "$PREFLIGHT" \
    --llm "$LLM_PRIMARY" --llm-fallback "$LLM_FALLBACK" \
    --embedding "$EMBEDDING_PRIMARY" --embedding-fallback "$EMBEDDING_FALLBACK" "${PROBE_ARGS[@]}" \
    || { echo "Preflight failed, nothing was run." >&2; exit 1; }

LLM="$(json_get "$PREFLIGHT" llm model)"
LLM_CTX="$(json_get "$PREFLIGHT" llm num_ctx)"
EMBEDDING="$(json_get "$PREFLIGHT" embedding model)"
EMBEDDING_CTX="$(json_get "$PREFLIGHT" embedding num_ctx)"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)_${TASK}_$(tag "$LLM")__$(tag "$EMBEDDING")}"
RUN_DIR="$LISSA/results-local/$RUN_ID"
if [ -f "$RUN_DIR/preflight.json" ]; then
    # Resuming: never mix models within one run folder
    PREVIOUS="$(json_get "$RUN_DIR/preflight.json" llm model) + $(json_get "$RUN_DIR/preflight.json" embedding model)"
    if [ "$PREVIOUS" != "$LLM + $EMBEDDING" ]; then
        echo "$RUN_DIR was run with $PREVIOUS, but the preflight chose $LLM + $EMBEDDING. Use a new --run-id." >&2
        exit 1
    fi
    mv "$PREFLIGHT" "$RUN_DIR/preflight-$(date +%Y%m%d-%H%M%S).json"
else
    mkdir -p "$RUN_DIR"
    mv "$PREFLIGHT" "$RUN_DIR/preflight.json"
fi
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/configs"

# 2. Configs for exactly these models (only LLM, embedding and cache folder differ from the GPT-4o configs)
CONFIGS="$(python3 local/generate_configs_local.py --list --llm "$LLM" --embedding "$EMBEDDING" \
    --llm-num-ctx "$LLM_CTX" --embedding-num-ctx "$EMBEDDING_CTX" "${GENERATOR_FILTERS[@]}")"
for cfg in $CONFIGS; do cp "$cfg" "$RUN_DIR/configs/"; done
{
    echo "commit: $(git -C "$ROOT" rev-parse HEAD)"
    echo "java:   $("$JAVA_HOME/bin/java" -version 2>&1 | head -n 1)"
    echo; git -C "$ROOT" status --short
} > "$RUN_DIR/code-version.txt"
git -C "$ROOT" diff -- "$LISSA/src" "$LISSA/pom.xml" > "$RUN_DIR/code.diff"
[ -s "$RUN_DIR/runs.tsv" ] || printf 'config\tstatus\tattempts\tseconds\n' > "$RUN_DIR/runs.tsv"

if [ -f .env ] && grep -q '^OLLAMA' .env; then
    echo "Note: $LISSA/.env sets OLLAMA_* variables; LiSSA prefers them over $OLLAMA_URL."
fi
export OLLAMA_HOST="$OLLAMA_URL" OLLAMA_EMBEDDING_HOST="$OLLAMA_URL"
KEEP_AWAKE=()
command -v caffeinate >/dev/null && KEEP_AWAKE=(caffeinate -is)

# 3. One JVM per config, so a failing config is detected (LiSSA itself only logs the exception) and retried
total=$(echo $CONFIGS | wc -w | tr -d ' ')
index=0
failed=0
for cfg in $CONFIGS; do
    index=$((index + 1))
    name="$(basename "$cfg")"
    if ls "$RUN_DIR/results-${name}_"*.md >/dev/null 2>&1; then
        echo "== [$index/$total] $name already done"
        continue
    fi
    status=FAILED
    attempt=0
    start=$(date +%s)
    while [ "$attempt" -le "$RETRIES" ]; do
        attempt=$((attempt + 1))
        # LiSSA writes into the working directory and does not truncate existing files: move leftovers away
        for leftover in "results-${name}_"*.md "traceLinks-${name}_"*.csv; do
            [ -e "$leftover" ] && mv "$leftover" "$RUN_DIR/logs/stale-$(date +%s)-$leftover"
        done
        echo ">> [$index/$total] $name (attempt $attempt)"
        "${KEEP_AWAKE[@]}" "$JAVA_HOME/bin/java" -jar "$JAR" eval -c "$cfg" 2>&1 \
            | tee -a "$RUN_DIR/logs/${name%.json}.log" || true
        if ls "results-${name}_"*.md >/dev/null 2>&1; then
            mv "results-${name}_"*.md "$RUN_DIR/"
            mv "traceLinks-${name}_"*.csv "$RUN_DIR/" 2>/dev/null || true
            status=OK
            break
        fi
        echo "!! $name produced no result, see $RUN_DIR/logs/${name%.json}.log" >&2
    done
    printf '%s\t%s\t%s\t%s\n' "$name" "$status" "$attempt" "$(( $(date +%s) - start ))" >> "$RUN_DIR/runs.tsv"
    # Evidence of GPU placement during the run
    { curl -s "$OLLAMA_URL/api/ps" || true; echo; } >> "$RUN_DIR/ollama-ps.jsonl"
    [ "$status" = OK ] || failed=$((failed + 1))
done

# 4. Summary next to the original GPT-4o results (read only)
python3 local/summarize.py --run-dir "$RUN_DIR" --baseline-dir "$BASELINE_DIR" || true
echo "Run folder: $RUN_DIR"
if [ "$failed" -gt 0 ]; then
    echo "$failed config(s) failed. Re-run with --run-id $RUN_ID to retry them (answers so far are cached)." >&2
    exit 1
fi
