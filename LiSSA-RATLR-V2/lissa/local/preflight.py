#!/usr/bin/env python3
"""Preflight for local LiSSA runs: chooses the LLM and the embedding model for a whole run.

For each role the primary model is tried first and the fallback only if the primary is unusable:
not pulled, fails to load, does not fit completely on the GPU, or fails a probe request.
The choice is made once, before LiSSA starts, and does not change during the run: mixing LLMs would make
the results uninterpretable, and mixing embedding models (bge-m3: 1024 dimensions, llama3.1: 4096) would
break the cosine-similarity retrieval.

The probes send the same Ollama options as LiSSA (temperature 0, seed and num_ctx; for embeddings
num_batch = num_ctx), so the models stay loaded exactly as the run needs them. The LLM probe uses the
largest requirement and code file, which shows whether the longest LiSSA prompt fits into num_ctx.
"""

import argparse
import datetime
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request
import uuid

# ReasoningClassifier (use_system_message = true, prompt 0 = REASON_WITH_NAME), as in all GPT-4o req2code configs.
SYSTEM_MESSAGE = "Your job is to determine if there is a traceability link between two artifacts of a system."
PROMPT = (
    "Below are two artifacts from the same software system. Is there a traceability link between (1) and (2)? "
    "Give your reasoning and then answer with 'yes' or 'no' enclosed in <trace> </trace>.\n"
    " (1) {source_type}: '''{source_content}''' \n (2) {target_type}: '''{target_content}''' "
)
# Tokens the classifier needs after the prompt for its reasoning and the <trace> answer.
ANSWER_RESERVE = 1024


class Unusable(Exception):
    pass


def call(url, path, body=None, timeout=900):
    request = urllib.request.Request(
        url.rstrip("/") + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        raise Unusable(f"HTTP {e.code} on {path}: {e.read().decode(errors='replace').strip()}") from None
    except (urllib.error.URLError, TimeoutError) as e:
        raise Unusable(f"request to {path} failed: {e}") from None


def full_name(model):
    return model if ":" in model else model + ":latest"


def model_card(url, model, local_models):
    if full_name(model) not in local_models:
        raise Unusable(f"not pulled (run: ollama pull {model})")
    show = call(url, "/api/show", {"model": model})
    max_ctx = next((v for k, v in show.get("model_info", {}).items() if k.endswith(".context_length")), None)
    tags = local_models[full_name(model)]
    return {
        "model": model,
        "digest": tags.get("digest"),
        "details": show.get("details", tags.get("details", {})),
        "capabilities": show.get("capabilities", []),
        "max_context": max_ctx,
    }


def gpu_placement(url, model, require_full_gpu):
    loaded = [m for m in call(url, "/api/ps").get("models", []) if full_name(m.get("name", "")) == full_name(model)]
    if not loaded:
        raise Unusable("not loaded after the probe request")
    size, size_vram = loaded[0].get("size", 0), loaded[0].get("size_vram", 0)
    fraction = size_vram / size if size else 0.0
    if require_full_gpu and fraction < 0.999:
        raise Unusable(f"only {fraction:.0%} on the GPU ({size_vram / 2**30:.1f} of {size / 2**30:.1f} GiB)")
    return {"gpu_fraction": round(fraction, 4), "loaded_size_gib": round(size / 2**30, 2),
            "loaded_context": loaded[0].get("context_length")}


def try_embedding(url, model, num_ctx, local_models, require_full_gpu, probe_text):
    card = model_card(url, model, local_models)
    ctx = min(num_ctx, card["max_context"]) if card["max_context"] else num_ctx
    start = time.time()
    response = call(url, "/api/embed", {
        "model": model,
        "input": probe_text or "Delete a cultural object in the system.",
        "truncate": True,
        "options": {"num_ctx": ctx, "num_batch": ctx},
    })
    vectors = response.get("embeddings") or [[]]
    if not vectors[0]:
        raise Unusable(f"no embedding returned: {response}")
    card.update(gpu_placement(url, model, require_full_gpu))
    card.update({
        "num_ctx_requested": num_ctx,
        "num_ctx": ctx,
        "dimensions": len(vectors[0]),
        "probe": {"tokens": response.get("prompt_eval_count"), "truncated": response.get("prompt_eval_count", 0) >= ctx,
                  "seconds": round(time.time() - start, 1)},
    })
    return card


def try_llm(url, model, num_ctx, seed, local_models, require_full_gpu, source_text, target_text):
    card = model_card(url, model, local_models)
    if "completion" not in card["capabilities"]:
        raise Unusable(f"no chat/completion capability ({card['capabilities']})")
    ctx = min(num_ctx, card["max_context"]) if card["max_context"] else num_ctx
    prompt = PROMPT.format(
        source_type="requirement", source_content=source_text or "Delete a cultural object in the system.",
        target_type="source code", target_content=target_text or "public class CulturalHeritage {}",
    )
    start = time.time()
    response = call(url, "/api/chat", {
        "model": model,
        # The nonce keeps Ollama from reusing a cached prompt prefix, so prompt_eval_count is the full prompt length
        "messages": [{"role": "system", "content": f"[{uuid.uuid4().hex[:8]}] {SYSTEM_MESSAGE}"},
                     {"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.0, "seed": seed, "num_ctx": ctx, "num_predict": 1},
    })
    if "message" not in response:
        raise Unusable(f"unexpected chat response: {response}")
    card.update(gpu_placement(url, model, require_full_gpu))
    prompt_tokens = response.get("prompt_eval_count", 0)
    card.update({
        "num_ctx_requested": num_ctx,
        "num_ctx": ctx,
        "thinking_capable": "thinking" in card["capabilities"],
        "probe": {"longest_prompt_tokens": prompt_tokens, "fits": prompt_tokens + ANSWER_RESERVE <= ctx,
                  "seconds": round(time.time() - start, 1)},
    })
    return card


def choose(role, candidates, attempt):
    rejected = []
    for model in [c for c in candidates if c]:
        try:
            print(f"[{role}] trying {model} ...", file=sys.stderr, flush=True)
            card = attempt(model)
            card["fallback_used"] = model != candidates[0]
            card["rejected"] = rejected
            return card
        except Unusable as e:
            print(f"[{role}] {model} rejected: {e}", file=sys.stderr, flush=True)
            rejected.append({"model": model, "reason": str(e)})
    return {"model": None, "rejected": rejected}


def largest_file(directories):
    files = [f for d in directories for f in pathlib.Path(d).rglob("*") if f.is_file()]
    return max(files, key=lambda f: f.stat().st_size) if files else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--llm", required=True)
    parser.add_argument("--llm-fallback", default="")
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--embedding-fallback", default="")
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=133742243)
    parser.add_argument("--allow-partial-gpu", action="store_true")
    parser.add_argument("--probe-source-dirs", nargs="*", default=[], help="requirements; the largest file is probed")
    parser.add_argument("--probe-target-dirs", nargs="*", default=[], help="code; the largest file is probed")
    parser.add_argument("--out", required=True, help="JSON file for the chosen models")
    args = parser.parse_args()

    url = args.ollama_url
    try:
        version = call(url, "/api/version", timeout=10).get("version")
        local_models = {full_name(m["name"]): m for m in call(url, "/api/tags", timeout=30).get("models", [])}
    except Unusable as e:
        sys.exit(f"Ollama is not reachable at {url} ({e}). Start the Ollama app or 'ollama serve'.")
    print(f"Ollama {version} at {url}", file=sys.stderr)

    source_file, target_file = largest_file(args.probe_source_dirs), largest_file(args.probe_target_dirs)
    source_text = source_file.read_text(errors="replace") if source_file else None
    target_text = target_file.read_text(errors="replace") if target_file else None
    require_full_gpu = not args.allow_partial_gpu

    # Same order as LiSSA: embeddings first, then classification
    embedding = choose("embedding", [args.embedding, args.embedding_fallback], lambda m: try_embedding(
        url, m, args.num_ctx, local_models, require_full_gpu, target_text))
    llm = choose("llm", [args.llm, args.llm_fallback], lambda m: try_llm(
        url, m, args.num_ctx, args.seed, local_models, require_full_gpu, source_text, target_text))

    result = {
        "checked_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "ollama_url": url,
        "ollama_version": version,
        "require_full_gpu": require_full_gpu,
        "probe_files": {"source": str(source_file) if source_file else None,
                        "target": str(target_file) if target_file else None},
        "embedding": embedding,
        "llm": llm,
    }
    pathlib.Path(args.out).write_text(json.dumps(result, indent=2) + "\n")

    print("", file=sys.stderr)
    for role, card in (("embedding", embedding), ("llm", llm)):
        if not card["model"]:
            print(f"[{role}] NO USABLE MODEL - see the reasons above", file=sys.stderr)
            continue
        if card["fallback_used"]:
            print(f"[{role}] !! FALLBACK IN USE: {card['model']} (primary rejected: {card['rejected'][0]['reason']})",
                  file=sys.stderr)
        print(f"[{role}] {card['model']}: num_ctx {card['num_ctx']}, {card['gpu_fraction']:.0%} GPU, "
              f"{card['loaded_size_gib']} GiB loaded", file=sys.stderr)
    if llm["model"] and not llm["probe"]["fits"]:
        print(f"[llm] WARNING: the longest prompt ({target_file.name if target_file else 'probe'}) needs "
              f"{llm['probe']['longest_prompt_tokens']} tokens; with {ANSWER_RESERVE} for the answer it exceeds "
              f"num_ctx {llm['num_ctx']}, so Ollama will truncate it.", file=sys.stderr)
    if llm["model"] and llm["thinking_capable"]:
        print(f"[llm] note: {llm['model']} is a thinking model; LiSSA does not set 'think', so Ollama's "
              "default applies (thinking text is not part of the answer LiSSA parses).", file=sys.stderr)
    sys.exit(0 if embedding["model"] and llm["model"] else 2)


if __name__ == "__main__":
    main()
