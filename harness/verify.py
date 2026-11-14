#!/usr/bin/env python3
"""RQ3: claim-level verification of the links a run accepted (an adaptation of factored Chain-of-Verification).

For every accepted link of a run (decisions.csv, with the strict or the tolerant parsing):
  1. Claims: the LLM sees the requirement and the code and returns up to 5 atomic claims as JSON, each citing one
     identifier declared in the code (prompts/claims.txt).
  2. Identifier check, no LLM: the cited class, method or field must be declared in the file (javaparse.py);
     otherwise the claim counts as not supported.
  3. Factored check: a second call sees only the cited element's code and the claim, never the requirement, and
     answers supported, unsupported or unclear (prompts/check.txt).
  4. A link is kept if the share of supported claims is at least tau. A link without claims has share 0.
Verification only removes links, so it can raise precision but never recall.

    python3 harness/verify.py --run harness/runs/<run id> --parser strict [--hours 6] [--dry-run]

Writes <run>/verify-<parser>/links.jsonl (claims and verdicts per link) and metrics.json / metrics.md with the
results for every tau in TAUS. Tau is chosen on the dev slice (analyze.py tau) and then applied to the test slice.
"""

import argparse
import csv
import json
import pathlib
import sys
import time

import evaluate
import javaparse
import lissa_compat as lc
import llm
from variants import prompt_text

TAUS = [0.2, 0.4, 0.6, 0.8, 1.0]
MAX_CLAIMS = 5
SUPPORTED = {"supported"}  # 'unclear' does not count as support
CLAIMS_SYSTEM = "You justify traceability links between requirements and source code."
CHECK_SYSTEM = "You check claims about source code."
CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {"claims": {"type": "array", "maxItems": MAX_CLAIMS, "items": {
        "type": "object", "properties": {"claim": {"type": "string"}, "identifier": {"type": "string"}},
        "required": ["claim", "identifier"]}}},
    "required": ["claims"],
}
CHECK_SCHEMA = {
    "type": "object",
    "properties": {"reason": {"type": "string"},
                   "verdict": {"type": "string", "enum": ["supported", "unsupported", "unclear"]}},
    "required": ["reason", "verdict"],
}


def fill(template, **values):
    for name, value in values.items():
        template = template.replace("{" + name + "}", value)
    return template


def parse_json(answer, field):
    try:
        value = json.loads(answer)[field]
        return value
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def accepted_links(run_dir, parser):
    with open(run_dir / "decisions.csv", newline="") as file:
        rows = list(csv.DictReader(file))
    return rows, [row for row in rows if row[f"related_{parser}"] == "True"]


def verify_link(row, cache, url, java_files, texts):
    """Claims, identifier checks and factored checks of one link. Returns the link record and the number of fresh calls."""
    project, requirement, code = row["project"], row["requirement"], row["code"]
    java = java_files[project][code]
    context = {"project": project, "requirement": requirement, "code": code}
    user = fill(prompt_text("claims.txt"), requirement=texts[project]["UC"][requirement],
                code=texts[project]["CC"][code])
    answer = cache.ask(url, CLAIMS_SYSTEM, user, CLAIMS_SCHEMA, kind="claims", **context)
    fresh = int(bool(answer.get("fresh")))
    claims = parse_json(answer["answer"], "claims")
    record = {**context, "claims_json_ok": claims is not None, "claims": []}
    for claim in (claims or [])[:MAX_CLAIMS]:
        text, identifier = str(claim.get("claim", "")), str(claim.get("identifier", ""))
        element = javaparse.element_code(java, identifier)
        if element is None:
            record["claims"].append({"claim": text, "identifier": identifier, "verdict": "identifier-not-declared"})
            continue
        user = fill(prompt_text("check.txt"), identifier=javaparse.normalize_identifier(identifier),
                    element=element, claim=text)
        check = cache.ask(url, CHECK_SYSTEM, user, CHECK_SCHEMA, kind="check", **context)
        fresh += int(bool(check.get("fresh")))
        verdict = parse_json(check["answer"], "verdict")
        record["claims"].append({"claim": text, "identifier": identifier,
                                 "verdict": verdict if verdict in ("supported", "unsupported", "unclear") else "invalid",
                                 "reason": parse_json(check["answer"], "reason")})
    supported = sum(c["verdict"] in SUPPORTED for c in record["claims"])
    record["share"] = supported / len(record["claims"]) if record["claims"] else 0.0
    return record, fresh


def metrics_by_tau(rows, records, parser):
    """Metrics without verification, then for each tau; always on the requirements the run judged."""
    accepted = {(r["project"], r["requirement"], r["code"]) for r in rows if r[f"related_{parser}"] == "True"}
    share = {(r["project"], r["requirement"], r["code"]): r["share"] for r in records}
    judged = {}
    for r in rows:
        judged.setdefault(r["project"], set()).add(lc.link_id(r["requirement"]))
    split = {"projects": {p: {"judged": sorted(reqs)} for p, reqs in judged.items()}}
    results = {}
    for tau in [None, *TAUS]:
        links = {p: set() for p in judged}
        for project, requirement, code in accepted:
            if tau is None or share[(project, requirement, code)] >= tau:
                links[project].add((lc.link_id(requirement), lc.link_id(code)))
        results["none" if tau is None else str(tau)] = evaluate.evaluate(links, "judged", split)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, type=pathlib.Path, help="a run folder made by run.py")
    parser.add_argument("--parser", choices=list(lc.PARSERS), required=True, help="which accepted links to verify")
    parser.add_argument("--model", help="verifier model (default: the run's model)")
    parser.add_argument("--hours", type=float, help="stop after this many hours; the same command continues")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--device", help="name of the chip for the cache file (default: detected)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_info = json.loads((args.run / "run.json").read_text())
    model = args.model or run_info["args"]["model"]
    device = args.device or llm.device_name()
    cache = llm.AnswerCache(model, device)
    rows, accepted = accepted_links(args.run, args.parser)
    projects = sorted({row["project"] for row in rows})
    texts = {p: {side: lc.load_artifacts(lc.DATASETS / p / side) for side in ("UC", "CC")} for p in projects}
    java_files = {p: {name: javaparse.parse(text) for name, text in texts[p]["CC"].items()} for p in projects}
    print(f"Device: {device}. {len(accepted)} accepted links ({args.parser} parsing) to verify with {model}.")
    if args.dry_run:
        return

    out = args.run / f"verify-{args.parser}"
    out.mkdir(exist_ok=True)
    deadline = time.monotonic() + args.hours * 3600 if args.hours else None
    records, fresh_calls, started = [], 0, time.monotonic()
    try:
        for index, row in enumerate(accepted, 1):
            if deadline and time.monotonic() >= deadline:
                print(f"Time limit reached after {fresh_calls} calls; run the same command to continue.")
                return
            record, fresh = verify_link(row, cache, args.ollama_url, java_files, texts)
            records.append(record)
            fresh_calls += fresh
            if fresh:
                verdicts = ", ".join(c["verdict"] for c in record["claims"]) or "no claims"
                eta = (len(accepted) - index) * (time.monotonic() - started) / index / 3600
                print(f"[{index}/{len(accepted)}] {row['project']} {row['requirement']} / {row['code']}: "
                      f"share {record['share']:.2f} ({verdicts}; about {eta:.1f} h left)", flush=True)
    except KeyboardInterrupt:
        sys.exit(f"\nInterrupted after {fresh_calls} calls; answers so far are cached. Run the same command to continue.")

    with open(out / "links.jsonl", "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    results = metrics_by_tau(rows, records, args.parser)
    (out / "metrics.json").write_text(json.dumps(results, indent=1) + "\n")
    verdicts = {}
    for record in records:
        for claim in record["claims"]:
            verdicts[claim["verdict"]] = verdicts.get(claim["verdict"], 0) + 1
    report = [f"# Verification of {args.run.name} ({args.parser} parsing, verifier {model} on {device})", "",
              f"- Links: {len(records)}; claims per link: "
              f"{sum(len(r['claims']) for r in records) / max(len(records), 1):.1f}; "
              f"invalid claims JSON: {sum(not r['claims_json_ok'] for r in records)}; verdicts: {verdicts}", ""]
    for tau, result in results.items():
        report += [f"## tau = {tau}" if tau != "none" else "## Without verification", "", evaluate.table(result), ""]
    (out / "metrics.md").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
