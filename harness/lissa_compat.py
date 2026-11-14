"""Exact Python copies of the LiSSA V2 behaviour that the harness relies on.

The harness must send byte-identical prompts to LiSSA's (so RQ1's cached answers can be reused and the baseline is
LiSSA's own) and must score links exactly as LiSSA does. Each function names the Java code it mirrors;
selftest.py and check_rq1.py test these copies against LiSSA's own files.
"""

import hashlib
import json
import pathlib
import re
import uuid

HARNESS = pathlib.Path(__file__).resolve().parent
ROOT = HARNESS.parent  # LiSSA-Replication-Package
LISSA = ROOT / "LiSSA-RATLR-V2" / "lissa"
DATASETS = LISSA / "datasets" / "req2code"
PROJECTS = ("SMOS", "eTour_en", "iTrust")

# Artifact(identifier, ArtifactType, content): type.toString().replace("_", " ").toLowerCase()
SOURCE_TYPE = "requirement"
TARGET_TYPE = "source code"

# ReasoningClassifier with use_system_message = true and prompt 0 (REASON_WITH_NAME), as in every req2code config
SYSTEM_MESSAGE = "Your job is to determine if there is a traceability link between two artifacts of a system."
COT_PROMPT = (
    "Below are two artifacts from the same software system. Is there a traceability link between (1) and (2)? "
    "Give your reasoning and then answer with 'yes' or 'no' enclosed in <trace> </trace>.\n"
    " (1) {source_type}: '''{source_content}''' \n (2) {target_type}: '''{target_content}''' "
)

# Ollama options of RQ1 (ChatLanguageModelProvider after Rahul's change; seed = first significance seed)
SEED = 133742243
NUM_CTX = 16384
NUM_PREDICT = 4096


# --- Artifacts -----------------------------------------------------------------------------------------------------


def read_artifact(path):
    """TextArtifactProvider reads the file as UTF-8; Knowledge.getContent() turns "\\r\\n" into "\\n".
    The bytes are decoded directly: Python's text mode would also turn a lone "\\r" into "\\n", Java does not.
    A UTF-8 byte-order mark stays in the content, as in Java."""
    return path.read_bytes().decode("utf-8", errors="replace").replace("\r\n", "\n")


def load_artifacts(folder):
    """{file name: content}, ordered by file name like TextArtifactProvider.getArtifacts()."""
    files = sorted((p for p in pathlib.Path(folder).iterdir() if p.is_file()), key=lambda p: p.name)
    hidden = [p.name for p in files if p.name.startswith(".")]
    if hidden:
        # LiSSA would load them as artifacts too; RQ1 refused to run with them, so the harness does as well
        raise ValueError(f"hidden files in {folder}: {hidden}; delete them first")
    return {p.name: read_artifact(p) for p in files}


def link_id(file_name):
    """ReqCodePostprocessor: everything before the first dot ("UC10E1.txt" -> "UC10E1")."""
    return file_name[: file_name.index(".")]


def load_gold(project):
    """Statistics.generateStatistics: answer.csv without header, 'source,target' per line."""
    lines = (DATASETS / project / "answer.csv").read_bytes().decode("utf-8").splitlines()
    return {tuple(line.split(",")[:2]) for line in lines if line}


# --- Prompts and cache keys ----------------------------------------------------------------------------------------


def fill(template, source_content, target_content, source_type=SOURCE_TYPE, target_type=TARGET_TYPE):
    """Same String.replace calls, in the same order, as ReasoningClassifier.classify(Element, Element)."""
    return (
        template.replace("{source_type}", source_type)
        .replace("{source_content}", source_content)
        .replace("{target_type}", target_type)
        .replace("{target_content}", target_content)
    )


def messages_to_string(system, user):
    """Java List.toString() of the messages, using langchain4j 0.33's SystemMessage/UserMessage/TextContent.toString()
    and Utils.quoted(). LiSSA hashes this string to get the cache key."""
    parts = [] if system is None else [f'SystemMessage {{ text = "{system}" }}']
    parts.append(f'UserMessage {{ name = null contents = [TextContent {{ text = "{user}" }}] }}')
    return "[" + ", ".join(parts) + "]"


def generate_key(text):
    """KeyGenerator.generateKey: UUID.nameUUIDFromBytes (MD5 with version-3 bits) of the text, "\\r\\n" -> "\\n"."""
    digest = hashlib.md5(text.replace("\r\n", "\n").encode("utf-8")).digest()
    return str(uuid.UUID(bytes=digest, version=3))


def message_key(system, user):
    return generate_key(messages_to_string(system, user))


# --- Answers -------------------------------------------------------------------------------------------------------

# Java's "." stops at any line terminator; CASE_INSENSITIVE without UNICODE_CASE only folds ASCII
_TRACE = re.compile(r"<trace>([^\n\r\u0085  ]*?)</trace>", re.IGNORECASE | re.ASCII)


def has_trace_tag(answer):
    return _TRACE.search(answer) is not None


def is_related_cot(answer):
    """ReasoningClassifier.isRelated: first <trace>...</trace> on one line; a link if it contains 'yes'.
    No tag counts as 'no'."""
    match = _TRACE.search(answer)
    return match is not None and "yes" in match.group().lower()


_LEADING_ANSWER = re.compile(r"\W*(yes|no)\b", re.IGNORECASE | re.ASCII)


def is_related_tolerant(answer):
    """Not LiSSA's rule: the <trace> answer if there is one, otherwise a 'yes' or 'no' that opens the answer
    ("Yes, there is a traceability link ..."). Used next to the strict rule, to separate a model's judgement from
    its compliance with the answer format."""
    if has_trace_tag(answer):
        return is_related_cot(answer)
    match = _LEADING_ANSWER.match(answer)
    return match is not None and match.group(1).lower() == "yes"


PARSERS = {"strict": is_related_cot, "tolerant": is_related_tolerant}


def lissa_cache_file(cache_dir, model, classifier="ReasoningClassifier"):
    """CacheManager.getCache(name): '<Classifier>_<model>' with ':' replaced by '__', plus '.json'."""
    return pathlib.Path(cache_dir) / f"{classifier}_{model}.json".replace(":", "__")


def read_lissa_cache(path):
    """Cache: a JSON object {key: value}. LLM answers are plain strings; embeddings are JSON-encoded float arrays."""
    return json.loads(pathlib.Path(path).read_bytes().decode("utf-8"))


# --- Metrics -------------------------------------------------------------------------------------------------------


def counts(predicted, gold):
    """True positives, false positives and false negatives, as ClassificationMetricsCalculator counts them."""
    tp = len(predicted & gold)
    return tp, len(predicted) - tp, len(gold) - tp


def scores(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    def f_beta(beta):
        denominator = beta**2 * precision + recall
        return (1 + beta**2) * precision * recall / denominator if denominator else 0.0

    return {"p": precision, "r": recall, "f1": f_beta(1), "f2": f_beta(2)}


def read_trace_links(path):
    """Statistics.saveTraceLinks: 'source,target' per line, no header."""
    text = pathlib.Path(path).read_text(encoding="utf-8")
    return {tuple(line.split(",")) for line in text.splitlines() if line}


def model_tag(model):
    return model.removesuffix(":latest").replace(":", "-").replace("/", "-")
