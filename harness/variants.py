"""Prompt variants for RQ2. A variant turns one candidate pair into (system message, user message).

'baseline' is LiSSA's chain-of-thought prompt, byte for byte, so its cache keys match RQ1's. Each RQ2 variant adds one
piece of information and changes nothing else: LiSSA's system message, question, answer format and artifacts stay
exactly as they are. The added texts come from prompts/ and data/ (see prompts/README.md).

  definition  a precise definition of a trace link, after the first sentence
  project     a short description of the system, after the first sentence
  examples    two labelled pairs from the other projects, after the first sentence
  summary     a structural summary of the code file (types, fields, signatures, Javadoc), after the code
"""

import functools
import json
from dataclasses import dataclass

import lissa_compat as lc

HEAD = "Below are two artifacts from the same software system."
QUESTION = lc.COT_PROMPT[len(HEAD) + 1:]  # "Is there a traceability link between (1) and (2)? ... '''{target_content}''' "
assert lc.COT_PROMPT.startswith(HEAD + " ")
PROMPTS = lc.HARNESS / "prompts"
DATA = lc.HARNESS / "data"


@dataclass(frozen=True)
class Pair:
    project: str
    requirement: str  # file name, e.g. "UC1.txt"
    code: str  # file name, e.g. "DBCulturalHeritage.java"
    requirement_text: str
    code_text: str


@functools.cache
def prompt_text(name):
    return (PROMPTS / name).read_text(encoding="utf-8").rstrip("\n")


@functools.cache
def summaries(project):
    return json.loads((DATA / f"summaries-{project}.json").read_text(encoding="utf-8"))


def _with_information(label, information, pair):
    question = lc.fill(QUESTION, pair.requirement_text, pair.code_text)
    return lc.SYSTEM_MESSAGE, f"{HEAD}\n{label}: {information}\n{question}"


def baseline(pair):
    return lc.SYSTEM_MESSAGE, lc.fill(lc.COT_PROMPT, pair.requirement_text, pair.code_text)


def definition(pair):
    return _with_information("Definition", prompt_text("definition.txt"), pair)


def project(pair):
    return _with_information("About this system", prompt_text(f"project-{pair.project}.txt"), pair)


def examples(pair):
    return _with_information("Labelled examples from other software systems",
                             "\n" + prompt_text(f"examples-{pair.project}.txt"), pair)


def summary(pair):
    system, user = baseline(pair)
    return system, f"{user}\n Structural summary of (2), extracted by a parser: '''{summaries(pair.project)[pair.code]}''' "


VARIANTS = {"baseline": baseline, "definition": definition, "summary": summary, "project": project,
            "examples": examples}


def files_used(variant, projects):
    """The prompt and data files a variant reads, for the provenance record of a run."""
    per_variant = {
        "baseline": [],
        "definition": ["prompts/definition.txt"],
        "summary": [f"data/summaries-{p}.json" for p in projects],
        "project": [f"prompts/project-{p}.txt" for p in projects],
        "examples": [f"prompts/examples-{p}.txt" for p in projects],
    }
    return per_variant[variant]
