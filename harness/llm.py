"""Talking to Ollama, and the answer cache shared by run.py (RQ2) and verify.py (RQ3)."""

import datetime
import json
import platform
import re
import subprocess
import time
import urllib.error
import urllib.request

import lissa_compat as lc

# The Ollama options of RQ1 (ChatLanguageModelProvider after Rahul's change). Every harness call uses them.
OPTIONS = {"temperature": 0.0, "seed": lc.SEED, "num_predict": lc.NUM_PREDICT, "num_ctx": lc.NUM_CTX}
TIMING_FIELDS = ("total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count",
                 "eval_duration")


def device_name():
    """The chip that produces the answers, e.g. 'apple-m5' (Prabhjot's Air) or 'apple-m5-pro' (Rahul's Pro).
    Temperature 0 is not guaranteed to be bit-identical across chips, so answers from different chips are cached
    separately and never mixed in one run."""
    try:
        command = ["sysctl", "-n", "machdep.cpu.brand_string"]
        name = subprocess.run(command, capture_output=True, text=True).stdout.splitlines()[0]
    except (OSError, IndexError):
        name = platform.machine()
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def ollama(url, path, body=None, timeout=900):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url.rstrip("/") + path, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def ask(url, model, system, user, schema=None, attempts=3):
    """The chat request langchain4j 0.33's OllamaChatModel sends for LiSSA (null fields omitted), with the same
    15-minute timeout and 3 attempts. 'schema' (RQ3 only) makes Ollama return JSON of that shape."""
    messages = ([] if system is None else [{"role": "system", "content": system}]) + [{"role": "user", "content": user}]
    body = {"model": model, "messages": messages, "options": OPTIONS, "stream": False}
    if schema is not None:
        body["format"] = schema
    for attempt in range(1, attempts + 1):
        try:
            start = time.monotonic()
            response = ollama(url, "/api/chat", body)
            record = {"answer": response["message"]["content"], "done_reason": response.get("done_reason"),
                      "wall_seconds": round(time.monotonic() - start, 3)}
            record.update({field: response.get(field) for field in TIMING_FIELDS})
            return record
        except (urllib.error.URLError, TimeoutError, ConnectionError, KeyError) as error:
            if attempt == attempts:
                raise
            print(f"   !! attempt {attempt} failed ({error}); retrying", flush=True)
            time.sleep(10 * attempt)


def call_key(system, user, schema=None):
    """LiSSA's cache key for a plain chat call; calls with a JSON schema get the schema added to the hashed text."""
    text = lc.messages_to_string(system, user)
    if schema is not None:
        text += "\nformat=" + json.dumps(schema, sort_keys=True)
    return lc.generate_key(text)


class AnswerCache:
    """Append-only JSONL per model and chip. A record is reused only if it was produced with the same Ollama options."""

    def __init__(self, model, device):
        self.model, self.device = model, device
        self.path = lc.HARNESS / "cache" / f"{lc.model_tag(model)}__{device}.jsonl"
        self.records = {}
        if self.path.exists():
            text = self.path.read_text(encoding="utf-8")
            damaged = 0
            for line in text.splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    damaged += 1  # cut off by a disconnect; that call is simply made again
                    continue
                if record["options"] == OPTIONS:
                    self.records[record["key"]] = record
            if damaged:
                print(f"Note: skipped {damaged} damaged line(s) in {self.path.name}")
            if text and not text.endswith("\n"):
                with open(self.path, "a", encoding="utf-8") as file:
                    file.write("\n")  # so the next record starts on its own line

    def get(self, key):
        return self.records.get(key)

    def add(self, record):
        self.path.parent.mkdir(exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.records[record["key"]] = record

    def ask(self, url, system, user, schema=None, **context):
        """The cached answer for this call, or a new one from Ollama (then saved with its timing and 'context')."""
        key = call_key(system, user, schema)
        record = self.get(key)
        if record is None:
            record = {"key": key, "model": self.model, "options": OPTIONS, **context,
                      **ask(url, self.model, system, user, schema), "device": self.device,
                      "machine": platform.node(), "at": datetime.datetime.now().isoformat(timespec="seconds")}
            self.add(record)
            record = dict(record, fresh=True)
        return record
