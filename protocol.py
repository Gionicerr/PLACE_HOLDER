"""Shared data, chat format and numeric answer convention for experiment 2."""

import hashlib
import json
import os
import re
import tempfile
from decimal import Decimal
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "project_data.zip"
CONFIG = json.loads((ROOT / "configs" / "teachers.json").read_text(encoding="utf-8"))
NUMBER = r"[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?|\.[0-9]+)"
FINAL_LINE = re.compile(rf"####[ \t]+({NUMBER})")


def read_split(path, name, stage):
    filename = name + ".jsonl"
    if path.is_dir():
        raw = (path / filename).read_bytes()
    else:
        with ZipFile(path) as source:
            raw = source.read(filename)
    spec = CONFIG["data"][name]
    if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
        raise ValueError(f"{filename} differs from the agreed split")
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != spec["count"] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"Invalid row count or duplicate IDs in {filename}")
    for row in rows:
        if not row["question"].strip() or not re.fullmatch(NUMBER, row["reference_answer"]):
            raise ValueError(f"Invalid source row {row['id']}")
    return rows[:CONFIG["pilot"][name]] if stage == "pilot" else rows


def user_prompt(question):
    return question + "\n\n" + CONFIG["generation"]["instruction"]


def chat_prompt(question):
    system = CONFIG["generation"]["system"]
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt(question)}<|im_end|>\n<|im_start|>assistant\n")


def score(response, reference):
    lines = response.strip().splitlines()
    match = FINAL_LINE.fullmatch(lines[-1].strip()) if lines else None
    if not match or any(line.lstrip().startswith("####") for line in lines[:-1]):
        return None, False
    value = Decimal(match[1].replace(",", ""))
    return format(value, "f"), value == Decimal(reference.replace(",", ""))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                     dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_json(path, data):
    atomic_text(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def save_jsonl(path, rows):
    atomic_text(path, "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows))
