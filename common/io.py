"""Portable configuration, dataset validation and atomic run metadata."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
        temporary = Path(f.name)
    temporary.replace(path)


def read_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    return config


def repo_path(path):
    p = Path(os.path.expandvars(path)).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def load_rows(path, expected_rows=None):
    path = Path(path)
    rows = (
        [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        if path.suffix == ".jsonl"
        else json.loads(path.read_text())
    )
    if not isinstance(rows, list) or not rows:
        raise ValueError("Training data must be a nonempty JSON array or JSONL")
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} training rows, found {len(rows)}")
    result = []
    for i, row in enumerate(rows):
        question = row.get("question")
        answers = row.get(
            "answers", row.get("golds", row.get("answer", row.get("best_answer")))
        )
        if isinstance(answers, str):
            answers = answers.split(";")
        if not isinstance(answers, list):
            raise ValueError(f"Row {i}: references required")
        aliases = row.get("aliases") or []
        if not isinstance(aliases, list):
            raise ValueError(f"Row {i}: aliases must be a list")
        refs = list(
            dict.fromkeys(
                s.strip() for s in answers + aliases if isinstance(s, str) and s.strip()
            )
        )
        if not isinstance(question, str) or not question.strip() or not refs:
            raise ValueError(
                f"Row {i}: nonempty question and reference answers required"
            )
        result.append(
            dict(
                id=str(row.get("id", i)),
                question=question,
                answers=refs,
                best_answer=";".join(refs),
            )
        )
    if len({r["id"] for r in result}) != len(result):
        raise ValueError("Training IDs must be unique")
    return result


def freeze_run(output, config, data):
    record = dict(config=config, data_sha256=sha(data))
    path = Path(output) / "run_config.json"
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError(
            "Run configuration or data changed; use a new output directory"
        )
    save(path, record)
    return record
