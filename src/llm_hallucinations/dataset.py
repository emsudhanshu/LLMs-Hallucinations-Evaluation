from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import random
from typing import Iterable

from .config import DATA_DIR


OPTION_KEYS = ("opa", "opb", "opc", "opd")
OPTION_LABELS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class MCQSample:
    question_id: str
    question: str
    options: dict[str, str]
    correct_answer: str
    subject_name: str | None
    topic_name: str | None
    explanation: str | None
    choice_type: str | None


def _parse_correct_option(raw_value: object) -> str | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, int):
        if 1 <= raw_value <= 4:
            return OPTION_LABELS[raw_value - 1]
        if 0 <= raw_value <= 3:
            return OPTION_LABELS[raw_value]
    if isinstance(raw_value, str):
        normalized = raw_value.strip().upper()
        if normalized in OPTION_LABELS:
            return normalized
        if normalized.isdigit():
            return _parse_correct_option(int(normalized))
    return None


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_mcq_split(split: str, *, require_labels: bool = True) -> list[MCQSample]:
    path = DATA_DIR / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {path}")

    samples: list[MCQSample] = []
    for row in _iter_jsonl(path):
        correct_answer = _parse_correct_option(row.get("cop"))
        if require_labels and correct_answer is None:
            continue

        options = {
            label: str(row.get(key, "")).strip()
            for label, key in zip(OPTION_LABELS, OPTION_KEYS)
        }

        samples.append(
            MCQSample(
                question_id=str(row.get("id", "")),
                question=str(row.get("question", "")).strip(),
                options=options,
                correct_answer=correct_answer or "",
                subject_name=row.get("subject_name"),
                topic_name=row.get("topic_name"),
                explanation=row.get("exp"),
                choice_type=row.get("choice_type"),
            )
        )
    return samples


def sample_mcq_questions(
    split: str,
    *,
    sample_size: int,
    seed: int = 42,
    require_labels: bool = True,
) -> list[MCQSample]:
    samples = load_mcq_split(split, require_labels=require_labels)
    if sample_size <= 0 or sample_size >= len(samples):
        return samples

    rng = random.Random(seed)
    return rng.sample(samples, sample_size)

