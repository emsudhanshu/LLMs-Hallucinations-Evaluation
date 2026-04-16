from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from preprocessing import clean_explanation


LETTER_MAP = {1: "A", 2: "B", 3: "C", 4: "D", 0: "A", 5: ""}


@dataclass(frozen=True)
class QARecord:
    id: str
    question: str
    A: str
    B: str
    C: str
    D: str
    correct_letter: str
    correct_text: str
    raw_exp: str
    clean_exp: str
    subject: str
    topic: str
    choice_type: str

    def asdict(self) -> dict:
        return asdict(self)


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _normalize_correct_letter(raw_value: object) -> str:
    if raw_value is None:
        return ""
    if isinstance(raw_value, int):
        return LETTER_MAP.get(raw_value, "")
    text = str(raw_value).strip().upper()
    if text in {"A", "B", "C", "D"}:
        return text
    if text.isdigit():
        return LETTER_MAP.get(int(text), "")
    return ""


def _correct_text(correct_letter: str, options: dict[str, str]) -> str:
    return options.get(correct_letter, "")


def load_split(data_dir: str | Path, split: str, *, require_labels: bool = True) -> list[QARecord]:
    path = Path(data_dir) / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    rows: list[QARecord] = []
    for raw in _iter_jsonl(path):
        options = {
            "A": str(raw.get("opa", "")).strip(),
            "B": str(raw.get("opb", "")).strip(),
            "C": str(raw.get("opc", "")).strip(),
            "D": str(raw.get("opd", "")).strip(),
        }
        correct_letter = _normalize_correct_letter(raw.get("cop"))
        if require_labels and not correct_letter:
            continue

        row = QARecord(
            id=str(raw.get("id", "")).strip(),
            question=str(raw.get("question", "")).strip(),
            A=options["A"],
            B=options["B"],
            C=options["C"],
            D=options["D"],
            correct_letter=correct_letter,
            correct_text=_correct_text(correct_letter, options),
            raw_exp=str(raw.get("exp") or "").strip(),
            clean_exp=clean_explanation(raw.get("exp")),
            subject=str(raw.get("subject_name") or "").strip(),
            topic=str(raw.get("topic_name") or "").strip(),
            choice_type=str(raw.get("choice_type") or "").strip(),
        )
        rows.append(row)
    return rows


def sample_records(records: list[QARecord], sample_size: int | None) -> list[QARecord]:
    if not sample_size or sample_size <= 0 or sample_size >= len(records):
        return records
    return records[:sample_size]


def to_dataframe(records: list[QARecord]) -> pd.DataFrame:
    return pd.DataFrame([record.asdict() for record in records])

