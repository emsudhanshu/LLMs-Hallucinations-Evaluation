from __future__ import annotations

import re


ANSWER_PREFIX_PATTERN = re.compile(
    r"^\s*(?:ans(?:wer)?\.?\s*[-:()]?\s*[a-d0-4]\)?\.?\s*|ans(?:wer)?\s+is\s+['\"]?[a-d0-4]['\"]?\s*(?:i\.e\.)?\s*)",
    re.IGNORECASE,
)
REFERENCE_PATTERN = re.compile(
    r"\b(?:ref|reference|harrison'?s|robbins|guyton|ganong)\b[^.:\n]{0,80}",
    re.IGNORECASE,
)
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def clean_explanation(text: str | None) -> str:
    if not text:
        return ""

    cleaned = text.replace("\n", " ").replace("\t", " ")
    cleaned = ANSWER_PREFIX_PATTERN.sub("", cleaned)
    cleaned = REFERENCE_PATTERN.sub("", cleaned)
    cleaned = cleaned.replace("*", " ")
    cleaned = normalize_whitespace(cleaned)
    return cleaned

