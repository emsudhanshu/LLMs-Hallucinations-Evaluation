from __future__ import annotations

from dataclasses import dataclass
import math
import re
from collections import Counter

from .dataset import MCQSample, load_mcq_split


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source_question_id: str
    score: float


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _build_chunk_text(sample: MCQSample) -> str:
    parts = [
        f"Question: {sample.question}",
        f"Options: A) {sample.options['A']} B) {sample.options['B']} C) {sample.options['C']} D) {sample.options['D']}",
    ]
    if sample.explanation:
        parts.append(f"Explanation: {sample.explanation}")
    if sample.subject_name:
        parts.append(f"Subject: {sample.subject_name}")
    if sample.topic_name:
        parts.append(f"Topic: {sample.topic_name}")
    return "\n".join(parts)


class LocalMedicalRetriever:
    def __init__(self, corpus_split: str = "train", max_corpus_size: int = 5000) -> None:
        self.samples = load_mcq_split(corpus_split, require_labels=False)[:max_corpus_size]
        self.documents = [_build_chunk_text(sample) for sample in self.samples]
        self.doc_term_counts = [Counter(self._tokenize(doc)) for doc in self.documents]
        self.doc_norms = [self._norm(counts) for counts in self.doc_term_counts]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token.lower() for token in TOKEN_PATTERN.findall(text)]

    @staticmethod
    def _norm(counts: Counter[str]) -> float:
        return math.sqrt(sum(value * value for value in counts.values()))

    @classmethod
    def _similarity(cls, query: str, doc_counts: Counter[str], doc_norm: float) -> float:
        query_counts = Counter(cls._tokenize(query))
        query_norm = cls._norm(query_counts)
        if query_norm == 0.0 or doc_norm == 0.0:
            return 0.0
        shared_terms = set(query_counts) & set(doc_counts)
        dot = sum(query_counts[token] * doc_counts[token] for token in shared_terms)
        return dot / (query_norm * doc_norm)

    def retrieve(self, query: str, *, k: int = 3) -> list[RetrievedChunk]:
        scores = [
            self._similarity(query, doc_counts, doc_norm)
            for doc_counts, doc_norm in zip(self.doc_term_counts, self.doc_norms)
        ]
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:k]
        return [
            RetrievedChunk(
                text=self.documents[idx],
                source_question_id=self.samples[idx].question_id,
                score=float(scores[idx]),
            )
            for idx in ranked_indices
        ]
