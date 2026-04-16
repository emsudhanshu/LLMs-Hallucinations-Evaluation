from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data_loader import QARecord, load_split


@dataclass(frozen=True)
class RetrievalResult:
    record_id: str
    text: str
    score: float


class ExplanationRAG:
    def __init__(
        self,
        data_dir: str | Path,
        knowledge_split: str = "train",
        embedding_model: str = "all-MiniLM-L6-v2",
        knowledge_limit: int | None = 20000,
    ) -> None:
        self.data_dir = Path(data_dir)
        knowledge_records = [
            row for row in load_split(self.data_dir, knowledge_split, require_labels=False) if row.clean_exp
        ]
        self.knowledge_records = knowledge_records[:knowledge_limit] if knowledge_limit else knowledge_records
        self.embedding_model_name = embedding_model
        self.backend = "empty"
        self.embedder = None
        self.index = None
        self.matrix = None
        self.vectorizer = None
        self.documents = [self._document_text(row) for row in self.knowledge_records]

    @staticmethod
    def _document_text(row: QARecord) -> str:
        return (
            f"Question: {row.question}\n"
            f"Explanation: {row.clean_exp}\n"
            f"Subject: {row.subject}\n"
            f"Topic: {row.topic}"
        ).strip()

    def build(self) -> None:
        if not self.documents:
            self.backend = "empty"
            return

        try:
            import faiss
            from sentence_transformers import SentenceTransformer

            self.embedder = SentenceTransformer(self.embedding_model_name)
            embeddings = self.embedder.encode(
                self.documents,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self.matrix = np.asarray(embeddings, dtype="float32")
            self.index = faiss.IndexFlatIP(self.matrix.shape[1])
            self.index.add(self.matrix)
            self.backend = "faiss"
            return
        except Exception:
            pass

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=30000)
        self.matrix = self.vectorizer.fit_transform(self.documents)
        self._cosine_similarity = cosine_similarity
        self.backend = "tfidf"

    def retrieve(self, query: str, *, top_k: int = 3, exclude_ids: set[str] | None = None) -> list[RetrievalResult]:
        if self.backend == "empty":
            return []

        exclude_ids = exclude_ids or set()
        if self.backend == "faiss":
            query_vector = self.embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)
            scores, indices = self.index.search(np.asarray(query_vector, dtype="float32"), top_k + len(exclude_ids) + 3)
            candidates = zip(indices[0].tolist(), scores[0].tolist())
        else:
            query_vector = self.vectorizer.transform([query])
            scores = self._cosine_similarity(query_vector, self.matrix)[0]
            ranked = np.argsort(scores)[::-1][: top_k + len(exclude_ids) + 3]
            candidates = ((int(idx), float(scores[idx])) for idx in ranked.tolist())

        results: list[RetrievalResult] = []
        for idx, score in candidates:
            if idx < 0:
                continue
            row = self.knowledge_records[idx]
            if row.id in exclude_ids:
                continue
            results.append(RetrievalResult(record_id=row.id, text=row.clean_exp, score=float(score)))
            if len(results) >= top_k:
                break
        return results
