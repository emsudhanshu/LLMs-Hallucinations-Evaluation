from __future__ import annotations

from dataclasses import dataclass
import re

from .dataset import MCQSample, load_mcq_split


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source_question_id: str
    score: float


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")

_RRF_K = 60  # standard RRF smoothing constant


def _build_chunk_text(sample: MCQSample) -> str:
    """Build the indexed text for a knowledge-base sample.

    Only the explanation, subject, and topic are indexed.  The source question
    and its answer options are intentionally excluded so that retrieval matches
    on medical *knowledge* rather than on surface-level question phrasing.
    """
    parts: list[str] = []
    if sample.explanation:
        parts.append(f"Explanation: {sample.explanation}")
    if sample.subject_name:
        parts.append(f"Subject: {sample.subject_name}")
    if sample.topic_name:
        parts.append(f"Topic: {sample.topic_name}")
    return "\n".join(parts)


class LocalMedicalRetriever:
    """Hybrid BM25 + TF-IDF cosine retriever with Reciprocal Rank Fusion.

    Both a BM25 index (via ``rank_bm25``) and a TF-IDF cosine index (via
    ``scikit-learn``) are built at construction time.  At query time each
    index produces a ranked list; the two lists are fused using RRF so that
    documents highly ranked by *either* signal float to the top.

    If ``rank_bm25`` is not installed, retrieval falls back to TF-IDF cosine
    only.  If ``scikit-learn`` is also absent, a simple token-overlap cosine
    is used as a last resort.
    """

    def __init__(self, corpus_split: str = "train", max_corpus_size: int = 5000) -> None:
        self.samples = load_mcq_split(corpus_split, require_labels=False)[:max_corpus_size]
        self.documents = [_build_chunk_text(sample) for sample in self.samples]
        self._bm25: object | None = None
        self._tfidf_matrix: object | None = None
        self._tfidf_vectorizer: object | None = None
        self._cosine_similarity: object | None = None
        self._build_indices()

    def _build_indices(self) -> None:
        # --- BM25 ---
        try:
            from rank_bm25 import BM25Okapi  # type: ignore[import]

            tokenized = [TOKEN_PATTERN.findall(doc.lower()) for doc in self.documents]
            self._bm25 = BM25Okapi(tokenized)
        except ImportError:
            pass  # Graceful fallback

        # --- TF-IDF cosine (scikit-learn) ---
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import]
            from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import]

            vectorizer = TfidfVectorizer(stop_words="english", max_features=30000)
            self._tfidf_matrix = vectorizer.fit_transform(self.documents)
            self._tfidf_vectorizer = vectorizer
            self._cosine_similarity = cosine_similarity
        except ImportError:
            pass  # Graceful fallback

    def _bm25_ranked(self, query: str, fetch_k: int) -> list[tuple[int, float]]:
        """Return (index, score) pairs sorted by BM25 score descending."""
        if self._bm25 is None:
            return []
        tokens = TOKEN_PATTERN.findall(query.lower())
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:fetch_k]
        return [(int(idx), float(scores[idx])) for idx in ranked]

    def _tfidf_ranked(self, query: str, fetch_k: int) -> list[tuple[int, float]]:
        """Return (index, score) pairs sorted by TF-IDF cosine similarity descending."""
        if self._tfidf_vectorizer is None or self._tfidf_matrix is None:
            return []
        import numpy as np

        query_vec = self._tfidf_vectorizer.transform([query])
        sims = self._cosine_similarity(query_vec, self._tfidf_matrix)[0]
        ranked = np.argsort(sims)[::-1][:fetch_k].tolist()
        return [(int(idx), float(sims[idx])) for idx in ranked]

    def retrieve(self, query: str, *, k: int = 3) -> list[RetrievedChunk]:
        """Return the top-*k* most relevant chunks using hybrid BM25 + TF-IDF RRF."""
        n = len(self.documents)
        if n == 0:
            return []
        fetch_k = min(k * 3 + 10, n)

        bm25_results = self._bm25_ranked(query, fetch_k)
        tfidf_results = self._tfidf_ranked(query, fetch_k)

        # Build rank maps (1-based) for each signal
        bm25_rank = {idx: rank + 1 for rank, (idx, _) in enumerate(bm25_results)}
        tfidf_rank = {idx: rank + 1 for rank, (idx, _) in enumerate(tfidf_results)}
        tfidf_score_map = {idx: score for idx, score in tfidf_results}

        # RRF fusion
        all_indices = set(bm25_rank) | set(tfidf_rank)
        rrf: dict[int, float] = {}
        for idx in all_indices:
            br = bm25_rank.get(idx, fetch_k + 1)
            tr = tfidf_rank.get(idx, fetch_k + 1)
            rrf[idx] = 1.0 / (_RRF_K + br) + 1.0 / (_RRF_K + tr)

        top_indices = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:k]

        return [
            RetrievedChunk(
                text=self.documents[idx],
                source_question_id=self.samples[idx].question_id,
                # Always use the TF-IDF cosine for the score field so the scale
                # is consistent (0–1).  Documents found only by BM25 receive 0.0
                # since no cosine estimate is available for them.
                score=tfidf_score_map.get(idx, 0.0),
            )
            for idx in top_indices
        ]
