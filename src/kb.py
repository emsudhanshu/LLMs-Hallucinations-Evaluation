from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from data_loader import load_split


class HybridRetriever:
    """FAISS dense retrieval + BM25 sparse retrieval fused with Reciprocal Rank Fusion.

    Scores returned by ``similarity_search_with_score`` are cosine similarities
    in the range ``[0, 1]`` (higher means more relevant), derived from the L2
    distances produced by the FAISS index.  Using a cosine scale makes the
    ``min_score_threshold`` config value easy to reason about regardless of
    whether BM25 is active.

    BM25 is used opportunistically: if ``rank_bm25`` is not installed the class
    falls back to dense-only retrieval without error.
    """

    _RRF_K: int = 60  # standard RRF smoothing constant

    def __init__(self, vector_store: FAISS, documents: list[Document]) -> None:
        self._vector_store = vector_store
        self._documents = documents
        self._bm25: object | None = None
        if documents:
            try:
                from rank_bm25 import BM25Okapi  # type: ignore[import]

                tokenized = [doc.page_content.lower().split() for doc in documents]
                self._bm25 = BM25Okapi(tokenized)
            except ImportError:
                pass  # BM25 unavailable; falls back to dense-only

    @staticmethod
    def _l2_to_cosine(l2: float) -> float:
        """Convert an L2 distance to cosine similarity for unit-norm embeddings."""
        return max(0.0, 1.0 - (l2 ** 2) / 2.0)

    def similarity_search_with_score(
        self, query: str, k: int = 5
    ) -> list[tuple[Document, float]]:
        """Return up to *k* ``(Document, cosine_similarity)`` pairs.

        Results are ranked by RRF when BM25 is available, otherwise by cosine
        similarity from FAISS alone.  The returned score is always a cosine
        similarity so callers can apply a consistent relevance threshold.
        """
        # Fetch more candidates than needed so RRF has a wide pool to rerank.
        fetch_k = k * 3 + 10

        # --- Dense retrieval (FAISS; returns L2 distances) ---
        dense_results = self._vector_store.similarity_search_with_score(query, k=fetch_k)
        dense_rank: dict[str, int] = {}
        cosine_map: dict[str, float] = {}
        doc_map: dict[str, Document] = {}
        for rank, (doc, l2_dist) in enumerate(dense_results):
            key = str(doc.metadata.get("id") or doc.page_content[:80])
            if key not in dense_rank:
                dense_rank[key] = rank + 1
                cosine_map[key] = self._l2_to_cosine(float(l2_dist))
                doc_map[key] = doc

        # --- Sparse retrieval (BM25) ---
        bm25_rank: dict[str, int] = {}
        if self._bm25 is not None and self._documents:
            bm25_scores = self._bm25.get_scores(query.lower().split())
            bm25_fetch = min(fetch_k, len(self._documents))
            bm25_order = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True,
            )[:bm25_fetch]
            # For documents found only by BM25 (not in dense results), use the
            # minimum cosine score from the dense batch as a floor.  This avoids
            # silently dropping BM25-found documents when a min_score_threshold
            # is set, because exact keyword matches are inherently relevant.
            min_dense_cosine = min(cosine_map.values()) if cosine_map else 0.0
            for rank, idx in enumerate(bm25_order):
                doc = self._documents[idx]
                key = str(doc.metadata.get("id") or doc.page_content[:80])
                if key not in bm25_rank:
                    bm25_rank[key] = rank + 1
                if key not in doc_map:
                    doc_map[key] = doc
                    cosine_map[key] = min_dense_cosine  # floor for BM25-only hits

        # --- RRF fusion ---
        all_keys = set(dense_rank) | set(bm25_rank)
        rrf_scores: dict[str, float] = {}
        for key in all_keys:
            dr = dense_rank.get(key, fetch_k + 1)
            br = bm25_rank.get(key, fetch_k + 1)
            rrf_scores[key] = 1.0 / (self._RRF_K + dr) + 1.0 / (self._RRF_K + br)

        top_keys = sorted(rrf_scores, key=lambda c: rrf_scores[c], reverse=True)[:k]
        return [(doc_map[key], cosine_map[key]) for key in top_keys if key in doc_map]


def _extract_documents_from_store(vector_store: FAISS) -> list[Document]:
    """Return the documents stored in *vector_store* in index order."""
    docs: list[Document] = []
    for idx in sorted(vector_store.index_to_docstore_id.keys()):
        doc_id = vector_store.index_to_docstore_id[idx]
        doc = vector_store.docstore._dict.get(doc_id)
        if doc is not None:
            docs.append(doc)
    return docs


def build_knowledge_base(
    *,
    data_dir: str | Path,
    split: str,
    output_path: str | Path,
    limit: int | None = None,
) -> Path:
    records = [row for row in load_split(data_dir, split, require_labels=False) if row.clean_exp]
    if limit:
        records = records[:limit]

    knowledge = [
        {
            "id": row.id,
            "question": row.question,
            "text": row.clean_exp,
            "subject": row.subject,
            "topic": row.topic,
        }
        for row in records
    ]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(knowledge, indent=2), encoding="utf-8")
    return path


def load_knowledge_base(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_or_create_vector_store(
    *,
    knowledge_base_path: str | Path,
    faiss_dir: str | Path,
    embedding_model: str,
    rebuild: bool = False,
    use_hybrid: bool = False,
) -> HybridRetriever:
    """Build or load the FAISS vector store and wrap it in a :class:`HybridRetriever`.

    The document format indexes only the *explanation*, *subject*, and *topic*
    fields.  Excluding the source question prevents the retriever from
    accidentally matching on question phrasing rather than on medical content.

    Pass ``use_hybrid=True`` (or set ``retrieval.use_hybrid_search`` in the
    config) to activate BM25 + dense RRF fusion.  When ``rank_bm25`` is not
    installed, the retriever silently falls back to dense-only search.

    .. note::
        Changing ``embedding_model`` or the document format requires rebuilding
        the FAISS index.  Pass ``rebuild=True`` or use ``--rebuild-kb``.
    """
    faiss_path = Path(faiss_dir)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    if faiss_path.exists() and not rebuild:
        vector_store = FAISS.load_local(
            str(faiss_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        documents = _extract_documents_from_store(vector_store) if use_hybrid else []
        return HybridRetriever(vector_store, documents)

    knowledge = load_knowledge_base(knowledge_base_path)
    # Index explanation + subject + topic only.
    # The question text is preserved in metadata for ID filtering but excluded
    # from page_content so retrieval is based on medical knowledge, not question
    # phrasing.
    documents = [
        Document(
            page_content=(
                f"Explanation: {item['text']}\n"
                f"Subject: {item['subject']}\n"
                f"Topic: {item['topic']}"
            ),
            metadata={
                "id": item["id"],
                "question": item["question"],
                "subject": item["subject"],
                "topic": item["topic"],
            },
        )
        for item in knowledge
    ]

    vector_store = FAISS.from_documents(documents, embeddings)
    faiss_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(faiss_path))
    return HybridRetriever(vector_store, documents if use_hybrid else [])

