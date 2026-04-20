from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from data_loader import load_split


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
) -> FAISS:
    faiss_path = Path(faiss_dir)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    if faiss_path.exists() and not rebuild:
        return FAISS.load_local(
            str(faiss_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    knowledge = load_knowledge_base(knowledge_base_path)
    documents = [
        Document(
            page_content=item["text"],
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
    return vector_store

