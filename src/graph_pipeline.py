from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from llm import answer_prompt, call_model, classify_with_llm, normalize_answer, verifier_prompt

_WORDS_PER_TOKEN_APPROX = 0.75  # rough conversion used for context truncation

# Tuple collected for each accepted retrieved chunk: (document, cosine_score, text)
_KeptChunk = tuple[Document, float, str]


def _truncate_words(text: str, max_words: int) -> str:
    """Return *text* truncated to at most *max_words* whitespace-separated tokens."""
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


class QAState(TypedDict, total=False):
    question: str
    options: dict[str, str]
    correct_letter: str
    correct_text: str
    clean_exp: str
    record_id: str
    answer_provider: str
    answer_model: str
    verifier_provider: str
    verifier_model: str
    retriever: object
    top_k: int
    fetch_k: int
    min_score_threshold: float
    max_chunk_words: int
    mode: str
    skip_verifier: bool
    retrieved_context: str
    retrieved_ids: str
    retrieved_scores: str
    predicted_letter: str
    predicted_text: str
    is_correct: bool
    hallucination_label: str
    hallucination_reason: str


def build_graph():
    graph = StateGraph(QAState)

    def retrieve(state: QAState) -> QAState:
        retriever = state.get("retriever")
        if state.get("mode") != "rag" or retriever is None:
            return {"retrieved_context": "", "retrieved_ids": "", "retrieved_scores": ""}

        top_k: int = state.get("top_k", 3)
        fetch_k: int = state.get("fetch_k", top_k * 3 + 5)
        min_score: float = state.get("min_score_threshold", 0.0)
        max_chunk_words: int = state.get("max_chunk_words", 0)
        # Cap total context at 3× the per-chunk budget to avoid overwhelming
        # smaller models with irrelevant text.
        max_total_words: int = max_chunk_words * top_k if max_chunk_words else 0

        # Enrich the retrieval query with the answer options so that medical
        # terminology in the choices helps surface more relevant KB entries.
        options = state.get("options", {})
        enriched_query = (
            state["question"]
            + " "
            + " ".join(options.values())
        ).strip()

        docs = retriever.similarity_search_with_score(enriched_query, k=fetch_k)

        kept: list[_KeptChunk] = []
        total_words = 0
        for doc, score in docs:
            if doc.metadata.get("id") == state.get("record_id"):
                continue
            # Drop chunks whose cosine similarity falls below the threshold.
            if score < min_score:
                continue
            content = doc.page_content
            if max_chunk_words:
                content = _truncate_words(content, max_chunk_words)
            chunk_words = len(content.split())
            # Stop adding chunks once the total context budget is exhausted.
            if max_total_words and total_words + chunk_words > max_total_words:
                break
            kept.append((doc, score, content))
            total_words += chunk_words
            if len(kept) >= top_k:
                break

        return {
            "retrieved_context": "\n\n".join(content for _, _, content in kept),
            "retrieved_ids": "|".join(str(doc.metadata.get("id", "")) for doc, _, _ in kept),
            "retrieved_scores": "|".join(f"{float(score):.4f}" for _, score, _ in kept),
        }

    def answer(state: QAState) -> QAState:
        prompt = answer_prompt(
            state["question"],
            state["options"],
            context=state.get("retrieved_context", ""),
        )
        raw = call_model(state["answer_provider"], state["answer_model"], prompt)
        predicted = normalize_answer(raw)
        predicted_text = state["options"].get(predicted, "") if predicted else ""
        return {
            "predicted_letter": predicted if predicted else f"ERROR:INVALID_RESPONSE",
            "predicted_text": predicted_text,
            "is_correct": predicted == state["correct_letter"],
        }

    def verify(state: QAState) -> QAState:
        if state["is_correct"] or state.get("skip_verifier", False):
            return {"hallucination_label": "", "hallucination_reason": ""}

        prompt = verifier_prompt(
            question=state["question"],
            options=state["options"],
            correct_letter=state["correct_letter"],
            correct_text=state["correct_text"],
            model_answer_letter=state["predicted_letter"],
            model_answer_text=state.get("predicted_text", ""),
            gold_explanation=state.get("clean_exp", ""),
            retrieved_context=state.get("retrieved_context", ""),
        )
        result = classify_with_llm(
            state["verifier_provider"],
            state["verifier_model"],
            prompt,
        )
        return {"hallucination_label": result.label, "hallucination_reason": result.reason}

    def route_after_answer(state: QAState) -> str:
        return "end" if state["is_correct"] or state.get("skip_verifier", False) else "verify"

    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)
    graph.add_node("verify", verify)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_conditional_edges(
        "answer",
        route_after_answer,
        {"verify": "verify", "end": END},
    )
    graph.add_edge("verify", END)
    return graph.compile()
