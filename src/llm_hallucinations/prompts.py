from __future__ import annotations

from .dataset import MCQSample


SYSTEM_PROMPT = (
    "You are a medical expert. Answer the following multiple choice medical "
    "question. Return ONLY one capital letter: A, B, C, or D."
)


def build_no_rag_prompt(sample: MCQSample) -> str:
    return (
        f"Question: {sample.question}\n"
        f"A) {sample.options['A']}\n"
        f"B) {sample.options['B']}\n"
        f"C) {sample.options['C']}\n"
        f"D) {sample.options['D']}\n"
        "Return only the correct option letter."
    )


def build_rag_prompt(sample: MCQSample, retrieved_chunks: list[str]) -> str:
    context = "\n\n".join(
        f"Context {idx + 1}: {chunk}" for idx, chunk in enumerate(retrieved_chunks)
    )
    return (
        "Use the following medical context if it is relevant to the question. "
        "If it does not apply, rely on your medical knowledge.\n"
        f"{context}\n\n"
        f"Question: {sample.question}\n"
        f"A) {sample.options['A']}\n"
        f"B) {sample.options['B']}\n"
        f"C) {sample.options['C']}\n"
        f"D) {sample.options['D']}\n"
        "Return only the correct option letter."
    )
