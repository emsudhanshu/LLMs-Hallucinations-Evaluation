from __future__ import annotations

import json
from dataclasses import dataclass

from .models import OpenAIAnswerModel


VERIFIER_LABELS = {
    "FACTUAL_ERROR",
    "REASONING_FAILURE",
}


@dataclass(frozen=True)
class VerificationResult:
    label: str
    explanation: str


def build_verifier_prompt(
    *,
    question: str,
    options: dict[str, str],
    correct_answer: str,
    model_answer_letter: str,
    model_answer_text: str,
    retrieved_context: str | None = None,
) -> str:
    context_text = retrieved_context if retrieved_context else "NONE"
    return (
        "A medical QA model answered a multiple-choice question incorrectly.\n"
        "Use only these labels:\n"
        "- FACTUAL_ERROR: the chosen option is medically or factually wrong.\n"
        "- REASONING_FAILURE: the chosen option is semantically close to the correct answer, "
        "but the model made an incorrect inference, distinction, or comparison.\n"
        f"Question: {question}\n"
        f"Options: A) {options['A']} | B) {options['B']} | C) {options['C']} | D) {options['D']}\n"
        f"Correct answer: {correct_answer}\n"
        f"Model answer letter: {model_answer_letter}\n"
        f"Model answer text: {model_answer_text}\n"
        f"Retrieved context: {context_text}\n"
        "Classify the error into exactly one label: FACTUAL_ERROR or REASONING_FAILURE.\n"
        "Return strict JSON with keys label and explanation."
    )


def classify_incorrect_answer(
    *,
    question: str,
    options: dict[str, str],
    correct_answer: str,
    model_answer_letter: str,
    model_answer_text: str,
    retrieved_context: str | None = None,
) -> VerificationResult:
    model = OpenAIAnswerModel()
    raw_response = model.client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict evaluator for medical multiple-choice QA errors. "
                    "Only output valid JSON."
                ),
            },
            {
                "role": "user",
                "content": build_verifier_prompt(
                    question=question,
                    options=options,
                    correct_answer=correct_answer,
                    model_answer_letter=model_answer_letter,
                    model_answer_text=model_answer_text,
                    retrieved_context=retrieved_context,
                ),
            },
        ],
    )
    content = raw_response.choices[0].message.content or "{}"
    data = json.loads(content)
    label = str(data.get("label", "")).strip().upper()
    explanation = str(data.get("explanation", "")).strip()
    if label not in VERIFIER_LABELS:
        label = "FACTUAL_ERROR"
    return VerificationResult(label=label, explanation=explanation)
