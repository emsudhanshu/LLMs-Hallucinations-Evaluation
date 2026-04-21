from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .config import get_ollama_base_url


VERIFIER_LABELS = {
    "FACTUAL_ERROR",
    "REASONING_FAILURE",
}

LABEL_RE = re.compile(r"\b(FACTUAL_ERROR|REASONING_FAILURE)\b")


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


def _call_verifier_openai(prompt: str, model: str) -> str:
    from .models import OpenAIAnswerModel

    verifier = OpenAIAnswerModel()
    raw_response = verifier.client.chat.completions.create(
        model=model,
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
            {"role": "user", "content": prompt},
        ],
    )
    return raw_response.choices[0].message.content or "{}"


def _call_verifier_gemini(prompt: str, model: str) -> str:
    import google.generativeai as genai

    from .config import get_gemini_api_key

    api_key = get_gemini_api_key()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    genai.configure(api_key=api_key)
    response = genai.GenerativeModel(model).generate_content(
        prompt,
        generation_config={"temperature": 0.0},
    )
    return getattr(response, "text", "") or "{}"


def _call_verifier_ollama(prompt: str, model: str) -> str:
    import requests

    base_url = get_ollama_base_url().rstrip("/")
    response = requests.post(
        f"{base_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.0}},
        timeout=180,
    )
    response.raise_for_status()
    return response.json().get("response", "{}")


def _parse_verifier_response(raw: str) -> VerificationResult:
    try:
        data = json.loads(raw)
        label = str(data.get("label", "")).strip().upper()
        explanation = str(data.get("explanation", "")).strip()
        if label in VERIFIER_LABELS:
            return VerificationResult(label=label, explanation=explanation or "No explanation returned.")
    except Exception:
        pass

    match = LABEL_RE.search(raw)
    if match:
        label = match.group(1)
        explanation = raw.replace(label, "").strip(" :-\n") or "Verifier returned an unstructured response."
        return VerificationResult(label=label, explanation=explanation)

    return VerificationResult(label="FACTUAL_ERROR", explanation="Verifier returned an invalid response.")


def classify_incorrect_answer(
    *,
    question: str,
    options: dict[str, str],
    correct_answer: str,
    model_answer_letter: str,
    model_answer_text: str,
    retrieved_context: str | None = None,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
) -> VerificationResult:
    prompt = build_verifier_prompt(
        question=question,
        options=options,
        correct_answer=correct_answer,
        model_answer_letter=model_answer_letter,
        model_answer_text=model_answer_text,
        retrieved_context=retrieved_context,
    )
    normalized = provider.strip().lower()
    if normalized == "openai":
        raw = _call_verifier_openai(prompt, model)
    elif normalized == "gemini":
        raw = _call_verifier_gemini(prompt, model)
    elif normalized in {"ollama", "llama", "llama3"}:
        raw = _call_verifier_ollama(prompt, model)
    else:
        raise ValueError(f"Unsupported verifier provider: {provider!r}")
    return _parse_verifier_response(raw)
