from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import time

import requests


ANSWER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
LABEL_RE = re.compile(r"(FACTUAL_ERROR|REASONING_FAILURE)")
VALID_LABELS = {"FACTUAL_ERROR", "REASONING_FAILURE"}
RETRY_AFTER_RE = re.compile(r"retry in ([0-9.]+)s", re.IGNORECASE)


@dataclass(frozen=True)
class VerificationResult:
    label: str
    reason: str


def normalize_answer(text: str | None) -> str:
    if not text:
        return ""
    match = ANSWER_RE.search(text.upper())
    return match.group(1) if match else ""


def answer_prompt(question: str, options: dict[str, str], context: str = "") -> str:
    context_block = f"Medical Context:\n{context}\n\n" if context else ""
    return (
        "You are a medical expert solving a multiple-choice medical question.\n"
        "Return ONLY one capital letter: A, B, C, or D.\n\n"
        f"{context_block}"
        f"Question: {question}\n"
        f"A) {options['A']}\n"
        f"B) {options['B']}\n"
        f"C) {options['C']}\n"
        f"D) {options['D']}\n"
        "Return only the single best option letter."
    )


def verifier_prompt(
    *,
    question: str,
    options: dict[str, str],
    correct_letter: str,
    correct_text: str,
    model_answer_letter: str,
    model_answer_text: str,
    gold_explanation: str = "",
    retrieved_context: str = "",
) -> str:
    return (
        "You are a strict evaluator for medical multiple-choice QA errors.\n"
        "The answer is already known to be incorrect.\n"
        "Use only these labels:\n"
        "- FACTUAL_ERROR: the chosen option is medically or factually wrong.\n"
        "- REASONING_FAILURE: the chosen option is semantically close to the correct answer, "
        "but the model made an incorrect inference, distinction, or comparison.\n"
        f"Question: {question}\n"
        f"Options: A) {options['A']} | B) {options['B']} | C) {options['C']} | D) {options['D']}\n"
        f"Correct option: {correct_letter}\n"
        f"Correct answer text: {correct_text}\n"
        f"Model answer letter: {model_answer_letter}\n"
        f"Model answer text: {model_answer_text}\n"
        f"Gold explanation: {gold_explanation or 'NONE'}\n"
        f"Retrieved context: {retrieved_context or 'NONE'}\n"
        "Return strict JSON with keys label and reason. "
        "label must be exactly one of FACTUAL_ERROR or REASONING_FAILURE."
    )


def _call_openai(prompt: str, model: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def _call_gemini(prompt: str, model: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            response = genai.GenerativeModel(model).generate_content(
                prompt,
                generation_config={"temperature": 0.0},
            )
            return getattr(response, "text", "") or ""
        except Exception as exc:  # pragma: no cover - depends on remote API
            last_error = exc
            error_text = str(exc)
            if "RESOURCE_EXHAUSTED" not in error_text and "429" not in error_text:
                raise
            match = RETRY_AFTER_RE.search(error_text)
            wait_seconds = float(match.group(1)) + 2.0 if match else min(60.0, 10.0 * (attempt + 1))
            time.sleep(wait_seconds)
    assert last_error is not None
    raise last_error


def _call_ollama(prompt: str, model: str) -> str:
    base_url = (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
    response = requests.post(
        f"{base_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.0}},
        timeout=180,
    )
    response.raise_for_status()
    return response.json().get("response", "")


def call_model(provider: str, model: str, prompt: str) -> str:
    normalized = provider.strip().lower()
    if normalized == "openai":
        return _call_openai(prompt, model)
    if normalized == "gemini":
        return _call_gemini(prompt, model)
    if normalized in {"ollama", "llama", "llama3"}:
        return _call_ollama(prompt, model)
    if normalized == "mock":
        return "A"
    raise ValueError(f"Unsupported provider: {provider}")


def classify_with_llm(provider: str, model: str, prompt: str) -> VerificationResult:
    raw = call_model(provider, model, prompt)
    try:
        parsed = json.loads(raw)
        label = str(parsed.get("label", "")).strip().upper()
        reason = str(parsed.get("reason", "")).strip()
        if label in VALID_LABELS:
            return VerificationResult(label=label, reason=reason or "No explanation returned.")
    except Exception:
        pass

    match = LABEL_RE.search(raw)
    if match:
        label = match.group(1)
        reason = raw.replace(label, "").strip(" :-\n") or "Verifier returned an unstructured response."
        return VerificationResult(label=label, reason=reason)

    return VerificationResult(label="FACTUAL_ERROR", reason="Verifier returned an invalid response.")
