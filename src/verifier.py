from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re

import requests


VALID_LABELS = {"FACTUAL_ERROR", "REASONING_FAILURE"}
LABEL_RE = re.compile(r"(FACTUAL_ERROR|REASONING_FAILURE)")


@dataclass(frozen=True)
class VerificationResult:
    label: str
    reason: str


def build_verifier_prompt(
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
        "The answer is already known to be incorrect. "
        "Classify the kind of hallucination.\n"
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


def _parse_llm_verifier_output(text: str) -> VerificationResult:
    try:
        parsed = json.loads(text)
        label = str(parsed.get("label", "")).strip().upper()
        reason = str(parsed.get("reason", "")).strip()
        if label in VALID_LABELS:
            return VerificationResult(label=label, reason=reason or "No explanation returned by verifier.")
    except Exception:
        pass

    match = LABEL_RE.search(text or "")
    if match:
        label = match.group(1)
        reason = (text or "").replace(label, "").strip(" :-\n") or "Verifier returned an unstructured response."
        return VerificationResult(label=label, reason=reason)

    return VerificationResult(label="FACTUAL_ERROR", reason="Verifier returned an invalid response.")


def _classify_with_openai(prompt: str, model_name: str) -> VerificationResult:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Output valid JSON only."},
            {"role": "user", "content": prompt},
        ],
    )
    return _parse_llm_verifier_output(response.choices[0].message.content or "")


def _classify_with_gemini(prompt: str, model_name: str) -> VerificationResult:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt, generation_config={"temperature": 0.0})
    return _parse_llm_verifier_output(getattr(response, "text", "") or "")


def _classify_with_ollama(prompt: str, model_name: str) -> VerificationResult:
    base_url = (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
    response = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},
        },
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    return _parse_llm_verifier_output(payload.get("response", ""))


def _heuristic_verify(
    *,
    correct_text: str,
    model_answer_text: str,
    retrieved_context: str = "",
) -> VerificationResult:
    if correct_text and model_answer_text and correct_text.split(" ")[0].lower() in model_answer_text.lower():
        return VerificationResult(
            label="REASONING_FAILURE",
            reason="The prediction is related to the expected content, but the option choice is still wrong.",
        )
    return VerificationResult(
        label="FACTUAL_ERROR",
        reason="The selected answer is medically incorrect relative to the gold answer.",
    )


def classify_error(
    *,
    question: str,
    options: dict[str, str],
    correct_letter: str,
    correct_text: str,
    model_answer: str,
    model_answer_text: str = "",
    gold_explanation: str = "",
    retrieved_context: str = "",
    provider: str = "llama",
    model_name: str = "llama3:latest",
) -> VerificationResult:
    prompt = build_verifier_prompt(
        question=question,
        options=options,
        correct_letter=correct_letter,
        correct_text=correct_text,
        model_answer_letter=model_answer,
        model_answer_text=model_answer_text,
        gold_explanation=gold_explanation,
        retrieved_context=retrieved_context,
    )

    try:
        normalized = provider.strip().lower()
        if normalized == "openai":
            return _classify_with_openai(prompt, model_name)
        if normalized == "gemini":
            return _classify_with_gemini(prompt, model_name)
        if normalized in {"llama", "ollama", "llama3"}:
            return _classify_with_ollama(prompt, model_name)
        raise ValueError(f"Unsupported verifier provider: {provider}")
    except Exception:
        return _heuristic_verify(
            correct_text=correct_text,
            model_answer_text=model_answer_text,
            retrieved_context=retrieved_context,
        )
