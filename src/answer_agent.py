from __future__ import annotations

from abc import ABC, abstractmethod
import os
import re

import requests


ANSWER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def normalize_model_answer(text: str | None) -> str:
    if not text:
        return ""
    match = ANSWER_RE.search(text.strip().upper())
    return match.group(1).upper() if match else ""


def build_answer_prompt(question: str, options: dict[str, str], context: str = "") -> tuple[str, str]:
    system = (
        "You are a medical expert solving a multiple-choice medical question. "
        "Return ONLY one capital letter: A, B, C, or D. Do not explain your answer."
    )
    context_block = f"Medical Context:\n{context}\n\n" if context else ""
    user = (
        f"{context_block}"
        f"Question: {question}\n"
        f"A) {options['A']}\n"
        f"B) {options['B']}\n"
        f"C) {options['C']}\n"
        f"D) {options['D']}\n"
        "Return only the single best option letter."
    )
    return system, user


class BaseAnswerAgent(ABC):
    @abstractmethod
    def answer(self, question: str, options: dict[str, str], context: str = "") -> str:
        raise NotImplementedError


class MockAnswerAgent(BaseAnswerAgent):
    def answer(self, question: str, options: dict[str, str], context: str = "") -> str:
        del question, context
        for letter in ("A", "B", "C", "D"):
            if options.get(letter):
                return letter
        return "A"


class OpenAIAnswerAgent(BaseAnswerAgent):
    def __init__(self, model_name: str = "gpt-4o-mini") -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def answer(self, question: str, options: dict[str, str], context: str = "") -> str:
        system, user = build_answer_prompt(question, options, context=context)
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return normalize_model_answer(response.choices[0].message.content)


class GeminiAnswerAgent(BaseAnswerAgent):
    def __init__(self, model_name: str = "gemini-2.5-flash") -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def answer(self, question: str, options: dict[str, str], context: str = "") -> str:
        system, user = build_answer_prompt(question, options, context=context)
        response = self.model.generate_content(
            f"{system}\n\n{user}",
            generation_config={"temperature": 0.0},
        )
        return normalize_model_answer(getattr(response, "text", ""))


class OllamaAnswerAgent(BaseAnswerAgent):
    def __init__(self, model_name: str = "llama3:latest", base_url: str | None = None) -> None:
        self.model_name = model_name
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")

    def answer(self, question: str, options: dict[str, str], context: str = "") -> str:
        system, user = build_answer_prompt(question, options, context=context)
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": f"{system}\n\n{user}",
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        return normalize_model_answer(payload.get("response", ""))


def create_answer_agent(name: str, model_name: str | None = None) -> BaseAnswerAgent:
    normalized = name.strip().lower()
    if normalized == "mock":
        return MockAnswerAgent()
    if normalized == "openai":
        return OpenAIAnswerAgent(model_name=model_name or "gpt-4o-mini")
    if normalized == "gemini":
        return GeminiAnswerAgent(model_name=model_name or "gemini-2.5-flash")
    if normalized in {"llama", "ollama", "llama3"}:
        return OllamaAnswerAgent(model_name=model_name or "llama3:latest")
    raise ValueError(f"Unsupported answer model: {name}")
