from __future__ import annotations

from abc import ABC, abstractmethod
import os
import re

import requests

from .config import get_gemini_api_key, get_ollama_base_url, get_openai_api_key


ANSWER_PATTERN = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def normalize_answer(text: str | None) -> str:
    if not text:
        return ""
    match = ANSWER_PATTERN.search(text.strip().upper())
    return match.group(1).upper() if match else ""


class BaseAnswerModel(ABC):
    @abstractmethod
    def answer(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        raise NotImplementedError


class MockModel(BaseAnswerModel):
    def answer(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        del system_prompt, temperature
        for letter in ("A", "B", "C", "D"):
            if f"{letter})" in user_prompt:
                return letter
        return "A"


class OpenAIAnswerModel(BaseAnswerModel):
    def __init__(self) -> None:
        api_key = get_openai_api_key()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)

    def answer(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return normalize_answer(response.choices[0].message.content)


class GeminiAnswerModel(BaseAnswerModel):
    def __init__(self) -> None:
        api_key = get_gemini_api_key()
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def answer(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        response = self.model.generate_content(
            f"{system_prompt}\n\n{user_prompt}",
            generation_config={"temperature": temperature},
        )
        return normalize_answer(response.text)


class OllamaLlamaAnswerModel(BaseAnswerModel):
    def __init__(self) -> None:
        self.base_url = get_ollama_base_url().rstrip("/")

    def answer(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        payload = {
            "model": os.getenv("OLLAMA_MODEL", "llama3:8b"),
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "stream": False,
            "options": {"temperature": temperature},
        }
        response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return normalize_answer(data.get("response", ""))


def build_model(name: str) -> BaseAnswerModel:
    normalized = name.strip().lower()
    if normalized == "mock":
        return MockModel()
    if normalized == "openai":
        return OpenAIAnswerModel()
    if normalized == "gemini":
        return GeminiAnswerModel()
    if normalized in {"llama", "ollama", "llama3"}:
        return OllamaLlamaAnswerModel()
    raise ValueError(f"Unsupported model name: {name}")

