from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    def load_dotenv() -> bool:
        return False


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str
    split: str
    sample_size: int
    seed: int = 42
    temperature: float = 0.0
    sleep_seconds: float = 0.5
    prompt_mode: str = "no_rag"


def get_openai_api_key() -> str | None:
    return os.getenv("OPENAI_API_KEY")


def get_gemini_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY")


def get_ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
