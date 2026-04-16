from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import time

import pandas as pd

from .config import RESULTS_DIR, ExperimentConfig
from .dataset import MCQSample, sample_mcq_questions
from .models import build_model
from .prompts import SYSTEM_PROMPT, build_no_rag_prompt, build_rag_prompt
from .retriever import LocalMedicalRetriever


@dataclass
class ExperimentRow:
    question_id: str
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    correct_answer: str
    model_name: str
    prompt_mode: str
    model_answer: str
    is_correct: bool
    subject_name: str | None
    topic_name: str | None
    choice_type: str | None
    retrieved_chunk_1: str
    retrieved_chunk_2: str
    retrieved_chunk_3: str
    verifier_label: str
    verifier_explanation: str
    timestamp_utc: str


def _sample_to_row(
    sample: MCQSample,
    *,
    model_name: str,
    prompt_mode: str,
    model_answer: str,
    retrieved_chunks: list[str] | None = None,
) -> ExperimentRow:
    retrieved_chunks = (retrieved_chunks or [])[:3]
    padded_chunks = retrieved_chunks + [""] * (3 - len(retrieved_chunks))
    return ExperimentRow(
        question_id=sample.question_id,
        question=sample.question,
        option_a=sample.options["A"],
        option_b=sample.options["B"],
        option_c=sample.options["C"],
        option_d=sample.options["D"],
        correct_answer=sample.correct_answer,
        model_name=model_name,
        prompt_mode=prompt_mode,
        model_answer=model_answer,
        is_correct=model_answer == sample.correct_answer,
        subject_name=sample.subject_name,
        topic_name=sample.topic_name,
        choice_type=sample.choice_type,
        retrieved_chunk_1=padded_chunks[0],
        retrieved_chunk_2=padded_chunks[1],
        retrieved_chunk_3=padded_chunks[2],
        verifier_label="",
        verifier_explanation="",
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


def run_no_rag_experiment(config: ExperimentConfig) -> tuple[pd.DataFrame, Path]:
    samples = sample_mcq_questions(
        config.split,
        sample_size=config.sample_size,
        seed=config.seed,
        require_labels=True,
    )
    model = build_model(config.model_name)

    rows: list[ExperimentRow] = []
    for sample in samples:
        prompt = build_no_rag_prompt(sample)
        try:
            model_answer = model.answer(SYSTEM_PROMPT, prompt, temperature=config.temperature)
        except Exception as exc:
            model_answer = f"ERROR:{type(exc).__name__}"

        rows.append(
            _sample_to_row(
                sample,
                model_name=config.model_name,
                prompt_mode=config.prompt_mode,
                model_answer=model_answer,
            )
        )
        time.sleep(config.sleep_seconds)

    df = pd.DataFrame(asdict(row) for row in rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / (
        f"results_{config.prompt_mode}_{config.model_name}_{config.split}_{len(df)}.csv"
    )
    df.to_csv(output_path, index=False)
    return df, output_path


def run_rag_experiment(config: ExperimentConfig, *, top_k: int = 3) -> tuple[pd.DataFrame, Path]:
    samples = sample_mcq_questions(
        config.split,
        sample_size=config.sample_size,
        seed=config.seed,
        require_labels=True,
    )
    model = build_model(config.model_name)
    retriever = LocalMedicalRetriever(corpus_split="train")

    rows: list[ExperimentRow] = []
    for sample in samples:
        retrieved = retriever.retrieve(sample.question, k=top_k)
        retrieved_texts = [item.text for item in retrieved]
        prompt = build_rag_prompt(sample, retrieved_texts)
        try:
            model_answer = model.answer(SYSTEM_PROMPT, prompt, temperature=config.temperature)
        except Exception as exc:
            model_answer = f"ERROR:{type(exc).__name__}"

        rows.append(
            _sample_to_row(
                sample,
                model_name=config.model_name,
                prompt_mode="rag",
                model_answer=model_answer,
                retrieved_chunks=retrieved_texts,
            )
        )
        time.sleep(config.sleep_seconds)

    df = pd.DataFrame(asdict(row) for row in rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"results_rag_{config.model_name}_{config.split}_{len(df)}.csv"
    df.to_csv(output_path, index=False)
    return df, output_path
