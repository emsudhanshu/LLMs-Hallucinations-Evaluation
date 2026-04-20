from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    def load_dotenv() -> bool:
        return False


load_dotenv(PROJECT_ROOT / ".env")

from data_loader import load_split, sample_records
from evaluation import compute_metrics, save_results
from graph_pipeline import build_graph
from kb import build_knowledge_base, load_or_create_vector_store
from llm import classify_with_llm, verifier_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Medical QA evaluation with LangGraph + FAISS")
    parser.add_argument("--mode", choices=["no_rag", "rag"], required=True)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--knowledge-limit", type=int, default=0)
    parser.add_argument("--rebuild-kb", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--answer-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def options_dict(record) -> dict[str, str]:
    return {"A": record.A, "B": record.B, "C": record.C, "D": record.D}


def safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return normalized.strip("._-") or "unknown_model"


def model_results_dir(config: dict) -> Path:
    answer_model = str(config["answer_agent"]["model"])
    verifier_model = str(config["verifier_agent"]["model"])
    folder_name = answer_model if answer_model == verifier_model else f"{answer_model}__{verifier_model}"
    results_dir = PROJECT_ROOT / "results" / safe_name(folder_name)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def result_paths(config: dict, mode: str) -> tuple[Path, Path]:
    results_dir = model_results_dir(config)
    base_name = "rag_results.csv" if mode == "rag" else "no_rag_results.csv"
    verified_name = f"{mode}_verified_results.csv"
    return results_dir / base_name, results_dir / verified_name


def verified_columns() -> list[str]:
    return [
        "id",
        "mode",
        "answer_provider",
        "answer_model_name",
        "verifier_provider",
        "verifier_model_name",
        "predicted_letter",
        "correct_letter",
        "is_correct",
        "hallucination_label",
        "hallucination_reason",
    ]


def load_resume_rows(
    path: Path,
    *,
    answer_provider: str,
    answer_model: str,
    verifier_provider: str,
    verifier_model: str,
) -> list[dict]:
    if not path.exists():
        return []
    existing_df = pd.read_csv(path)
    if existing_df.empty:
        return []
    required = {"model_name", "model_id", "verifier_provider", "verifier_model"}
    if not required.issubset(existing_df.columns):
        return []
    matching = existing_df[
        (existing_df["model_name"].astype(str) == answer_provider)
        & (existing_df["model_id"].astype(str) == answer_model)
        & (existing_df["verifier_provider"].astype(str) == verifier_provider)
        & (existing_df["verifier_model"].astype(str) == verifier_model)
    ]
    return matching.to_dict("records")


def ensure_knowledge_assets(config: dict, *, knowledge_limit: int, rebuild_kb: bool) -> object | None:
    retrieval = config["retrieval"]
    runtime = config["runtime"]
    kb_path = PROJECT_ROOT / retrieval["knowledge_base_path"]
    if not kb_path.exists() or rebuild_kb:
        build_knowledge_base(
            data_dir=PROJECT_ROOT / runtime["data_dir"],
            split=runtime["knowledge_split"],
            output_path=kb_path,
            limit=knowledge_limit,
        )
    return load_or_create_vector_store(
        knowledge_base_path=kb_path,
        faiss_dir=PROJECT_ROOT / retrieval["faiss_dir"],
        embedding_model=retrieval["embedding_model"],
        rebuild=rebuild_kb or retrieval.get("rebuild", False),
    )


def run_verifier_only(*, config: dict, mode: str, base_output_path: Path, verified_output_path: Path) -> None:
    if not base_output_path.exists():
        raise FileNotFoundError(f"No base results found at {base_output_path}")

    df = pd.read_csv(base_output_path)
    if df.empty:
        raise ValueError("Base results file is empty.")

    for column, default in {
        "answer_provider": config["answer_agent"]["provider"],
        "answer_model_name": config["answer_agent"]["model"],
        "verifier_provider": config["verifier_agent"]["provider"],
        "verifier_model": config["verifier_agent"]["model"],
        "verifier_model_name": config["verifier_agent"]["model"],
        "predicted_text": "",
        "hallucination_label": "",
        "hallucination_reason": "",
        "mode": mode,
    }.items():
        if column not in df.columns:
            df[column] = default

    for column in ["hallucination_label", "hallucination_reason", "verifier_provider", "verifier_model", "verifier_model_name"]:
        df[column] = df[column].fillna("").astype(str)

    pending_mask = (df["is_correct"] == False) & (
        df["hallucination_label"].isna() | df["hallucination_label"].astype(str).str.strip().eq("")
    )
    pending_indices = df.index[pending_mask].tolist()

    try:
        from tqdm import tqdm

        iterator = tqdm(pending_indices, total=len(pending_indices))
    except Exception:
        iterator = pending_indices

    for idx in iterator:
        row = df.loc[idx]
        prompt = verifier_prompt(
            question=str(row["question"]),
            options={
                "A": str(row["A"]),
                "B": str(row["B"]),
                "C": str(row["C"]),
                "D": str(row["D"]),
            },
            correct_letter=str(row["correct_letter"]),
            correct_text=str(row["correct_text"]),
            model_answer_letter=str(row["predicted_letter"]),
            model_answer_text=str(row.get("predicted_text", "")),
            gold_explanation=str(row.get("clean_exp", "")),
            retrieved_context=str(row.get("retrieved_context", "")),
        )
        result = classify_with_llm(
            config["verifier_agent"]["provider"],
            config["verifier_agent"]["model"],
            prompt,
        )
        df.loc[idx, "verifier_provider"] = config["verifier_agent"]["provider"]
        df.loc[idx, "verifier_model"] = config["verifier_agent"]["model"]
        df.loc[idx, "verifier_model_name"] = config["verifier_agent"]["model"]
        df.loc[idx, "hallucination_label"] = result.label
        df.loc[idx, "hallucination_reason"] = result.reason

        save_results(df, base_output_path)
        save_results(df[verified_columns()], verified_output_path)

    metrics = compute_metrics(df)
    base_path = save_results(df, base_output_path)
    verified_path = save_results(df[verified_columns()], verified_output_path)
    print(f"Saved results to {base_path}")
    print(f"Saved verifier labels to {verified_path}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Error rate: {metrics['error_rate']:.4f}")
    print(f"Hallucination counts: {metrics['hallucination_counts']}")


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    runtime = config["runtime"]
    knowledge_limit = args.knowledge_limit or runtime["knowledge_limit"]
    if args.answer_only and args.verify_only:
        raise ValueError("--answer-only and --verify-only cannot be used together.")

    records = load_split(PROJECT_ROOT / runtime["data_dir"], runtime["eval_split"], require_labels=True)
    records = sample_records(records, args.sample_size or runtime["sample_size"])

    retriever = (
        ensure_knowledge_assets(config, knowledge_limit=knowledge_limit, rebuild_kb=args.rebuild_kb)
        if args.mode == "rag"
        else None
    )
    graph = build_graph()
    base_output_path, verified_output_path = result_paths(config, args.mode)

    if args.verify_only:
        run_verifier_only(
            config=config,
            mode=args.mode,
            base_output_path=base_output_path,
            verified_output_path=verified_output_path,
        )
        return

    rows = []
    done_ids: set[str] = set()
    if args.resume and base_output_path.exists():
        rows = load_resume_rows(
            base_output_path,
            answer_provider=config["answer_agent"]["provider"],
            answer_model=config["answer_agent"]["model"],
            verifier_provider=config["verifier_agent"]["provider"],
            verifier_model=config["verifier_agent"]["model"],
        )
        done_ids = {str(row["id"]) for row in rows}
        records = [record for record in records if record.id not in done_ids]
    if args.batch_size:
        records = records[: args.batch_size]
    try:
        from tqdm import tqdm

        iterator = tqdm(records, total=len(records))
    except Exception:
        iterator = records

    for record in iterator:
        state = graph.invoke(
            {
                "question": record.question,
                "options": options_dict(record),
                "correct_letter": record.correct_letter,
                "correct_text": record.correct_text,
                "clean_exp": record.clean_exp,
                "record_id": record.id,
                "answer_provider": config["answer_agent"]["provider"],
                "answer_model": config["answer_agent"]["model"],
                "verifier_provider": config["verifier_agent"]["provider"],
                "verifier_model": config["verifier_agent"]["model"],
                "retriever": retriever,
                "top_k": config["retrieval"]["top_k"],
                "mode": args.mode,
                "skip_verifier": args.answer_only,
            }
        )
        rows.append(
            {
                "id": record.id,
                "question": record.question,
                "A": record.A,
                "B": record.B,
                "C": record.C,
                "D": record.D,
                "correct_letter": record.correct_letter,
                "correct_text": record.correct_text,
                "clean_exp": record.clean_exp,
                "mode": args.mode,
                "model_name": config["answer_agent"]["provider"],
                "model_id": config["answer_agent"]["model"],
                "answer_provider": config["answer_agent"]["provider"],
                "answer_model_name": config["answer_agent"]["model"],
                "verifier_provider": config["verifier_agent"]["provider"],
                "verifier_model": config["verifier_agent"]["model"],
                "verifier_model_name": config["verifier_agent"]["model"],
                "predicted_letter": state.get("predicted_letter", ""),
                "predicted_text": state.get("predicted_text", ""),
                "is_correct": state.get("is_correct", False),
                "retrieved_context": state.get("retrieved_context", ""),
                "retrieved_ids": state.get("retrieved_ids", ""),
                "retrieved_scores": state.get("retrieved_scores", ""),
                "hallucination_label": state.get("hallucination_label", ""),
                "hallucination_reason": state.get("hallucination_reason", ""),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        df = pd.DataFrame(rows)
        save_results(df, base_output_path)
        save_results(df[verified_columns()], verified_output_path)
        if runtime.get("sleep_seconds", 0):
            time.sleep(runtime["sleep_seconds"])

    df = pd.DataFrame(rows)
    metrics = compute_metrics(df)

    base_path = save_results(df, base_output_path)
    verified_path = save_results(df[verified_columns()], verified_output_path)

    print(f"Saved results to {base_path}")
    print(f"Saved verifier labels to {verified_path}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Error rate: {metrics['error_rate']:.4f}")
    print(f"Hallucination counts: {metrics['hallucination_counts']}")


if __name__ == "__main__":
    main()
