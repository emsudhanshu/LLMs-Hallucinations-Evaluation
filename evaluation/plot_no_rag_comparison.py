from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "outputs"


@dataclass(frozen=True)
class RunSummary:
    run_folder: str
    run_label: str
    total_records: int
    correct_records: int
    incorrect_records: int
    factual_error: int
    reasoning_failure: int


def discover_no_rag_runs() -> list[Path]:
    return sorted(RESULTS_ROOT.glob("*/no_rag_results.csv"))


def build_run_label(df: pd.DataFrame, run_folder: str) -> str:
    answer_model = str(df["answer_model_name"].dropna().iloc[0]) if "answer_model_name" in df.columns else "unknown"
    verifier_model = (
        str(df["verifier_model_name"].dropna().iloc[0]) if "verifier_model_name" in df.columns else "unknown"
    )
    return f"Answer: {answer_model} | Verifier: {verifier_model}"


def summarize_run(path: Path) -> RunSummary:
    df = pd.read_csv(path)
    incorrect_mask = df["is_correct"] == False
    label_counts = df.loc[incorrect_mask, "hallucination_label"].fillna("").value_counts()
    factual_error = int(label_counts.get("FACTUAL_ERROR", 0))
    reasoning_failure = int(label_counts.get("REASONING_FAILURE", 0))
    incorrect_records = int(incorrect_mask.sum())
    total_records = len(df)
    return RunSummary(
        run_folder=path.parent.name,
        run_label=build_run_label(df, path.parent.name),
        total_records=total_records,
        correct_records=total_records - incorrect_records,
        incorrect_records=incorrect_records,
        factual_error=factual_error,
        reasoning_failure=reasoning_failure,
    )


def summary_frame(summaries: list[RunSummary]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_folder": item.run_folder,
                "run_label": item.run_label,
                "total_records": item.total_records,
                "correct_records": item.correct_records,
                "incorrect_records": item.incorrect_records,
                "factual_error": item.factual_error,
                "reasoning_failure": item.reasoning_failure,
            }
            for item in summaries
        ]
    )


def save_summary_csv(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "no_rag_summary.csv", index=False)


def plot_total_vs_incorrect(df: pd.DataFrame) -> None:
    melted = df.melt(
        id_vars=["run_label"],
        value_vars=["total_records", "correct_records", "incorrect_records"],
        var_name="metric",
        value_name="count",
    )
    metric_order = ["total_records", "correct_records", "incorrect_records"]
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=melted, x="run_label", y="count", hue="metric", hue_order=metric_order)
    ax.set_title("No-RAG Record Counts by Run")
    ax.set_xlabel("Run")
    ax.set_ylabel("Number of Records")
    ax.legend(title="Metric", labels=["Total", "Correct", "Incorrect"])
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "no_rag_record_counts.png", dpi=200)
    plt.close()


def plot_hallucination_breakdown(df: pd.DataFrame) -> None:
    plot_df = df[["run_label", "factual_error", "reasoning_failure"]].copy()
    plot_df = plot_df.rename(
        columns={
            "factual_error": "FACTUAL_ERROR",
            "reasoning_failure": "REASONING_FAILURE",
        }
    )
    long_df = plot_df.melt(id_vars=["run_label"], var_name="hallucination_label", value_name="count")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=long_df, x="run_label", y="count", hue="hallucination_label")
    ax.set_title("No-RAG Hallucination Label Distribution")
    ax.set_xlabel("Run")
    ax.set_ylabel("Incorrect Records")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "no_rag_hallucination_distribution.png", dpi=200)
    plt.close()


def plot_hallucination_share(df: pd.DataFrame) -> None:
    share_df = df[["run_label", "incorrect_records", "factual_error", "reasoning_failure"]].copy()
    share_df["factual_share"] = share_df["factual_error"] / share_df["incorrect_records"]
    share_df["reasoning_share"] = share_df["reasoning_failure"] / share_df["incorrect_records"]

    plt.figure(figsize=(10, 6))
    plt.bar(share_df["run_label"], share_df["factual_share"], label="FACTUAL_ERROR")
    plt.bar(
        share_df["run_label"],
        share_df["reasoning_share"],
        bottom=share_df["factual_share"],
        label="REASONING_FAILURE",
    )
    plt.title("No-RAG Hallucination Share Within Incorrect Predictions")
    plt.xlabel("Run")
    plt.ylabel("Share of Incorrect Predictions")
    plt.ylim(0, 1)
    plt.legend()
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "no_rag_hallucination_share.png", dpi=200)
    plt.close()


def main() -> None:
    run_paths = discover_no_rag_runs()
    if not run_paths:
        raise FileNotFoundError(f"No no_rag_results.csv files found under {RESULTS_ROOT}")

    summaries = [summarize_run(path) for path in run_paths]
    df = summary_frame(summaries)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_summary_csv(df)
    plot_total_vs_incorrect(df)
    plot_hallucination_breakdown(df)
    plot_hallucination_share(df)

    print("Saved outputs to", OUTPUT_DIR)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
