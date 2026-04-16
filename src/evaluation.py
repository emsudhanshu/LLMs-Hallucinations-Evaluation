from __future__ import annotations

from pathlib import Path

import pandas as pd


def compute_metrics(df: pd.DataFrame) -> dict[str, object]:
    if df.empty:
        return {"accuracy": 0.0, "error_rate": 0.0, "hallucination_counts": {}}
    accuracy = float(df["is_correct"].mean())
    error_rate = 1.0 - accuracy
    hallucination_counts = (
        df.loc[df["is_correct"] == False, "hallucination_label"].value_counts(dropna=True).to_dict()
        if "hallucination_label" in df.columns
        else {}
    )
    return {
        "accuracy": accuracy,
        "error_rate": error_rate,
        "hallucination_counts": hallucination_counts,
    }


def save_results(df: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path

