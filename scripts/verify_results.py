from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm_hallucinations.verifier import classify_incorrect_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify wrong answers and assign hallucination labels.")
    parser.add_argument("--input", required=True, help="Path to a baseline or RAG CSV file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    df = pd.read_csv(input_path)

    if "verifier_label" not in df.columns:
        df["verifier_label"] = None
    df["verifier_label"] = df["verifier_label"].astype(object)

    if "verifier_explanation" not in df.columns:
        df["verifier_explanation"] = None
    df["verifier_explanation"] = df["verifier_explanation"].astype(object)

    for idx, row in df.loc[df["is_correct"] == False].iterrows():
        result = classify_incorrect_answer(
            question=row["question"],
            options={
                "A": row["option_a"],
                "B": row["option_b"],
                "C": row["option_c"],
                "D": row["option_d"],
            },
            correct_answer=row["correct_answer"],
            model_answer=row["model_answer"],
        )
        df.at[idx, "verifier_label"] = result.label
        df.at[idx, "verifier_explanation"] = result.explanation

    output_path = input_path.with_name(f"{input_path.stem}_verified.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved verified results to {output_path}")


if __name__ == "__main__":
    main()
