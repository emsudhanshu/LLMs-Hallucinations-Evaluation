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
    parser.add_argument(
        "--verifier-provider",
        default="openai",
        choices=["openai", "gemini", "ollama", "llama"],
        help="LLM provider to use for the verifier (default: openai).",
    )
    parser.add_argument(
        "--verifier-model",
        default=None,
        help=(
            "Model name for the verifier. "
            "Defaults: openai→gpt-4o-mini, gemini→gemini-1.5-flash, ollama/llama→llama3.2"
        ),
    )
    return parser.parse_args()


_PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-1.5-flash",
    "ollama": "llama3.2",
    "llama": "llama3.2",
}


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    df = pd.read_csv(input_path)

    provider = args.verifier_provider
    model = args.verifier_model or _PROVIDER_DEFAULT_MODELS.get(provider, "gpt-4o-mini")
    print(f"Using verifier: provider={provider!r}  model={model!r}")

    # Ensure output columns exist with object dtype so string assignment never
    # raises LossySetitemError (pandas reads all-NaN columns as float64).
    for col in ("verifier_label", "verifier_explanation"):
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].astype(object)

    option_map = {"A": "option_a", "B": "option_b", "C": "option_c", "D": "option_d"}

    for idx, row in df.loc[df["is_correct"] == False].iterrows():
        options = {
            "A": row["option_a"],
            "B": row["option_b"],
            "C": row["option_c"],
            "D": row["option_d"],
        }
        answer_letter = str(row["model_answer"]).strip().upper()
        answer_text = row.get(option_map.get(answer_letter, "option_a"), "")
        result = classify_incorrect_answer(
            question=row["question"],
            options=options,
            correct_answer=row["correct_answer"],
            model_answer_letter=answer_letter,
            model_answer_text=answer_text,
            provider=provider,
            model=model,
        )
        df.at[idx, "verifier_label"] = result.label
        df.at[idx, "verifier_explanation"] = result.explanation

    output_path = input_path.with_name(f"{input_path.stem}_verified.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved verified results to {output_path}")


if __name__ == "__main__":
    main()
