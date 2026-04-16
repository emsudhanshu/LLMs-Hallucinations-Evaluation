from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm_hallucinations.config import ExperimentConfig
from llm_hallucinations.pipeline import run_no_rag_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the no-RAG medical QA baseline.")
    parser.add_argument("--model", required=True, choices=["mock", "openai", "gemini", "llama"])
    parser.add_argument("--split", default="dev", choices=["train", "dev"])
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        model_name=args.model,
        split=args.split,
        sample_size=args.sample_size,
        seed=args.seed,
        temperature=args.temperature,
        sleep_seconds=args.sleep_seconds,
    )
    df, output_path = run_no_rag_experiment(config)
    accuracy = df["is_correct"].mean() if not df.empty else 0.0
    print(f"Saved {len(df)} rows to {output_path}")
    print(f"Accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()
