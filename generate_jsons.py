from datasets import load_dataset

dataset = load_dataset("medmcqa")

# Export to JSONL files in the data/ directory
dataset["train"].to_json("data/train.json", orient="records", lines=True)
 