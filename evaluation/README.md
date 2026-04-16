# Evaluation Outputs

This folder contains plotting helpers and generated charts for comparing `no_rag` runs.

Run the plotting script with:

```bash
./.venv/bin/python evaluation/plot_no_rag_comparison.py
```

Generated files are saved in:

- `evaluation/outputs/no_rag_summary.csv`
- `evaluation/outputs/no_rag_record_counts.png`
- `evaluation/outputs/no_rag_hallucination_distribution.png`
- `evaluation/outputs/no_rag_hallucination_share.png`

The current script compares every `results/*/no_rag_results.csv` file it finds.
