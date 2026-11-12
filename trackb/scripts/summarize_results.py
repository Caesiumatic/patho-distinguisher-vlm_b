#!/usr/bin/env python3
"""Summarize the saved final classifier and explanation metrics without a GPU."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir", type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs/validation/gpu_final_2026-10-18/final_4080_v1",
    )
    args = parser.parse_args()
    with (args.report_dir / "results/vlm_summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    print("Stage          Feedback  Accuracy  Balanced acc.  STR/MUS error")
    for row in rows:
        print(f"{row['stage']:<14} {int(row['n_feedback']):>8}  "
              f"{float(row['accuracy']):>8.2%}  "
              f"{float(row['balanced_accuracy']):>13.2%}  "
              f"{float(row['pair_error_STR_MUS']):>13.2%}")

    final = max(rows, key=lambda row: int(row["n_feedback"]))
    model_dir = args.report_dir / "results/by_model" / final["model_version"]
    metrics = json.loads((model_dir / "metrics.json").read_text())
    explanations = json.loads((model_dir / "explanation_summary.json").read_text())
    matrix = metrics["confusion_matrix"]
    errors = sum(count for actual, predictions in matrix.items()
                 for predicted, count in predictions.items() if predicted != actual)
    supported = metrics["explanation_sources"]["evidence_conditioned_student"]
    n = metrics["n_eval"]
    print(f"\nFinal errors: {errors}/{n}")
    print(f"Grounded explanations: {supported}/{n} ({supported / n:.1%})")
    print(f"Classifier: {final['model_version']}")
    print(f"Explanation model: {explanations['qwen_version']}")
    print("Feedback source: simulated dataset labels; pool and eval share the CRC-VAL domain.")


if __name__ == "__main__":
    main()
