#!/usr/bin/env python3
"""Plot the recorded final feedback sequence; requires matplotlib.

Run from any directory. The default input and SVG output are repository-relative.
Use --output /path/to/preview.png to render a PNG instead.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(
    "trackb/docs/validation/gpu_final_2026-10-18/"
    "final_4080_v1/results/vlm_summary.csv"
)
OUTPUT = Path("trackb/docs/figures/feedback_results.svg")


def read_results():
    with (ROOT / SOURCE).open(newline="", encoding="utf-8") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["n_feedback"]))
    if [int(row["n_feedback"]) for row in rows] != [200, 300, 400]:
        raise ValueError("Expected the recorded 200, 300, and 400-label stages")
    if any(int(row["n_eval"]) != 3590 for row in rows):
        raise ValueError("All stages must use the fixed 3,590-tile evaluation set")
    if any(row["run_id"] != "final_4080_v1" for row in rows):
        raise ValueError("Expected the final_4080_v1 recorded sequence")
    pair_counts = set()
    for row in rows:
        metrics = ROOT / SOURCE.parent / "by_model" / row["model_version"] / "metrics.json"
        matrix = json.loads(metrics.read_text())["confusion_matrix"]
        pair_n = sum(matrix["MUS"].values()) + sum(matrix["STR"].values())
        pair_errors = matrix["MUS"].get("STR", 0) + matrix["STR"].get("MUS", 0)
        if not math.isclose(pair_errors / pair_n, float(row["pair_error_STR_MUS"]), abs_tol=1e-12):
            raise ValueError("Mutual-error rate does not match the saved confusion matrix")
        pair_counts.add(pair_n)
    if len(pair_counts) != 1:
        raise ValueError("The MUS/STR evaluation denominator changed between stages")
    return rows, pair_counts.pop()


def plot(output):
    rows, pair_n = read_results()
    labels = [int(row["n_feedback"]) for row in rows]
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelcolor": "#334155",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "svg.hashsalt": "hardpair-feedback-results",
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9))
    fig.subplots_adjust(left=0.075, right=0.97, bottom=0.22, top=0.73, wspace=0.32)
    fig.text(0.075, 0.94, "Classification across feedback updates",
             fontsize=19, fontweight="bold", color="#0f172a")
    fig.text(0.075, 0.88,
             "Recorded 200 / 300 / 400-label stages on 3,590 evaluation tiles",
             fontsize=11.5, color="#475569")

    panels = [
        ("Balanced accuracy", "balanced_accuracy", "#2563eb", "o",
         (97.5, 100), [97.5, 98, 98.5, 99, 99.5, 100]),
        ("Muscle–stroma mutual error", "pair_error_STR_MUS", "#c25e16", "s",
         (0, 6.5), [0, 2, 4, 6]),
    ]
    for ax, (title, key, color, marker, limits, ticks) in zip(axes, panels):
        values = [100 * float(row[key]) for row in rows]
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold",
                     color="#0f172a", pad=16)
        ax.plot(labels, values, color=color, linewidth=2.4, marker=marker,
                markersize=7, markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        for x, y in zip(labels, values):
            below = (key == "balanced_accuracy" and x == 200) or (
                key == "pair_error_STR_MUS" and x > 200
            )
            ax.annotate(f"{y:.2f}%", (x, y), xytext=(0, -13 if below else 11),
                        textcoords="offset points", ha="center", va="top" if below else "bottom",
                        fontsize=11, fontweight="bold", color=color)
        ax.set_xlim(175, 425)
        ax.set_ylim(*limits)
        ax.set_xticks(labels)
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g%%"))
        ax.set_xlabel("Cumulative feedback labels", labelpad=10)
        ax.grid(axis="y", color="#e2e8f0", linewidth=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#cbd5e1")
        ax.tick_params(axis="both", length=0, pad=7)

    fig.text(0.075, 0.04,
             f"Simulated dataset labels · One recorded sequence · MUS/STR subset: {pair_n} tiles",
             fontsize=10.5, color="#475569")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Date": None} if output.suffix.lower() == ".svg" else None
    fig.savefig(output, dpi=180, metadata=metadata)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT)
    plot(parser.parse_args().output)
