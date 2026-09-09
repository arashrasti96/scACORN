#!/usr/bin/env python3
"""Visualize comparison results from saved CSV.

Usage
-----
  python plot_comparison.py                           # auto-detect results/comparison_results.csv
  python plot_comparison.py --csv /path/to/results.csv
  python plot_comparison.py --metric R@10             # change the primary bar-chart metric
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


_DEFAULT_CSV = Path(__file__).resolve().parent / "results" / "comparison_results.csv"

_METRICS = ["R@1", "R@5", "R@10", "NMI", "ARI", "Silhouette"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot comparison results.")
    parser.add_argument(
        "--csv",
        type=str,
        default=str(_DEFAULT_CSV),
        help="Path to the comparison_results.csv file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output plots (default: same as CSV dir).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="R@10",
        choices=_METRICS,
        help="Primary metric for the main bar chart.",
    )
    return parser.parse_args()


def _short_dataset_name(full_name: str) -> str:
    """Shorten e.g. 'tabula_sapiens_heart_cell_annotation' → 'heart'."""
    parts = full_name.replace("tabula_sapiens_", "").replace("_cell_annotation", "")
    return parts


def plot_grouped_bar(
    df: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    """Grouped bar chart: datasets × methods for one metric."""
    datasets = df["dataset_short"].unique()
    methods = df["method"].unique()

    x = np.arange(len(datasets))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 1.5), 6))

    for i, method in enumerate(methods):
        vals = []
        for ds in datasets:
            row = df[(df["dataset_short"] == ds) & (df["method"] == method)]
            v = row[metric].values[0] if len(row) and pd.notna(row[metric].values[0]) else 0
            vals.append(v)
        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=method)

    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by Method and Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_method_summary(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Horizontal bar chart: average of each metric across datasets per method."""
    methods = df["method"].unique()
    available_metrics = [m for m in _METRICS if m in df.columns]

    fig, ax = plt.subplots(figsize=(10, max(4, len(methods) * 0.6)))

    y = np.arange(len(methods))
    bar_height = 0.8 / len(available_metrics)

    for i, metric in enumerate(available_metrics):
        means = []
        for method in methods:
            vals = df[df["method"] == method][metric].dropna()
            means.append(vals.mean() if len(vals) else 0)
        offset = (i - len(available_metrics) / 2 + 0.5) * bar_height
        ax.barh(y + offset, means, bar_height, label=metric)

    ax.set_xlabel("Score")
    ax.set_title("Average Metrics per Method (across datasets)")
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_heatmap(
    df: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    """Heatmap: methods (rows) × datasets (cols) for one metric."""
    pivot = df.pivot_table(
        values=metric, index="method", columns="dataset_short", aggfunc="first"
    )

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.2), max(4, len(pivot) * 0.6)))
    data = pivot.values.astype(float)
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = data[i, j]
            if np.isnan(v):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=8, color="gray")
            else:
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.6 else "black")

    ax.set_title(f"{metric} Heatmap")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"ERROR: CSV not found: {csv_path}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df["dataset_short"] = df["dataset"].apply(_short_dataset_name)

    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Datasets: {df['dataset_short'].unique().tolist()}")
    print(f"Methods:  {df['method'].unique().tolist()}\n")

    # 1. Grouped bar chart for the primary metric
    plot_grouped_bar(
        df, args.metric, output_dir / f"bar_{args.metric.replace('@', '_at_')}.png"
    )

    # 2. Method summary (all metrics averaged)
    plot_method_summary(df, output_dir / "method_summary.png")

    # 3. Heatmaps for each metric
    for metric in _METRICS:
        if metric in df.columns and df[metric].notna().any():
            plot_heatmap(
                df, metric, output_dir / f"heatmap_{metric.replace('@', '_at_')}.png"
            )

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
