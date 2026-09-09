from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT_DIR = ROOT / "analysis"
FIGURES_DIR = MANUSCRIPT_DIR / "figures"
TABLES_DIR = MANUSCRIPT_DIR / "tables"
STAGE2_OUTPUTS_ROOT = ROOT / "stage2_task_adapter" / "outputs"

RUN_NAMES = (
    "bladder_stage2_from_grouped_split_l40s4",
    "ear_stage2_from_grouped_split_l40s4",
    "eye_stage2_from_grouped_split_l40s4",
    "heart_stage2_from_grouped_split_l40s4",
    "ovary_stage2_from_grouped_split_l40s4",
    "pancreas_stage2_from_grouped_split_l40s4",
    "prostate_stage2_from_grouped_split_l40s4",
    "salivary_gland_stage2_from_grouped_split_l40s4",
    "small_intestine_stage2_from_grouped_split_l40s4",
    "spleen_stage2_from_grouped_split_l40s4",
)

SUMMARY_CSV = TABLES_DIR / "stage2_grouped_summary.csv"
MAIN_TABLE_TEX = TABLES_DIR / "stage2_main_table.tex"
MAIN_FIGURE_PDF = FIGURES_DIR / "stage2_grouped_benchmark.pdf"
MAIN_FIGURE_PNG = FIGURES_DIR / "stage2_grouped_benchmark.png"

plt.rcParams.update(
    {
        "font.size": 12,
        "font.weight": "semibold",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "axes.labelweight": "bold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pretty_dataset_name(run_name: str) -> str:
    dataset = run_name.split("_stage2_from_grouped_split", 1)[0]
    return dataset.replace("_", " ").title()


def display_dataset_name(label: str) -> str:
    if label == "Salivary Gland":
        return "Salivary\ngland"
    if label == "Small Intestine":
        return "Small\nintestine"
    wrapped = textwrap.wrap(label, width=10)
    return "\n".join(wrapped) if len(wrapped) > 1 else label


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def load_run_summary(run_name: str) -> dict[str, float | int | str]:
    run_dir = STAGE2_OUTPUTS_ROOT / run_name
    rows = load_jsonl(run_dir / "test_predictions.jsonl")

    total_examples = len(rows)
    exact_matches = sum(bool(row.get("exact_match")) for row in rows)
    total_generated_evidence = sum(len(row.get("generated_evidence_genes") or []) for row in rows)
    total_ground_truth_evidence = sum(len(row.get("ground_truth_evidence_genes") or []) for row in rows)
    total_evidence_in_input = sum(int(row.get("evidence_in_input_count") or 0) for row in rows)
    total_evidence_ground_truth = sum(int(row.get("evidence_ground_truth_match_count") or 0) for row in rows)
    total_elapsed = sum(float(row.get("elapsed_sec") or 0.0) for row in rows)

    return {
        "run_name": run_name,
        "dataset": pretty_dataset_name(run_name),
        "n_test": total_examples,
        "exact_match_rate": safe_divide(exact_matches, total_examples),
        "evidence_in_input_rate": safe_divide(total_evidence_in_input, total_generated_evidence),
        "evidence_ground_truth_match_rate": safe_divide(
            total_evidence_ground_truth,
            total_ground_truth_evidence,
        ),
        "mean_elapsed_sec": safe_divide(total_elapsed, total_examples),
    }


def build_summary_table() -> tuple[pd.DataFrame, dict[str, float]]:
    rows = [load_run_summary(run_name) for run_name in RUN_NAMES]
    summary_df = pd.DataFrame(rows)
    overall = {
        "n_tissues": float(summary_df.shape[0]),
        "total_samples": float(summary_df["n_test"].sum()),
        "mean_exact_match_rate": float(summary_df["exact_match_rate"].mean()),
        "std_exact_match_rate": float(summary_df["exact_match_rate"].std(ddof=1)),
        "mean_evidence_in_input_rate": float(summary_df["evidence_in_input_rate"].mean()),
        "mean_evidence_ground_truth_match_rate": float(summary_df["evidence_ground_truth_match_rate"].mean()),
        "mean_elapsed_sec": float(summary_df["mean_elapsed_sec"].mean()),
    }
    return summary_df, overall


def bold_y_tick_labels(ax: plt.Axes) -> None:
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.04, label, transform=ax.transAxes, fontsize=16, fontweight="bold")


def draw_metric_panel(
    ax: plt.Axes,
    positions: list[int],
    values: list[float],
    color: str,
    title: str,
    ylabel: str,
    *,
    y_limit: float | None = None,
    label_formatter=None,
) -> None:
    bars = ax.bar(positions, values, color=color, width=0.65)
    upper_limit = y_limit if y_limit is not None else (max(values) * 1.18 if values else 1.0)
    ax.set_ylim(0, upper_limit)
    ax.margins(x=0.07)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    format_label = label_formatter or (lambda value: f"{value:.1f}")
    offset = upper_limit * 0.03
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            format_label(value),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    bold_y_tick_labels(ax)


def draw_figure(summary_df: pd.DataFrame) -> None:
    labels = [display_dataset_name(label) for label in summary_df["dataset"].tolist()]
    positions = list(range(len(labels)))
    exact_values = (summary_df["exact_match_rate"] * 100).tolist()
    evidence_values = (summary_df["evidence_ground_truth_match_rate"] * 100).tolist()
    support_values = summary_df["n_test"].tolist()

    fig = plt.figure(figsize=(13.8, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.05), hspace=0.18, wspace=0.24)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, :]),
    ]

    draw_metric_panel(
        axes[0],
        positions,
        exact_values,
        "#2563EB",
        "Exact match",
        "Percent",
        y_limit=105.0,
    )
    add_panel_label(axes[0], "a")

    draw_metric_panel(
        axes[1],
        positions,
        evidence_values,
        "#0F766E",
        "Evidence grounding",
        "Percent",
        y_limit=105.0,
    )
    add_panel_label(axes[1], "b")

    draw_metric_panel(
        axes[2],
        positions,
        support_values,
        "#D97706",
        "Held-out cells",
        "Cells",
        label_formatter=lambda value: f"{int(value)}",
    )
    add_panel_label(axes[2], "c")

    axes[0].set_xticks(positions, [])
    axes[1].set_xticks(positions, [])
    axes[2].set_xticks(positions, labels)
    axes[2].tick_params(axis="x", rotation=24, labelsize=10)
    for label in axes[2].get_xticklabels():
        label.set_horizontalalignment("right")

    fig.savefig(MAIN_FIGURE_PDF, bbox_inches="tight")
    fig.savefig(MAIN_FIGURE_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_main_table_tex(summary_df: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Stage 2 cell-type annotation performance across ten Tabula Sapiens tissues. Exact match is the primary task metric. Evidence-grounding reports the fraction of ground-truth evidence genes recovered by the generated rationale.}",
        r"\label{tab:stage2_grouped_main}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Tissue & $N_{\mathrm{test}}$ & Exact match & Evidence-grounding & Mean latency (s) \\",
        r"\midrule",
    ]

    for row in summary_df.itertuples(index=False):
        lines.append(
            f"{row.dataset} & {int(row.n_test)} & {row.exact_match_rate:.3f} & {row.evidence_ground_truth_match_rate:.3f} & {row.mean_elapsed_sec:.2f} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    MAIN_TABLE_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    summary_df, overall = build_summary_table()
    summary_df.to_csv(SUMMARY_CSV, index=False)
    draw_figure(summary_df)
    write_main_table_tex(summary_df)

    print(summary_df.to_string(index=False))
    print(json.dumps(overall, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()