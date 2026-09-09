from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT_DIR = ROOT / "analysis"
FIGURES_DIR = MANUSCRIPT_DIR / "figures"
TABLES_DIR = MANUSCRIPT_DIR / "tables"
EMBEDDING_ROOT = MANUSCRIPT_DIR / "artifacts" / "stage1_embedding_exports" / "grouped"
STAGE2_SCRIPT_DIR = MANUSCRIPT_DIR / "scripts"

DOMAIN_STEM = FIGURES_DIR / "results_domain_alignment_composite"
EXPERT_STEM = FIGURES_DIR / "results_expert_specialization_composite"
PNG_DPI = 450
UMAP_RANDOM_STATE = 42

METHOD_ORDER = (
    "c2s_base",
    "c2s_lora_projections",
    "c2s_lora_features",
    "tfidf_svd",
    "rank_pca",
)
METHOD_LABELS = {
    "c2s_base": "scSLM base",
    "c2s_lora_projections": "scSLM projections",
    "c2s_lora_features": "scSLM hidden features",
    "tfidf_svd": "TF-IDF + SVD",
    "rank_pca": "Rank PCA",
}
METHOD_COLORS = {
    "c2s_base": "#777777",
    "c2s_lora_projections": "#9C3D54",
    "c2s_lora_features": "#D17B0F",
    "tfidf_svd": "#0B6E4F",
    "rank_pca": "#3A7CA5",
}
TISSUES = (
    ("tabula_sapiens_prostate_cell_annotation", "Prostate"),
    ("tabula_sapiens_stomach_cell_annotation", "Stomach"),
)


def set_figure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 5.8,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def add_panel_label(ax: plt.Axes, label: str, *, x: float = -0.15, y: float = 1.05) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color="#111111",
    )


def clean_metric_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#E2E2E2", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".png"), dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def draw_domain_metric(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    mean_column: str,
    std_column: str,
    xlabel: str,
    xlim: tuple[float, float],
    panel: str,
) -> None:
    ordered = summary.set_index("method_key").loc[list(METHOD_ORDER)]
    y = np.arange(len(METHOD_ORDER))
    ax.barh(
        y,
        ordered[mean_column],
        xerr=ordered[std_column],
        height=0.66,
        color=[METHOD_COLORS[key] for key in METHOD_ORDER],
        edgecolor="none",
        error_kw={"elinewidth": 0.7, "capsize": 1.8, "capthick": 0.7, "ecolor": "#333333"},
        zorder=2,
    )
    ax.set_yticks(y, [METHOD_LABELS[key] for key in METHOD_ORDER])
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    clean_metric_axis(ax)
    add_panel_label(ax, panel)


def draw_tissue_heatmap(ax: plt.Axes, shared: pd.DataFrame) -> None:
    heatmap = (
        shared.pivot(index="tissue_name", columns="method_key", values="transfer_knn_macro_f1")
        .reindex(index=sorted(shared["tissue_name"].unique()), columns=list(METHOD_ORDER))
    )
    image = ax.imshow(heatmap.to_numpy(), aspect="auto", cmap="YlGn", vmin=0.20, vmax=0.85)
    ax.set_xticks(
        np.arange(len(METHOD_ORDER)),
        [METHOD_LABELS[key].replace("scSLM ", "") for key in METHOD_ORDER],
        rotation=25,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(np.arange(len(heatmap.index)), heatmap.index)
    ax.tick_params(length=0, pad=1.3)
    for row in range(heatmap.shape[0]):
        for column in range(heatmap.shape[1]):
            value = heatmap.iat[row, column]
            text_color = "white" if value > 0.70 else "#111111"
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=4.6, color=text_color)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Macro-F1", labelpad=2)
    colorbar.ax.tick_params(labelsize=5.6, width=0.5, length=2)
    colorbar.outline.set_linewidth(0.5)
    add_panel_label(ax, "c", x=-0.24)


def draw_existing_umap_panel(ax: plt.Axes) -> None:
    image_path = FIGURES_DIR / "stage1_embedding_umaps.png"
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing UMAP montage: {image_path}")
    image = mpimg.imread(image_path)
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    add_panel_label(ax, "d", x=-0.03, y=1.01)


def draw_existing_embedding_panel(ax: plt.Axes, image_name: str, title: str) -> None:
    image_path = FIGURES_DIR / image_name
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing embedding panel: {image_path}")
    image = mpimg.imread(image_path)
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, pad=3.0, fontsize=7.8, fontweight="semibold")
    for spine in ax.spines.values():
        spine.set_visible(False)


def generate_domain_alignment_figure() -> None:
    summary = pd.read_csv(TABLES_DIR / "stage1_grouped_summary.csv")
    shared = pd.read_csv(TABLES_DIR / "stage1_grouped_test_shared.csv")

    fig = plt.figure(figsize=(7.18, 6.55))
    grid = fig.add_gridspec(
        2,
        15,
        height_ratios=(1.02, 1.98),
        hspace=0.20,
        wspace=0.22,
    )
    macro_ax = fig.add_subplot(grid[0, 0:4])
    recall_ax = fig.add_subplot(grid[0, 4:8])
    heatmap_ax = fig.add_subplot(grid[0, 8:15])

    draw_domain_metric(
        macro_ax,
        summary,
        mean_column="transfer_knn_macro_f1_mean",
        std_column="transfer_knn_macro_f1_std",
        xlabel="Transfer macro-F1",
        xlim=(0.0, 0.92),
        panel="a",
    )
    draw_domain_metric(
        recall_ax,
        summary,
        mean_column="transfer_R5_mean",
        std_column="transfer_R5_std",
        xlabel="Transfer Recall@5",
        xlim=(0.76, 1.01),
        panel="b",
    )
    draw_tissue_heatmap(heatmap_ax, shared)
    prostate_ax = fig.add_subplot(grid[1, 0:8])
    stomach_ax = fig.add_subplot(grid[1, 8:15])
    draw_existing_embedding_panel(prostate_ax, "stage1_embedding_methods_prostate.png", "Prostate")
    draw_existing_embedding_panel(stomach_ax, "stage1_embedding_methods_stomach.png", "Stomach")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.97, bottom=0.05)
    save_figure(fig, DOMAIN_STEM)


def stage2_data():
    if str(STAGE2_SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(STAGE2_SCRIPT_DIR))
    from stage2_exp_figure_common import EVIDENCE_METRICS, LABEL_METRICS, METHOD_SPECS, NEGATIVE_METRICS

    return METHOD_SPECS, LABEL_METRICS, EVIDENCE_METRICS, NEGATIVE_METRICS


def draw_expert_metric(
    ax: plt.Axes,
    values: dict[str, float],
    method_specs,
    *,
    title: str,
    panel: str,
    show_method_labels: bool,
) -> None:
    y = np.arange(len(method_specs))
    for index, spec in enumerate(method_specs):
        value = values[spec.key]
        ax.hlines(index, 0, value, color=spec.color, linewidth=1.4, alpha=0.58, zorder=1)
        marker = "D" if spec.key == "scslm" else "o"
        size = 29 if spec.key == "scslm" else 20
        ax.scatter(value, index, s=size, marker=marker, color=spec.color, edgecolors="white", linewidths=0.45, zorder=3)
        ax.text(
            min(value + 0.025, 0.985),
            index,
            f"{value:.2f}",
            va="center",
            ha="left" if value < 0.96 else "right",
            fontsize=5.4,
            fontweight="semibold" if spec.key == "scslm" else "normal",
        )

    boundaries = [0.5, 3.5, 10.5]
    for boundary in boundaries:
        ax.axhline(boundary, color="#BEBEBE", linewidth=0.55, linestyle=(0, (2, 2)), zorder=0)
    ax.set_ylim(len(method_specs) - 0.45, -0.55)
    ax.set_xlim(0, 1.03)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("")
    ax.set_title(title, pad=5, fontsize=7.2, fontweight="semibold")
    if show_method_labels:
        ax.set_yticks(y, [spec.display_name for spec in method_specs])
    else:
        ax.set_yticks(y, [])
    ax.tick_params(axis="y", length=0, pad=2)
    clean_metric_axis(ax)
    add_panel_label(ax, panel, x=-0.18 if show_method_labels else -0.08, y=1.06)


def draw_preservation_ablation(fig: plt.Figure, grid_cell) -> None:
    ax = fig.add_subplot(grid_cell)
    ax.axis("off")
    add_panel_label(ax, "e", x=-0.01, y=1.02)
    ax.text(
        0.02,
        1.02,
        "Embedding-preservation ablation",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.8,
        fontweight="semibold",
    )
    ax.text(
        0.98,
        1.02,
        "Domain-aligned baseline Recall@10 = 0.969",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color="#555555",
    )

    table = ax.table(
        cellText=[
            ["Without preservation regularization", "1.185", "0.915"],
            ["With preservation regularization", "1.143", "0.969"],
        ],
        colLabels=["Variant", "Validation loss", "Hidden Recall@10"],
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.56, 0.20, 0.20],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.7)
    table.scale(1.0, 1.55)

    for (row, column), cell in table.get_celld().items():
        cell.set_linewidth(0.55)
        cell.set_edgecolor("#C8C8C8")
        if row == 0:
            cell.set_facecolor("#F1F1F1")
            cell.set_text_props(fontweight="semibold")
        elif row == 2:
            cell.set_facecolor("#F7FBF7")
            cell.set_text_props(fontweight="semibold")
        else:
            cell.set_facecolor("white")

    table[(1, 0)].set_text_props(ha="left")
    table[(2, 0)].set_text_props(ha="left")


def generate_expert_specialization_figure() -> None:
    method_specs, label_metrics, evidence_metrics, negative_metrics = stage2_data()
    fig = plt.figure(figsize=(7.18, 8.05))
    grid = fig.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 0.42), hspace=0.16, wspace=0.16)

    draw_expert_metric(
        fig.add_subplot(grid[0, 0]),
        label_metrics["ontology_credit"],
        method_specs,
        title="Ontology-aware label credit",
        panel="a",
        show_method_labels=True,
    )
    draw_expert_metric(
        fig.add_subplot(grid[0, 1]),
        label_metrics["macro_f1"],
        method_specs,
        title="Label macro-F1",
        panel="b",
        show_method_labels=False,
    )
    draw_expert_metric(
        fig.add_subplot(grid[1, 0]),
        evidence_metrics["f1"],
        method_specs,
        title="Evidence-gene F1",
        panel="c",
        show_method_labels=True,
    )
    draw_expert_metric(
        fig.add_subplot(grid[1, 1]),
        negative_metrics["f1"],
        method_specs,
        title="Negative-marker F1",
        panel="d",
        show_method_labels=False,
    )
    draw_preservation_ablation(fig, grid[2, :])

    category_handles = [
        Patch(facecolor="#111111", label="BioAgent"),
        Patch(facecolor="#729A70", label="Specialized"),
        Patch(facecolor="#4C8BB8", label="Open-source"),
        Patch(facecolor="#CC7430", label="Closed-source"),
    ]
    fig.legend(
        handles=category_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        handlelength=1.3,
        columnspacing=1.6,
    )
    fig.subplots_adjust(left=0.12, right=0.985, top=0.92, bottom=0.055)
    save_figure(fig, EXPERT_STEM)


def main() -> None:
    set_figure_style()
    generate_domain_alignment_figure()
    generate_expert_specialization_figure()
    print(f"Wrote {DOMAIN_STEM.with_suffix('.pdf')}")
    print(f"Wrote {EXPERT_STEM.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()