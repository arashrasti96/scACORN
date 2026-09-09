from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT_DIR = ROOT / "analysis"
PANEL_ROOT = MANUSCRIPT_DIR / "figures" / "results_panels"
DOMAIN_ROOT = PANEL_ROOT / "domain_alignment"
EXPERT_ROOT = PANEL_ROOT / "expert_specialization"
SCRIPT_DIR = MANUSCRIPT_DIR / "scripts"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_stage1_results as stage1
import stage2_exp_figure_common as stage2


PNG_DPI = 600
FULL_EMBEDDING_PNG_DPI = 900


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
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
            "savefig.facecolor": "white",
        }
    )


def save_panel(fig: plt.Figure, stem: Path, png_dpi: int = PNG_DPI) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(stem.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def render_embedding_legend_panel(
    ax: plt.Axes,
    labels: np.ndarray,
    colors: dict[str, tuple[float, float, float, float]],
    panel_label: str,
) -> None:
    ordered = sorted(set(labels.tolist()), key=lambda label: (-int(np.sum(labels == label)), label))
    display_labels = [stage1.shorten_cell_type_label(label) for label in ordered]
    n_rows = int(np.ceil(len(ordered) / 2))
    y_values = np.linspace(0.92, 0.10, n_rows)
    for index, (label, display_label) in enumerate(zip(ordered, display_labels)):
        column = index // n_rows
        row = index % n_rows
        x_base = 0.06 + column * 0.41
        y = y_values[row]
        ax.scatter([x_base], [y], s=60, color=colors[label], linewidths=0.0, transform=ax.transAxes, clip_on=False)
        ax.text(
            x_base + 0.045,
            y,
            display_label,
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontsize=10.5,
            fontweight="bold",
            color="#111111",
            clip_on=False,
        )
    ax.set_title("Cell-type legend", pad=7, fontweight="bold")
    ax.axis("off")
    stage1.add_panel_label(ax, panel_label)


def export_domain_metric_panels() -> None:
    summary_df = pd.read_csv(stage1.SUMMARY_CSV)
    shared_df = pd.read_csv(stage1.SHARED_GROUPED_CSV)
    method_order = summary_df["method_key"].tolist()
    display_names = dict(zip(summary_df["method_key"], summary_df["method_name"]))
    short_labels = stage1.method_short_labels(method_order)
    palette = stage1.method_palette(method_order)
    y_pos = np.arange(len(summary_df))
    colors = [palette[key] for key in method_order]
    labels = [short_labels[key] for key in method_order]

    fig_a, ax_a = plt.subplots(figsize=(7.2, 4.8))
    ax_a.barh(
        y_pos,
        summary_df["transfer_knn_macro_f1_mean"],
        xerr=summary_df["transfer_knn_macro_f1_std"],
        color=colors,
        alpha=0.94,
    )
    ax_a.set_yticks(y_pos, labels=labels)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Macro-F1")
    ax_a.set_title("Test classification")
    ax_a.set_xlim(0.0, max(1.0, summary_df["transfer_knn_macro_f1_mean"].max() + 0.08))
    ax_a.grid(axis="x", color="#d9d9d9", linewidth=0.8, alpha=0.7)
    ax_a.set_axisbelow(True)
    stage1.bold_tick_labels(ax_a)
    stage1.add_panel_label(ax_a, "a")
    save_panel(fig_a, DOMAIN_ROOT / "panel_a_transfer_macro_f1")

    fig_b, ax_b = plt.subplots(figsize=(7.2, 4.8))
    recall_lower = float(np.floor((summary_df["transfer_R5_mean"] - summary_df["transfer_R5_std"]).min() * 50.0) / 50.0)
    recall_lower = max(0.80, min(recall_lower - 0.01, 0.95))
    ax_b.barh(
        y_pos,
        summary_df["transfer_R5_mean"],
        xerr=summary_df["transfer_R5_std"],
        color=colors,
        alpha=0.94,
    )
    ax_b.set_yticks(y_pos, labels=labels)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Recall@5")
    ax_b.set_title("Test retrieval F1")
    ax_b.set_xlim(recall_lower, 1.005)
    ax_b.grid(axis="x", color="#d9d9d9", linewidth=0.8, alpha=0.7)
    ax_b.set_axisbelow(True)
    stage1.bold_tick_labels(ax_b)
    stage1.add_panel_label(ax_b, "b")
    save_panel(fig_b, DOMAIN_ROOT / "panel_b_transfer_recall_at_5")

    heatmap_df = (
        shared_df.pivot(index="tissue_name", columns="method_name", values="transfer_knn_macro_f1")
        .rename(columns={display_names[key]: short_labels[key] for key in method_order})
        .reindex(index=sorted(shared_df["tissue_name"].unique()), columns=labels)
    )
    fig_c, ax_c = plt.subplots(figsize=(8.6, 5.1))
    image = ax_c.imshow(
        heatmap_df.values,
        aspect="auto",
        cmap="YlGn",
        vmin=float(heatmap_df.values.min()),
        vmax=float(heatmap_df.values.max()),
    )
    ax_c.set_xticks(np.arange(len(heatmap_df.columns)), labels=heatmap_df.columns, rotation=20, ha="right")
    ax_c.set_yticks(np.arange(len(heatmap_df.index)), labels=heatmap_df.index)
    ax_c.set_title("Per-tissue test macro-F1")
    stage1.bold_tick_labels(ax_c)
    for row in range(heatmap_df.shape[0]):
        for column in range(heatmap_df.shape[1]):
            value = heatmap_df.iat[row, column]
            ax_c.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8.8, fontweight="bold")
    stage1.add_panel_label(ax_c, "c")
    cbar = fig_c.colorbar(image, ax=ax_c, fraction=0.024, pad=0.02)
    cbar.set_label("Macro-F1")
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")
    fig_c.subplots_adjust(left=0.19, right=0.92, top=0.90, bottom=0.18)
    save_panel(fig_c, DOMAIN_ROOT / "panel_c_tissue_heatmap")


def export_embedding_panels() -> None:
    from umap import UMAP

    panel_names = {
        "c2s_base": ("01_c2s_base", "A"),
        "c2s_lora_projections": ("02_c2s_lora_projections", "B"),
        "c2s_lora_features": ("03_c2s_lora_features", "C"),
        "tfidf_svd": ("04_tfidf_svd", "D"),
        "rank_pca": ("05_rank_pca", "E"),
    }
    for dataset, tissue_label in stage1.QUALITATIVE_TISSUES:
        tissue_dir = DOMAIN_ROOT / dataset.removeprefix("tabula_sapiens_").removesuffix("_cell_annotation")
        tissue_dir.mkdir(parents=True, exist_ok=True)

        combined_labels = []
        embeddings: dict[str, np.ndarray] = {}
        label_sets: dict[str, np.ndarray] = {}
        for method_key in stage1.QUALITATIVE_METHOD_ORDER:
            payload = stage1.load_exported_embedding_arrays(dataset, method_key, split_family="grouped")
            combined_embedding, combined_label = stage1.combine_exported_splits(payload)
            embeddings[method_key] = combined_embedding
            label_sets[method_key] = combined_label
            combined_labels.append(combined_label)

        all_labels = np.concatenate(combined_labels, axis=0)
        colors = stage1.label_palette(all_labels)
        short_labels = stage1.method_short_labels(stage1.QUALITATIVE_METHOD_ORDER)

        fig_full, axes_full = plt.subplots(2, 3, figsize=(15.8, 10.4), constrained_layout=True)
        flat_axes_full = axes_full.flatten()
        for ax, method_key, panel_label in zip(flat_axes_full[:5], stage1.QUALITATIVE_METHOD_ORDER, "ABCDE"):
            reducer = UMAP(n_components=2, random_state=stage1.UMAP_RANDOM_STATE, n_neighbors=28, min_dist=0.18)
            reduced = reducer.fit_transform(embeddings[method_key])
            stage1.plot_method_embedding_panel(
                ax,
                reduced,
                label_sets[method_key],
                short_labels[method_key],
                panel_label,
                colors,
            )
        stage1.plot_label_legend_panel(flat_axes_full[5], all_labels, colors, "F")
        save_panel(fig_full, tissue_dir / "full_embedding_figure", png_dpi=FULL_EMBEDDING_PNG_DPI)

        for method_key in stage1.QUALITATIVE_METHOD_ORDER:
            reducer = UMAP(n_components=2, random_state=stage1.UMAP_RANDOM_STATE, n_neighbors=28, min_dist=0.18)
            reduced = reducer.fit_transform(embeddings[method_key])
            stem_name, panel_label = panel_names[method_key]
            fig, ax = plt.subplots(figsize=(5.35, 5.15), constrained_layout=True)
            stage1.plot_method_embedding_panel(
                ax,
                reduced,
                label_sets[method_key],
                short_labels[method_key],
                panel_label,
                colors,
            )
            save_panel(fig, tissue_dir / stem_name)

        fig_legend, ax_legend = plt.subplots(figsize=(11.2, 6.4), constrained_layout=True)
        render_embedding_legend_panel(ax_legend, all_labels, colors, "F")
        save_panel(fig_legend, tissue_dir / "legend")


def export_expert_panels() -> None:
    positions, _ = stage2.grouped_positions()

    fig_a, ax_a = plt.subplots(figsize=(16.2, 6.2))
    stage2.style_metric_axis(
        ax_a,
        stage2.LABEL_METRICS["ontology_credit"],
        positions,
        title=stage2.METRIC_TITLES["ontology_credit"],
        ylabel="Value",
        show_xlabels=True,
        show_group_headers=True,
    )
    stage2.panel_label(ax_a, "A")
    fig_a.subplots_adjust(left=0.055, right=0.995, top=0.90, bottom=0.28)
    save_panel(fig_a, EXPERT_ROOT / "panel_a_ontology_credit")

    fig_b, ax_b = plt.subplots(figsize=(16.2, 6.2))
    stage2.style_metric_axis(
        ax_b,
        stage2.LABEL_METRICS["macro_f1"],
        positions,
        title=stage2.METRIC_TITLES["macro_f1"],
        ylabel="Value",
        show_xlabels=True,
        show_group_headers=True,
    )
    stage2.panel_label(ax_b, "B")
    fig_b.subplots_adjust(left=0.055, right=0.995, top=0.90, bottom=0.28)
    save_panel(fig_b, EXPERT_ROOT / "panel_b_macro_f1")

    fig_c, ax_c = plt.subplots(figsize=(16.5, 6.3))
    stage2.style_metric_axis(
        ax_c,
        stage2.EVIDENCE_METRICS["f1"],
        positions,
        title=stage2.METRIC_TITLES["f1"],
        ylabel="Value",
        show_xlabels=True,
        show_group_headers=True,
    )
    stage2.panel_label(ax_c, "C")
    fig_c.subplots_adjust(left=0.055, right=0.995, top=0.90, bottom=0.31)
    save_panel(fig_c, EXPERT_ROOT / "panel_c_evidence_f1")

    fig_d, ax_d = plt.subplots(figsize=(16.5, 6.3))
    stage2.style_metric_axis(
        ax_d,
        stage2.NEGATIVE_METRICS["f1"],
        positions,
        title=stage2.METRIC_TITLES["f1"],
        ylabel="Value",
        show_xlabels=True,
        show_group_headers=True,
    )
    stage2.panel_label(ax_d, "D")
    fig_d.subplots_adjust(left=0.055, right=0.995, top=0.90, bottom=0.31)
    save_panel(fig_d, EXPERT_ROOT / "panel_d_negative_marker_f1")

    variants = ["Without\npreservation", "With\npreservation"]
    val_loss = [1.185, 1.143]
    recall10 = [0.915, 0.969]
    colors = ["#B9BEC5", "#2E6F95"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"wspace": 0.28})
    fig.text(0.02, 0.98, "e", fontsize=14, fontweight="bold", va="top")
    fig.text(0.08, 0.98, "Embedding-preservation ablation", fontsize=12, fontweight="bold", va="top")

    metric_specs = (
        (axes[0], val_loss, "Validation loss", (1.10, 1.22), False),
        (axes[1], recall10, "Hidden Recall@10", (0.88, 0.99), True),
    )

    for ax, values, title, ylim, higher_is_better in metric_specs:
        x = np.arange(len(variants))
        bars = ax.bar(x, values, color=colors, width=0.62, zorder=2)
        ax.set_xticks(x, variants)
        ax.set_title(title, pad=6, fontsize=11.5, fontweight="bold")
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (ylim[1] - ylim[0]) * 0.02,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        if higher_is_better:
            ax.axhline(0.969, color="#666666", linewidth=1.0, linestyle=(0, (2, 2)), zorder=1)
            ax.text(
                0.03,
                0.969 + (ylim[1] - ylim[0]) * 0.045,
                "Domain-aligned baseline",
                ha="left",
                va="bottom",
                fontsize=8.5,
                color="#555555",
                transform=ax.get_yaxis_transform(),
            )

    fig.subplots_adjust(left=0.08, right=0.99, top=0.78, bottom=0.18)
    save_panel(fig, EXPERT_ROOT / "panel_e_preservation_ablation")


def main() -> None:
    set_style()
    DOMAIN_ROOT.mkdir(parents=True, exist_ok=True)
    EXPERT_ROOT.mkdir(parents=True, exist_ok=True)
    export_domain_metric_panels()
    export_embedding_panels()
    export_expert_panels()
    print(f"Wrote panel exports under {PANEL_ROOT}")


if __name__ == "__main__":
    main()