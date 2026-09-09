from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT_DIR = ROOT / "analysis"
FIGURES_DIR = MANUSCRIPT_DIR / "figures"
TABLES_DIR = MANUSCRIPT_DIR / "tables"
ARTIFACTS_DIR = MANUSCRIPT_DIR / "artifacts"
CONTRASTIVE_ROOT = ROOT
BENCHMARK_SUITE_SRC = CONTRASTIVE_ROOT / "stage1_domain_adapter" / "benchmark_suite"
DATA_EXPORTS_ROOT = CONTRASTIVE_ROOT / "dataset_pipeline" / "data" / "exports"
BENCHMARK_ROOT = (
    CONTRASTIVE_ROOT
    / "stage1_domain_adapter"
    / "benchmark_suite"
    / "results"
)
STAGE1_OUTPUTS_ROOT = BENCHMARK_ROOT.parent.parent / "outputs"
BASELINE_CSV = BENCHMARK_ROOT / "all_tabula_stage1" / "benchmark_results.csv"
RETRY_ROOT = BENCHMARK_ROOT / "all_tabula_stage1_lora_retry"

BASELINE_METHODS = {"rank_pca", "tfidf_svd", "c2s_base"}
LORA_METHODS = {"c2s_lora_features", "c2s_lora_projections"}
SPLIT_FAMILIES = ("standard", "grouped")
SPLITS = ("train", "val", "test")

MERGED_CSV = TABLES_DIR / "stage1_benchmark_merged.csv"
SHARED_GROUPED_CSV = TABLES_DIR / "stage1_grouped_test_shared.csv"
BASELINE_GROUPED_CSV = TABLES_DIR / "stage1_grouped_test_baseline12.csv"
SUMMARY_CSV = TABLES_DIR / "stage1_grouped_summary.csv"
MAIN_TABLE_TEX = TABLES_DIR / "stage1_main_table.tex"
MAIN_FIGURE_PDF = FIGURES_DIR / "stage1_grouped_benchmark.pdf"
MAIN_FIGURE_PNG = FIGURES_DIR / "stage1_grouped_benchmark.png"
EMBEDDING_FIGURE_PREFIX = FIGURES_DIR / "stage1_embedding_methods"
EMBEDDING_EXPORT_ROOT = ARTIFACTS_DIR / "stage1_embedding_exports"

QUALITATIVE_TISSUES = (
    ("tabula_sapiens_prostate_cell_annotation", "Prostate"),
    ("tabula_sapiens_stomach_cell_annotation", "Stomach"),
)
QUALITATIVE_METHOD_ORDER = (
    "c2s_base",
    "c2s_lora_projections",
    "c2s_lora_features",
    "tfidf_svd",
    "rank_pca",
)
UMAP_RANDOM_STATE = 42

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


def pretty_tissue_name(dataset: str) -> str:
    tissue = dataset.removeprefix("tabula_sapiens_").removesuffix("_cell_annotation")
    return tissue.replace("_", " ").title()


def method_palette(method_order: Iterable[str]) -> dict[str, str]:
    colors = {
        "tfidf_svd": "#0b6e4f",
        "rank_pca": "#3a7ca5",
        "c2s_lora_features": "#d17b0f",
        "c2s_lora_projections": "#9c3d54",
        "c2s_base": "#666666",
    }
    return {method: colors.get(method, "#444444") for method in method_order}


def method_short_labels(method_order: Iterable[str]) -> dict[str, str]:
    labels = {
        "tfidf_svd": "TF-IDF + SVD",
        "rank_pca": "Rank PCA",
        "c2s_lora_features": "scSLM features",
        "c2s_lora_projections": "scSLM projections",
        "c2s_base": "scSLM base",
    }
    return {method: labels.get(method, method) for method in method_order}


def bold_tick_labels(ax: plt.Axes) -> None:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def shorten_cell_type_label(label: str) -> str:
    replacements = {
        "CD4-positive, alpha-beta T cell": "CD4 T",
        "CD8-positive, alpha-beta T cell": "CD8 T",
        "mature NK T cell": "NKT",
        "natural killer cell": "NK",
        "regulatory T cell": "Treg",
        "luminal cell of prostate epithelium": "Luminal epith.",
        "basal cell of prostate epithelium": "Basal epith.",
        "endothelial cell of lymphatic vessel": "Lymphatic endo.",
        "smooth muscle cell": "Smooth muscle",
        "enteroendocrine cell": "Enteroendo.",
        "type II pneumocyte": "Type II pneumo.",
        "mucous cell": "Mucous",
        "glandular epithelial cell": "Glandular epith.",
        "secretory cell": "Secretory",
    }
    return replacements.get(label, label)


def ensure_benchmark_imports() -> None:
    for path in (BENCHMARK_SUITE_SRC, CONTRASTIVE_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def import_benchmark_module(module_name: str):
    ensure_benchmark_imports()
    return importlib.import_module(module_name)


def load_baseline_rows() -> pd.DataFrame:
    df = pd.read_csv(BASELINE_CSV)
    df = df[df["method_key"].isin(BASELINE_METHODS)].copy()
    df["source_suite"] = "all_tabula_stage1"
    return df


def parse_retry_json(json_path: Path) -> list[dict[str, object]]:
    payload = json.loads(json_path.read_text())
    method = payload["method"]
    rows: list[dict[str, object]] = []
    for split_name in SPLITS:
        split_payload = payload["splits"][split_name]
        self_metrics = split_payload["self_metrics"]
        transfer_metrics = split_payload["transfer_metrics"]
        rows.append(
            {
                "dataset": method["dataset_name"],
                "embed_seconds": split_payload["embed_seconds"],
                "error": None,
                "fit_seconds": payload.get("fit_seconds", 0.0),
                "method_key": method["method_key"],
                "method_name": method["display_name"],
                "n_labels": split_payload["n_labels"],
                "n_samples": split_payload["n_samples"],
                "reference_split": "train",
                "self_ARI": self_metrics["ARI"],
                "self_NMI": self_metrics["NMI"],
                "self_R1": self_metrics["R1"],
                "self_R10": self_metrics["R10"],
                "self_R5": self_metrics["R5"],
                "self_Silhouette": self_metrics["Silhouette"],
                "split": split_name,
                "split_family": json_path.parent.name,
                "status": "ok",
                "transfer_R1": transfer_metrics["R1"],
                "transfer_R10": transfer_metrics["R10"],
                "transfer_R5": transfer_metrics["R5"],
                "transfer_knn_acc": transfer_metrics["knn_acc"],
                "transfer_knn_macro_f1": transfer_metrics["knn_macro_f1"],
                "source_suite": "all_tabula_stage1_lora_retry",
            }
        )
    return rows


def load_lora_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method_key in sorted(LORA_METHODS):
        pattern = f"tabula_sapiens_*/**/{method_key}.json"
        for json_path in sorted(RETRY_ROOT.glob(pattern)):
            if json_path.parent.name not in SPLIT_FAMILIES:
                continue
            rows.extend(parse_retry_json(json_path))
    return pd.DataFrame(rows)


def build_merged_table() -> tuple[pd.DataFrame, list[str]]:
    baseline_df = load_baseline_rows()
    lora_df = load_lora_rows()
    shared_tissues = sorted(
        set(lora_df["dataset"].unique())
        & set(
            lora_df[
                (lora_df["method_key"] == "c2s_lora_features")
                & (lora_df["split_family"] == "grouped")
                & (lora_df["split"] == "test")
            ]["dataset"].unique()
        )
        & set(
            lora_df[
                (lora_df["method_key"] == "c2s_lora_projections")
                & (lora_df["split_family"] == "grouped")
                & (lora_df["split"] == "test")
            ]["dataset"].unique()
        )
    )
    merged_df = pd.concat([baseline_df, lora_df], ignore_index=True, sort=False)
    merged_df["tissue_name"] = merged_df["dataset"].map(pretty_tissue_name)
    merged_df = merged_df.sort_values(["dataset", "split_family", "split", "method_key"]).reset_index(drop=True)
    return merged_df, shared_tissues


def summarize_grouped_test(shared_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        shared_df.groupby(["method_key", "method_name"], as_index=False)
        .agg(
            n_tissues=("dataset", "nunique"),
            transfer_knn_macro_f1_mean=("transfer_knn_macro_f1", "mean"),
            transfer_knn_macro_f1_std=("transfer_knn_macro_f1", "std"),
            transfer_R5_mean=("transfer_R5", "mean"),
            transfer_R5_std=("transfer_R5", "std"),
        )
        .sort_values("transfer_knn_macro_f1_mean", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def write_main_table_tex(summary_df: pd.DataFrame, n_shared_tissues: int) -> None:
    short_labels = method_short_labels(summary_df["method_key"].tolist())
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{Stage 1 benchmark performance on the grouped test split across the {n_shared_tissues} tissues with complete five-method coverage. Values are mean $\\pm$ s.d. across tissues.}}",
        r"\label{tab:stage1_grouped_main}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Tissues & Transfer macro-F1 & Transfer Recall@5 \\",
        r"\midrule",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(
            f"{short_labels[row.method_key]} & {row.n_tissues} & "
            f"{row.transfer_knn_macro_f1_mean:.3f} $\\pm$ {row.transfer_knn_macro_f1_std:.3f} & "
            f"{row.transfer_R5_mean:.3f} $\\pm$ {row.transfer_R5_std:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    MAIN_TABLE_TEX.write_text("\n".join(lines) + "\n")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom")


def generate_main_figure(summary_df: pd.DataFrame, shared_df: pd.DataFrame) -> None:
    method_order = summary_df["method_key"].tolist()
    n_shared_tissues = shared_df["dataset"].nunique()
    display_names = dict(zip(summary_df["method_key"], summary_df["method_name"]))
    short_labels = method_short_labels(method_order)
    palette = method_palette(method_order)

    fig = plt.figure(figsize=(13.4, 9.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.35], hspace=0.18, wspace=0.15)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    y_pos = np.arange(len(summary_df))
    colors = [palette[key] for key in method_order]
    labels = [short_labels[key] for key in method_order]

    ax1.barh(
        y_pos,
        summary_df["transfer_knn_macro_f1_mean"],
        xerr=summary_df["transfer_knn_macro_f1_std"],
        color=colors,
        alpha=0.92,
    )
    ax1.set_yticks(y_pos, labels=labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("Transfer macro-F1")
    ax1.set_title("Test classification transfer")
    ax1.set_xlim(0.0, max(1.0, summary_df["transfer_knn_macro_f1_mean"].max() + 0.08))
    ax1.grid(axis="x", color="#d9d9d9", linewidth=0.8, alpha=0.7)
    ax1.set_axisbelow(True)
    bold_tick_labels(ax1)
    add_panel_label(ax1, "A")

    recall_lower = float(np.floor((summary_df["transfer_R5_mean"] - summary_df["transfer_R5_std"]).min() * 50.0) / 50.0)
    recall_lower = max(0.80, min(recall_lower - 0.01, 0.95))

    ax2.barh(
        y_pos,
        summary_df["transfer_R5_mean"],
        xerr=summary_df["transfer_R5_std"],
        color=colors,
        alpha=0.92,
    )
    ax2.set_yticks(y_pos, labels=labels)
    ax2.invert_yaxis()
    ax2.set_xlabel("Transfer Recall@5")
    ax2.set_title("Test retrieval transfer")
    ax2.set_xlim(recall_lower, 1.005)
    ax2.grid(axis="x", color="#d9d9d9", linewidth=0.8, alpha=0.7)
    ax2.set_axisbelow(True)
    bold_tick_labels(ax2)
    add_panel_label(ax2, "B")

    heatmap_df = (
        shared_df.pivot(index="tissue_name", columns="method_name", values="transfer_knn_macro_f1")
        .rename(columns={display_names[key]: short_labels[key] for key in method_order})
        .reindex(index=sorted(shared_df["tissue_name"].unique()), columns=labels)
    )
    im = ax3.imshow(heatmap_df.values, aspect="auto", cmap="YlGn", vmin=heatmap_df.values.min(), vmax=heatmap_df.values.max())
    ax3.set_xticks(np.arange(len(heatmap_df.columns)), labels=heatmap_df.columns, rotation=20, ha="right")
    ax3.set_yticks(np.arange(len(heatmap_df.index)), labels=heatmap_df.index)
    ax3.set_title("Per-tissue test macro-F1")
    bold_tick_labels(ax3)
    for i in range(heatmap_df.shape[0]):
        for j in range(heatmap_df.shape[1]):
            value = heatmap_df.iat[i, j]
            ax3.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8.5, fontweight="bold")
    add_panel_label(ax3, "C")
    cbar = fig.colorbar(im, ax=ax3, fraction=0.022, pad=0.015)
    cbar.set_label("Transfer macro-F1")
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")

    fig.savefig(MAIN_FIGURE_PDF, bbox_inches="tight")
    fig.savefig(MAIN_FIGURE_PNG, dpi=450, bbox_inches="tight")
    plt.close(fig)


def load_exported_embedding_arrays(dataset: str, method_key: str, split_family: str = "grouped") -> dict[str, np.ndarray]:
    path = EMBEDDING_EXPORT_ROOT / split_family / dataset / f"{method_key}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing exported embeddings for {dataset} / {method_key}. Expected: {path}\n"
            "Run manuscript/scripts/export_stage1_plot_embeddings.py on a GPU first."
        )
    with np.load(path, allow_pickle=True) as payload:
        return {key: payload[key] for key in payload.files}


def label_palette(labels: np.ndarray) -> dict[str, tuple[float, float, float, float]]:
    unique_labels = sorted(set(labels.tolist()), key=lambda label: (-int(np.sum(labels == label)), label))
    cmap = plt.get_cmap("tab20", max(len(unique_labels), 3))
    return {label: cmap(index) for index, label in enumerate(unique_labels)}


def style_embedding_axes(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#fbfbfb")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#d0d0d0")
        spine.set_linewidth(0.9)


def plot_method_embedding_panel(
    ax: plt.Axes,
    reduced: np.ndarray,
    labels: np.ndarray,
    title: str,
    panel_label: str,
    colors: dict[str, tuple[float, float, float, float]],
) -> None:
    for label in sorted(colors, key=lambda value: (-int(np.sum(labels == value)), value)):
        mask = labels == label
        ax.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            s=11,
            alpha=0.78,
            color=colors[label],
            linewidths=0.0,
            rasterized=True,
        )
    ax.margins(0.04)
    ax.set_title(title, pad=7, fontweight="bold")
    style_embedding_axes(ax)
    add_panel_label(ax, panel_label)


def plot_label_legend_panel(ax: plt.Axes, labels: np.ndarray, colors: dict[str, tuple[float, float, float, float]], panel_label: str) -> None:
    ordered = sorted(set(labels.tolist()), key=lambda label: (-int(np.sum(labels == label)), label))
    display_labels = [shorten_cell_type_label(label) for label in ordered]
    n_rows = int(np.ceil(len(ordered) / 2))
    y_values = np.linspace(0.92, 0.10, n_rows)
    for index, (label, display_label) in enumerate(zip(ordered, display_labels)):
        column = index // n_rows
        row = index % n_rows
        x_base = 0.06 + column * 0.48
        y = y_values[row]
        ax.scatter([x_base], [y], s=60, color=colors[label], linewidths=0.0, transform=ax.transAxes, clip_on=False)
        ax.text(
            x_base + 0.05,
            y,
            display_label,
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold",
            color="#111111",
        )
    ax.set_title("Cell-type legend", pad=7, fontweight="bold")
    ax.axis("off")
    add_panel_label(ax, panel_label)


def combine_exported_splits(payload: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    embedding_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    for split_name in payload["splits"].tolist():
        embedding_blocks.append(np.asarray(payload[f"{split_name}_embeddings"], dtype=np.float32))
        label_blocks.append(np.asarray(payload[f"{split_name}_labels"], dtype=object))
    return np.concatenate(embedding_blocks, axis=0), np.concatenate(label_blocks, axis=0)


def render_grouped_embedding_figure(dataset: str, tissue_label: str) -> None:
    from umap import UMAP

    combined_labels = []
    embeddings: dict[str, np.ndarray] = {}
    label_sets: dict[str, np.ndarray] = {}
    for method_key in QUALITATIVE_METHOD_ORDER:
        payload = load_exported_embedding_arrays(dataset, method_key, split_family="grouped")
        combined_embedding, combined_label = combine_exported_splits(payload)
        embeddings[method_key] = combined_embedding
        label_sets[method_key] = combined_label
        combined_labels.append(combined_label)

    all_labels = np.concatenate(combined_labels, axis=0)
    colors = label_palette(all_labels)

    fig, axes = plt.subplots(2, 3, figsize=(15.8, 10.4), constrained_layout=True)
    flat_axes = axes.flatten()
    short_labels = method_short_labels(QUALITATIVE_METHOD_ORDER)
    for ax, method_key, panel_label in zip(flat_axes[:5], QUALITATIVE_METHOD_ORDER, "ABCDE"):
        reducer = UMAP(n_components=2, random_state=UMAP_RANDOM_STATE, n_neighbors=28, min_dist=0.18)
        reduced = reducer.fit_transform(embeddings[method_key])
        plot_method_embedding_panel(ax, reduced, label_sets[method_key], short_labels[method_key], panel_label, colors)
    plot_label_legend_panel(flat_axes[5], all_labels, colors, "F")

    output_stem = EMBEDDING_FIGURE_PREFIX.with_name(f"stage1_embedding_methods_{dataset.removeprefix('tabula_sapiens_').removesuffix('_cell_annotation')}")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"Rendered grouped embedding figure for {tissue_label}: {output_stem.with_suffix('.pdf')}")


def generate_embedding_figures() -> None:
    for dataset, tissue_label in QUALITATIVE_TISSUES:
        render_grouped_embedding_figure(dataset, tissue_label)


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    merged_df, shared_tissues = build_merged_table()
    grouped_test_df = merged_df[(merged_df["split_family"] == "grouped") & (merged_df["split"] == "test")].copy()
    shared_grouped_df = grouped_test_df[grouped_test_df["dataset"].isin(shared_tissues)].copy()
    baseline12_df = grouped_test_df[grouped_test_df["method_key"].isin(BASELINE_METHODS)].copy()

    summary_df = summarize_grouped_test(shared_grouped_df)

    merged_df.to_csv(MERGED_CSV, index=False)
    shared_grouped_df.to_csv(SHARED_GROUPED_CSV, index=False)
    baseline12_df.to_csv(BASELINE_GROUPED_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    write_main_table_tex(summary_df, len(shared_tissues))
    generate_main_figure(summary_df, shared_grouped_df)
    generate_embedding_figures()

    print(f"Shared tissues ({len(shared_tissues)}):")
    for dataset in shared_tissues:
        print(f"- {dataset}")
    print("\nGrouped test summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()