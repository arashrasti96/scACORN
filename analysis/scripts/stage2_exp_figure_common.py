from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT_DIR = ROOT / "analysis"
FIGURES_DIR = MANUSCRIPT_DIR / "figures" / "exp_result_stage2"

PNG_DPI = 450

plt.rcParams.update(
    {
        "font.size": 14,
        "font.weight": "semibold",
        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "axes.labelsize": 16,
        "axes.labelweight": "bold",
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


@dataclass(frozen=True)
class MethodSpec:
    key: str
    display_name: str
    category: str
    color: str


METHOD_SPECS: tuple[MethodSpec, ...] = (
    MethodSpec("scslm", "scSLM", "Ours", "#111111"),
    MethodSpec("cell-o1", "Cell-o1", "Specialized models", "#4B7F52"),
    MethodSpec("c2s-scale-1b", "C2S Scale 1B", "Specialized models", "#7AA36F"),
    MethodSpec("c2s-pythia-410m", "C2S Pythia 410M", "Specialized models", "#A4BE8C"),
    MethodSpec("mistral-small-3.2-24b-instruct-2506", "mistral24B", "Open-source", "#2E6F95"),
    MethodSpec("qwen3-14b", "Qwen3-14B", "Open-source", "#4C8BB8"),
    MethodSpec("qwen2.5-14b", "Qwen2.5-14B", "Open-source", "#6FA8C9"),
    MethodSpec("gemma-2-9b-it", "Gemma-2-9b-it", "Open-source", "#8BC1D6"),
    MethodSpec("llama-3.1-8b-instruct", "llama3-8B", "Open-source", "#9ED8E6"),
    MethodSpec("qwen3-8b", "Qwen3-8B", "Open-source", "#6F9EAF"),
    MethodSpec("qwen2.5-7b", "Qwen2.5-7B", "Open-source", "#4E7488"),
    MethodSpec("gpt-4o-mini", "gpt-4o-mini", "Close-source", "#C65D21"),
    MethodSpec("gpt-5", "gpt-5", "Close-source", "#D18A39"),
)

METHOD_INDEX = {spec.key: spec for spec in METHOD_SPECS}


def method_keys() -> list[str]:
    return [spec.key for spec in METHOD_SPECS]


def grouped_positions(gap: float = 0.45) -> tuple[list[float], dict[str, float]]:
    positions: list[float] = []
    category_centers: dict[str, float] = {}
    current = 0.0

    grouped: dict[str, list[float]] = {}
    for spec in METHOD_SPECS:
        grouped.setdefault(spec.category, [])
        grouped[spec.category].append(current)
        positions.append(current)
        current += 1.0
        next_index = len(positions)
        if next_index < len(METHOD_SPECS) and METHOD_SPECS[next_index].category != spec.category:
            current += gap

    for category, values in grouped.items():
        category_centers[category] = sum(values) / len(values)

    return positions, category_centers


def category_boundaries(positions: Sequence[float]) -> list[float]:
    boundaries: list[float] = []
    for index, spec in enumerate(METHOD_SPECS[:-1]):
        next_spec = METHOD_SPECS[index + 1]
        if spec.category != next_spec.category:
            boundaries.append((positions[index] + positions[index + 1]) / 2)
    return boundaries


def compute_y_max(values_by_method: dict[str, float]) -> float:
    max_value = max(values_by_method.values())
    if max_value <= 0.10:
        return 0.12
    if max_value <= 0.20:
        return 0.20
    if max_value <= 0.35:
        return 0.40
    if max_value <= 0.55:
        return 0.60
    if max_value <= 0.75:
        return 0.80
    if max_value <= 0.90:
        return 0.95
    return 1.00


def compute_tick_step(y_max: float) -> float:
    if y_max <= 0.20:
        return 0.05
    if y_max <= 0.60:
        return 0.10
    return 0.20


def apply_bar_style(ax: plt.Axes, *, ylabel: str, title: str, lower_is_better: bool = False) -> None:
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_category_headers(ax: plt.Axes, centers: dict[str, float], y_offset: float = 0.965) -> None:
    for category, center in centers.items():
        if category == "Ours":
            continue
        ax.text(
            center,
            y_offset,
            HEADER_LABELS.get(category, category),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.2},
        )


def add_group_separators(ax: plt.Axes, positions: Sequence[float]) -> None:
    for x in category_boundaries(positions):
        ax.axvline(x=x, color="#8c8c8c", linewidth=1.2, linestyle=(0, (1.0, 3.0)), alpha=0.95, zorder=0)


def set_method_axis(ax: plt.Axes, positions: Sequence[float], *, show_labels: bool = True) -> None:
    labels = [spec.display_name for spec in METHOD_SPECS]
    ax.set_xticks(list(positions), labels if show_labels else [])
    if show_labels:
        ax.tick_params(axis="x", rotation=56, labelsize=13)
        for label in ax.get_xticklabels():
            label.set_ha("right")
            label.set_fontweight("bold")
    else:
        ax.tick_params(axis="x", length=0)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")


def draw_bars(
    ax: plt.Axes,
    values_by_method: dict[str, float],
    positions: Sequence[float],
    *,
    y_max: float = 1.0,
    annotate: bool = True,
    lower_is_better: bool = False,
) -> None:
    for spec, x in zip(METHOD_SPECS, positions):
        value = values_by_method[spec.key]
        ax.bar(x, value, width=0.78, color=spec.color)
        if annotate:
            offset = max(y_max * 0.018, 0.012)
            if spec.category == "Ours":
                offset += max(y_max * 0.035, 0.02)
            ax.text(
                x,
                min(value + offset, y_max - (0.02 if y_max > 0.2 else 0.006)),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                rotation=90,
            )
    ax.set_ylim(0.0, y_max)
    ax.margins(x=0.02)


def export_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES_DIR / f"{stem}.pdf"
    png_path = FIGURES_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=PNG_DPI, bbox_inches="tight")
    return pdf_path, png_path


def validate_metric_map(values_by_method: dict[str, float]) -> None:
    expected = set(method_keys())
    actual = set(values_by_method)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"metric map mismatch: missing={missing}, extra={extra}")


def validate_family(metric_maps: Iterable[dict[str, float]]) -> None:
    for metric_map in metric_maps:
        validate_metric_map(metric_map)


LABEL_METRIC_ORDER: tuple[str, ...] = (
    "ontology_credit",
    "macro_f1",
)

SINGLE_METRIC_ORDER: tuple[str, ...] = (
    "precision",
    "recall",
    "f1",
    "hallucination",
)

METRIC_TITLES = {
    "canonical_accuracy": "Canonical accuracy",
    "exact_accuracy": "Exact accuracy",
    "ontology_credit": "Ontology credit",
    "macro_f1": "Macro F1",
    "weighted_f1": "Weighted F1",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "hallucination": "Hallucination rate",
}

FAMILY_LABELS = {
    "label": "Label",
    "evidence": "Evidence gene",
    "negative": "Negative marker",
}

HEADER_LABELS = {
    "Ours": "Ours",
    "Specialized models": "Specialized models",
    "Open-source": "Open-source",
    "Close-source": "Close-source",
}

LABEL_METRICS: dict[str, dict[str, float]] = {
    "canonical_accuracy": {
        "scslm": 0.95,
        "gpt-4o-mini": 0.54,
        "gpt-5": 0.45,
        "mistral-small-3.2-24b-instruct-2506": 0.76,
        "qwen3-14b": 0.54,
        "qwen2.5-14b": 0.49,
        "gemma-2-9b-it": 0.41,
        "llama-3.1-8b-instruct": 0.40,
        "qwen3-8b": 0.40,
        "qwen2.5-7b": 0.38,
        "cell-o1": 0.70,
        "c2s-scale-1b": 0.06,
        "c2s-pythia-410m": 0.01,
    },
    "exact_accuracy": {
        "scslm": 0.90,
        "gpt-4o-mini": 0.47,
        "gpt-5": 0.42,
        "mistral-small-3.2-24b-instruct-2506": 0.66,
        "qwen3-14b": 0.45,
        "qwen2.5-14b": 0.40,
        "gemma-2-9b-it": 0.38,
        "llama-3.1-8b-instruct": 0.34,
        "qwen3-8b": 0.32,
        "qwen2.5-7b": 0.25,
        "cell-o1": 0.59,
        "c2s-scale-1b": 0.03,
        "c2s-pythia-410m": 0.01,
    },
    "ontology_credit": {
        "scslm": 0.95,
        "gpt-4o-mini": 0.59,
        "gpt-5": 0.45,
        "mistral-small-3.2-24b-instruct-2506": 0.78,
        "qwen3-14b": 0.56,
        "qwen2.5-14b": 0.53,
        "gemma-2-9b-it": 0.45,
        "llama-3.1-8b-instruct": 0.45,
        "qwen3-8b": 0.42,
        "qwen2.5-7b": 0.41,
        "cell-o1": 0.71,
        "c2s-scale-1b": 0.06,
        "c2s-pythia-410m": 0.01,
    },
    "macro_f1": {
        "scslm": 0.73,
        "gpt-4o-mini": 0.35,
        "gpt-5": 0.40,
        "mistral-small-3.2-24b-instruct-2506": 0.41,
        "qwen3-14b": 0.33,
        "qwen2.5-14b": 0.19,
        "gemma-2-9b-it": 0.21,
        "llama-3.1-8b-instruct": 0.17,
        "qwen3-8b": 0.21,
        "qwen2.5-7b": 0.12,
        "cell-o1": 0.41,
        "c2s-scale-1b": 0.03,
        "c2s-pythia-410m": 0.00,
    },
    "weighted_f1": {
        "scslm": 0.96,
        "gpt-4o-mini": 0.56,
        "gpt-5": 0.53,
        "mistral-small-3.2-24b-instruct-2506": 0.78,
        "qwen3-14b": 0.54,
        "qwen2.5-14b": 0.47,
        "gemma-2-9b-it": 0.42,
        "llama-3.1-8b-instruct": 0.40,
        "qwen3-8b": 0.39,
        "qwen2.5-7b": 0.28,
        "cell-o1": 0.73,
        "c2s-scale-1b": 0.04,
        "c2s-pythia-410m": 0.02,
    },
}

EVIDENCE_METRICS: dict[str, dict[str, float]] = {
    "precision": {
        "scslm": 0.82,
        "gpt-4o-mini": 0.34,
        "gpt-5": 0.48,
        "mistral-small-3.2-24b-instruct-2506": 0.41,
        "qwen3-14b": 0.30,
        "qwen2.5-14b": 0.16,
        "gemma-2-9b-it": 0.19,
        "llama-3.1-8b-instruct": 0.18,
        "qwen3-8b": 0.15,
        "qwen2.5-7b": 0.05,
        "cell-o1": 0.38,
        "c2s-scale-1b": 0.00,
        "c2s-pythia-410m": 0.00,
    },
    "recall": {
        "scslm": 0.82,
        "gpt-4o-mini": 0.47,
        "gpt-5": 0.33,
        "mistral-small-3.2-24b-instruct-2506": 0.54,
        "qwen3-14b": 0.41,
        "qwen2.5-14b": 0.37,
        "gemma-2-9b-it": 0.25,
        "llama-3.1-8b-instruct": 0.24,
        "qwen3-8b": 0.26,
        "qwen2.5-7b": 0.32,
        "cell-o1": 0.58,
        "c2s-scale-1b": 0.00,
        "c2s-pythia-410m": 0.00,
    },
    "f1": {
        "scslm": 0.82,
        "gpt-4o-mini": 0.40,
        "gpt-5": 0.39,
        "mistral-small-3.2-24b-instruct-2506": 0.46,
        "qwen3-14b": 0.35,
        "qwen2.5-14b": 0.22,
        "gemma-2-9b-it": 0.22,
        "llama-3.1-8b-instruct": 0.21,
        "qwen3-8b": 0.19,
        "qwen2.5-7b": 0.09,
        "cell-o1": 0.46,
        "c2s-scale-1b": 0.00,
        "c2s-pythia-410m": 0.00,
    },
    "hallucination": {
        "scslm": 0.14,
        "gpt-4o-mini": 0.04,
        "gpt-5": 0.06,
        "mistral-small-3.2-24b-instruct-2506": 0.02,
        "qwen3-14b": 0.02,
        "qwen2.5-14b": 0.03,
        "gemma-2-9b-it": 0.11,
        "llama-3.1-8b-instruct": 0.10,
        "qwen3-8b": 0.03,
        "qwen2.5-7b": 0.03,
        "cell-o1": 0.03,
        "c2s-scale-1b": 0.00,
        "c2s-pythia-410m": 0.00,
    },
}

NEGATIVE_METRICS: dict[str, dict[str, float]] = {
    "precision": {
        "scslm": 0.96,
        "gpt-4o-mini": 0.01,
        "gpt-5": 0.27,
        "mistral-small-3.2-24b-instruct-2506": 0.01,
        "qwen3-14b": 0.01,
        "qwen2.5-14b": 0.00,
        "gemma-2-9b-it": 0.00,
        "llama-3.1-8b-instruct": 0.00,
        "qwen3-8b": 0.01,
        "qwen2.5-7b": 0.00,
        "cell-o1": 0.03,
        "c2s-scale-1b": 0.00,
        "c2s-pythia-410m": 0.00,
    },
    "recall": {
        "scslm": 0.94,
        "gpt-4o-mini": 0.00,
        "gpt-5": 0.14,
        "mistral-small-3.2-24b-instruct-2506": 0.01,
        "qwen3-14b": 0.01,
        "qwen2.5-14b": 0.01,
        "gemma-2-9b-it": 0.00,
        "llama-3.1-8b-instruct": 0.00,
        "qwen3-8b": 0.00,
        "qwen2.5-7b": 0.00,
        "cell-o1": 0.02,
        "c2s-scale-1b": 0.00,
        "c2s-pythia-410m": 0.00,
    },
    "f1": {
        "scslm": 0.95,
        "gpt-4o-mini": 0.01,
        "gpt-5": 0.18,
        "mistral-small-3.2-24b-instruct-2506": 0.01,
        "qwen3-14b": 0.01,
        "qwen2.5-14b": 0.00,
        "gemma-2-9b-it": 0.00,
        "llama-3.1-8b-instruct": 0.00,
        "qwen3-8b": 0.01,
        "qwen2.5-7b": 0.00,
        "cell-o1": 0.03,
        "c2s-scale-1b": 0.00,
        "c2s-pythia-410m": 0.00,
    },
    "hallucination": {
        "scslm": 0.03,
        "gpt-4o-mini": 0.01,
        "gpt-5": 0.22,
        "mistral-small-3.2-24b-instruct-2506": 0.48,
        "qwen3-14b": 0.63,
        "qwen2.5-14b": 0.69,
        "gemma-2-9b-it": 0.09,
        "llama-3.1-8b-instruct": 0.28,
        "qwen3-8b": 0.88,
        "qwen2.5-7b": 0.49,
        "cell-o1": 0.46,
        "c2s-scale-1b": 0.00,
        "c2s-pythia-410m": 0.00,
    },
}

FAMILY_DATA = {
    "label": LABEL_METRICS,
    "evidence": EVIDENCE_METRICS,
    "negative": NEGATIVE_METRICS,
}


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.03, label, transform=ax.transAxes, fontsize=15, fontweight="bold")


def style_metric_axis(
    ax: plt.Axes,
    values_by_method: dict[str, float],
    positions: Sequence[float],
    *,
    title: str,
    ylabel: str,
    show_xlabels: bool,
    show_group_headers: bool,
    lower_is_better: bool = False,
) -> None:
    validate_metric_map(values_by_method)
    y_max = compute_y_max(values_by_method)
    apply_bar_style(ax, ylabel=ylabel, title=title, lower_is_better=lower_is_better)
    draw_bars(ax, values_by_method, positions, y_max=y_max, lower_is_better=lower_is_better)
    add_group_separators(ax, positions)
    set_method_axis(ax, positions, show_labels=show_xlabels)
    ax.set_ylim(0.0, y_max)
    step = compute_tick_step(y_max)
    tick_count = int(math.floor(y_max / step)) + 1
    ax.set_yticks([round(index * step, 2) for index in range(tick_count + 1) if index * step <= y_max + 1e-9])
    if show_group_headers:
        _, centers = grouped_positions()
        add_category_headers(ax, centers)


def render_label_metrics_figure(stem: str = "stage2_label_metrics_all_methods") -> tuple[Path, Path]:
    validate_family(LABEL_METRICS.values())
    positions, _ = grouped_positions()
    fig = plt.figure(figsize=(16.2, 6.8))
    grid = fig.add_gridspec(1, 2, wspace=0.16)

    axes = {
        "ontology_credit": fig.add_subplot(grid[0, 0]),
        "macro_f1": fig.add_subplot(grid[0, 1]),
    }

    label_order = {
        "ontology_credit": "A",
        "macro_f1": "B",
    }

    for metric_key in LABEL_METRIC_ORDER:
        ax = axes[metric_key]
        style_metric_axis(
            ax,
            LABEL_METRICS[metric_key],
            positions,
            title=METRIC_TITLES[metric_key],
            ylabel="Value",
            show_xlabels=True,
            show_group_headers=True,
        )
        panel_label(ax, label_order[metric_key])

    fig.subplots_adjust(left=0.055, right=0.995, top=0.92, bottom=0.28)
    return export_figure(fig, stem)


def render_single_label_metric_figure(metric_key: str, *, stem: str) -> tuple[Path, Path]:
    validate_metric_map(LABEL_METRICS[metric_key])
    positions, _ = grouped_positions()
    fig, ax = plt.subplots(figsize=(16.2, 6.2))
    style_metric_axis(
        ax,
        LABEL_METRICS[metric_key],
        positions,
        title=METRIC_TITLES[metric_key],
        ylabel="Value",
        show_xlabels=True,
        show_group_headers=True,
    )
    fig.subplots_adjust(left=0.055, right=0.995, top=0.90, bottom=0.28)
    return export_figure(fig, stem)


def render_single_metric_figure(
    family: str,
    metric_key: str,
    *,
    stem: str,
) -> tuple[Path, Path]:
    metric_data = FAMILY_DATA[family][metric_key]
    validate_metric_map(metric_data)
    positions, _ = grouped_positions()
    fig, ax = plt.subplots(figsize=(16.5, 6.3))
    style_metric_axis(
        ax,
        metric_data,
        positions,
        title=METRIC_TITLES[metric_key],
        ylabel="Value",
        show_xlabels=True,
        show_group_headers=True,
        lower_is_better=metric_key == "hallucination",
    )
    fig.subplots_adjust(left=0.055, right=0.995, top=0.90, bottom=0.31)
    return export_figure(fig, stem)