from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT_DIR = ROOT / "analysis"
FIGURES_DIR = MANUSCRIPT_DIR / "figures"
ASSESS_SCRIPT = Path(__file__).with_name("assess_performance_over_time.py")
RUN_DIR = (
    ROOT
    / "stage3_ace_orchestrator"
    / "runs"
    / "stage3_playbook_gpt54mini_empty_full_20260602_seed17"
)
OUTPUT_PDF = FIGURES_DIR / "stage3_playbook_learning_figure.pdf"
OUTPUT_PNG = FIGURES_DIR / "stage3_playbook_learning_figure.png"
GROUNDING_OUTPUT_PDF = FIGURES_DIR / "stage3_playbook_grounding_figure.pdf"
GROUNDING_OUTPUT_PNG = FIGURES_DIR / "stage3_playbook_grounding_figure.png"
ONTOLOGY_OUTPUT_PDF = FIGURES_DIR / "stage3_playbook_ontology_figure.pdf"
ONTOLOGY_OUTPUT_PNG = FIGURES_DIR / "stage3_playbook_ontology_figure.png"
SUPPORT_EVIDENCE_OUTPUT_PDF = FIGURES_DIR / "stage3_playbook_support_evidence_coverage_figure.pdf"
SUPPORT_EVIDENCE_OUTPUT_PNG = FIGURES_DIR / "stage3_playbook_support_evidence_coverage_figure.png"
INITIAL_HALLUCINATION_RATE = 0.145
INITIAL_GROUNDING_SCORE = 1.0 - INITIAL_HALLUCINATION_RATE
INITIAL_ONTOLOGY_SCORE = 0.754
CLAUDE_SAMPLES = list(range(0, 241, 20))
CLAUDE_GROUNDING_SCORES = [
    0.866,
    0.834,
    0.876,
    0.911,
    0.885,
    0.922,
    0.905,
    0.931,
    0.879,
    0.9343,
    0.941,
    0.963,
    0.945,
]
CLAUDE_ONTOLOGY_SCORES = [
    0.752,
    0.732,
    0.814,
    0.791,
    0.859,
    0.871,
    0.843,
    0.865,
    0.867,
    0.853,
    0.846,
    0.862,
    0.859,
]

METRIC_NAME = "support_grounding_score"
ONTOLOGY_METRIC_NAME = "label_semantic_score"
TEST_COLOR = "#0F766E"
GOLD_COLOR = "#1D4ED8"
CLAUDE_COLOR = "#C2410C"
BASELINE_COLOR = "#6B7280"

plt.rcParams.update(
    {
        "font.size": 15,
        "font.weight": "semibold",
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "axes.labelsize": 17,
        "axes.labelweight": "bold",
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_assessor_module():
    spec = importlib.util.spec_from_file_location("assess_performance_over_time", ASSESS_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load assessor script from {ASSESS_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def support_evidence_coverage(module, record: dict[str, Any]) -> float:
    expected = module.normalized_set(module.extract_ground_truth(record).get("supporting_genes"))
    answer_fields = module.extract_answer_fields(record)
    freeform_answer = module.normalize_text(module.extract_freeform_answer(record))
    observed = set(module.normalized_set(answer_fields.get("supporting_genes")))

    for gene in expected:
        if gene in freeform_answer:
            observed.add(gene)

    if not expected:
        return 1.0
    return len(observed & expected) / len(expected)


def load_stage3_summaries(module):
    grouped, label_meta, *_ = module.load_results(RUN_DIR, "test")
    metric_names = [METRIC_NAME, ONTOLOGY_METRIC_NAME]
    summaries = module.summarize_checkpoints(grouped, label_meta, metric_names)
    complete_sample_count = max(summary["samples"] for summary in summaries)
    filtered_summaries = [
        summary
        for summary in summaries
        if summary["samples"] == complete_sample_count and summary["step"] <= 240
    ]

    coverage_by_label: dict[str, float] = {}
    for summary in filtered_summaries:
        records = grouped[summary["label"]]
        coverage_by_label[summary["label"]] = sum(support_evidence_coverage(module, record) for record in records) / len(records)
    return filtered_summaries, coverage_by_label


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#d9d9d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def render_static_figure(test_summaries: list[dict], support_evidence_by_label: dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)

    test_x = [0, *[summary["step"] for summary in test_summaries]]
    test_y = [INITIAL_GROUNDING_SCORE, *[summary["averages"][METRIC_NAME] for summary in test_summaries]]
    support_x = [summary["step"] for summary in test_summaries]
    support_y = [support_evidence_by_label[summary["label"]] for summary in test_summaries]

    ax.axhline(
        INITIAL_GROUNDING_SCORE,
        color=BASELINE_COLOR,
        linewidth=2.0,
        linestyle=(0, (4, 3)),
        label="Initial grounding baseline",
        zorder=1,
    )
    ax.plot(
        test_x,
        test_y,
        color=TEST_COLOR,
        linewidth=3.2,
        marker="o",
        markersize=7.0,
        label="Held-out test set",
        zorder=3,
    )
    ax.plot(
        support_x,
        support_y,
        color=GOLD_COLOR,
        linewidth=3.0,
        marker="s",
        markersize=6.6,
        label="Support-evidence coverage",
        zorder=2,
    )

    ax.set_xlabel("Training samples seen")
    ax.set_ylabel("Score")
    ax.set_xlim(-5, 245)
    ax.set_ylim(0.75, 1.01)
    ax.set_xticks([0, 40, 80, 120, 160, 200, 240])
    ax.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    style_axes(ax)

    ax.legend(loc="lower right", frameon=False)
    ax.text(
        0.02,
        0.07,
        "Initial hallucination = 14.5%",
        transform=ax.transAxes,
        fontsize=13.5,
        fontweight="bold",
        color=BASELINE_COLOR,
    )

    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=450, bbox_inches="tight")
    plt.close(fig)


def render_grounding_figure(test_summaries: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)

    test_x = [0, *[summary["step"] for summary in test_summaries]]
    test_y = [INITIAL_GROUNDING_SCORE, *[summary["averages"][METRIC_NAME] for summary in test_summaries]]

    ax.plot(
        test_x,
        test_y,
        color=TEST_COLOR,
        linewidth=3.6,
        marker="o",
        markersize=7.6,
        label="GPT-5.4-mini orchestrator",
        zorder=3,
    )
    ax.plot(
        CLAUDE_SAMPLES,
        CLAUDE_GROUNDING_SCORES,
        color=CLAUDE_COLOR,
        linewidth=3.6,
        marker="s",
        markersize=7.2,
        label="Claude Sonnet 4.5 orchestrator",
        zorder=3,
    )

    ax.set_xlabel("Training samples seen")
    ax.set_ylabel("Evidence alignment")
    ax.set_xlim(-5, 245)
    ax.set_ylim(0.82, 0.98)
    ax.set_xticks([0, 40, 80, 120, 160, 200, 240])
    ax.set_yticks([0.84, 0.88, 0.92, 0.96])
    style_axes(ax)

    ax.legend(loc="lower right", frameon=False, fontsize=12.5)

    fig.savefig(GROUNDING_OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(GROUNDING_OUTPUT_PNG, dpi=450, bbox_inches="tight")
    plt.close(fig)


def render_ontology_figure(test_summaries: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)

    test_x = [0, *[summary["step"] for summary in test_summaries]]
    test_y = [INITIAL_ONTOLOGY_SCORE, *[summary["averages"][ONTOLOGY_METRIC_NAME] for summary in test_summaries]]

    ax.plot(
        test_x,
        test_y,
        color=TEST_COLOR,
        linewidth=3.6,
        marker="o",
        markersize=7.6,
        label="GPT-5.4-mini orchestrator",
        zorder=3,
    )
    ax.plot(
        CLAUDE_SAMPLES,
        CLAUDE_ONTOLOGY_SCORES,
        color=CLAUDE_COLOR,
        linewidth=3.6,
        marker="s",
        markersize=7.2,
        label="Claude Sonnet 4.5 orchestrator",
        zorder=3,
    )

    ax.set_xlabel("Training samples seen")
    ax.set_ylabel("Ontology-aware label agreement")
    ax.set_xlim(-5, 245)
    ax.set_ylim(0.72, 0.89)
    ax.set_xticks([0, 40, 80, 120, 160, 200, 240])
    ax.set_yticks([0.72, 0.76, 0.80, 0.84, 0.88])
    style_axes(ax)

    ax.legend(loc="lower right", frameon=False, fontsize=12.5)

    fig.savefig(ONTOLOGY_OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(ONTOLOGY_OUTPUT_PNG, dpi=450, bbox_inches="tight")
    plt.close(fig)


def render_support_evidence_figure(test_summaries: list[dict], support_evidence_by_label: dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)

    support_x = [summary["step"] for summary in test_summaries]
    support_y = [support_evidence_by_label[summary["label"]] for summary in test_summaries]

    ax.plot(
        support_x,
        support_y,
        color=GOLD_COLOR,
        linewidth=3.6,
        marker="s",
        markersize=7.4,
        label="Held-out test set",
        zorder=3,
    )

    ax.set_xlabel("Training samples seen")
    ax.set_ylabel("Support-evidence coverage")
    ax.set_xlim(15, 245)
    ax.set_ylim(0.78, 0.87)
    ax.set_xticks([20, 40, 80, 120, 160, 200, 240])
    ax.set_yticks([0.78, 0.80, 0.82, 0.84, 0.86])
    style_axes(ax)

    ax.legend(loc="lower right", frameon=False)

    fig.savefig(SUPPORT_EVIDENCE_OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(SUPPORT_EVIDENCE_OUTPUT_PNG, dpi=450, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    module = load_assessor_module()
    test_summaries, support_evidence_by_label = load_stage3_summaries(module)
    render_static_figure(test_summaries, support_evidence_by_label)
    render_grounding_figure(test_summaries)
    render_ontology_figure(test_summaries)
    render_support_evidence_figure(test_summaries, support_evidence_by_label)


if __name__ == "__main__":
    main()