from __future__ import annotations

from stage2_exp_figure_common import render_single_metric_figure


def main() -> None:
    render_single_metric_figure(
        "evidence",
        "recall",
        stem="stage2_evidence_recall_all_methods",
    )


if __name__ == "__main__":
    main()