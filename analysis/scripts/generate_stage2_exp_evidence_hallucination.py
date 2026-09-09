from __future__ import annotations

from stage2_exp_figure_common import render_single_metric_figure


def main() -> None:
    render_single_metric_figure(
        "evidence",
        "hallucination",
        stem="stage2_evidence_hallucination_all_methods",
    )


if __name__ == "__main__":
    main()