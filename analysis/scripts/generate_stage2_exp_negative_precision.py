from __future__ import annotations

from stage2_exp_figure_common import render_single_metric_figure


def main() -> None:
    render_single_metric_figure(
        "negative",
        "precision",
        stem="stage2_negative_precision_all_methods",
    )


if __name__ == "__main__":
    main()