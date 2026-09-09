from __future__ import annotations

from stage2_exp_figure_common import render_single_label_metric_figure


def main() -> None:
    render_single_label_metric_figure(
        "ontology_credit",
        stem="stage2_label_ontology_credit_all_methods",
    )
    render_single_label_metric_figure(
        "macro_f1",
        stem="stage2_label_macro_f1_all_methods",
    )


if __name__ == "__main__":
    main()