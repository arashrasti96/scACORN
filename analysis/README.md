# Paper analysis

This directory contains the metric aggregation, figure generation, and panel export scripts used for the scACORN paper. Generated tables, figures, and embedding artifacts are intentionally ignored by Git.

The scripts expect outputs from the public training and evaluation entry points in the repository:

- `stage1_domain_adapter/outputs/` and `stage1_domain_adapter/benchmark_suite/results/`
- `stage2_task_adapter/outputs/`
- `stage3_ace_orchestrator/runs/`
- `dataset_pipeline/data/exports/`

Run scripts from the repository root. Typical commands are:

```bash
python analysis/scripts/generate_stage1_results.py
python analysis/scripts/generate_stage2_results.py
python analysis/scripts/generate_stage3_playbook_learning_figure.py
python analysis/scripts/generate_results_composite_figures.py
```

Stage 1 UMAP exports are generated separately because model-backed methods can require substantial GPU memory:

```bash
python analysis/scripts/export_stage1_plot_embeddings.py \
  --split-family grouped \
  --datasets tabula_sapiens_prostate_cell_annotation tabula_sapiens_stomach_cell_annotation \
  --methods rank_pca tfidf_svd c2s_base c2s_lora_features c2s_lora_projections
```

The Stage 3 learning-curve script expects the run name encoded in the script. Change `RUN_DIR` when analyzing a differently named run. The Claude Sonnet 4.5 trajectory reported in the paper is stored directly in that plotting script; the GPT-5.4-mini trajectory is recomputed from the local run output with `assess_performance_over_time.py`.