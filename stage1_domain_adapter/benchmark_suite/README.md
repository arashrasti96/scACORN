# Stage-1 Embedding Benchmark Suite

This folder provides a fresh stage-1 benchmark runner for the per-dataset LoRA adapters under `stage1_domain_adapter/outputs`.

The suite benchmarks two split families separately:

- `standard`: the original cell-level `train/val/test` JSONL files
- `grouped`: the donor/specimen-style holdout files `*_grouped_train.jsonl`, `*_grouped_val.jsonl`, and `*_grouped_test.jsonl`

Each dataset is paired with its matching stage-1 adapter run by name. For example:

- dataset: `tabula_sapiens_ear_cell_annotation`
- adapter run: `gemma_stage1_tabula_sapiens_ear_cell_annotation`

## What the runner measures

For every dataset, split family, split, and method, the runner reports two views of embedding quality:

- `self_*`: within-split structure metrics on the current split alone
- `transfer_*`: train-reference label transfer metrics from the family-specific reference split to the query split

Implemented metrics:

- `self_R1`, `self_R5`, `self_R10`: within-split Recall@K under cosine kNN
- `self_NMI`, `self_ARI`, `self_Silhouette`: clustering quality on the split embeddings
- `transfer_R1`, `transfer_R5`, `transfer_R10`: whether the true label appears in the top-K train-reference neighbors
- `transfer_knn_acc`, `transfer_knn_macro_f1`: weighted kNN label transfer from the reference split

This keeps the grouped benchmark meaningful: the grouped `train -> val/test` transfer metrics expose how well embeddings generalize across held-out donor-like groups.

## Default methods

The default method set stays local and runnable from the current project assets:

- `rank_pca`
- `tfidf_svd`
- `c2s_base`
- `c2s_lora_features`
- `c2s_lora_projections`

Optional methods are implemented but not enabled by default:

- `sentence_transformer`
- `scgpt`
- `geneformer`

`scgpt` needs a local pretrained checkpoint directory. `geneformer` additionally needs the `geneformer` package installed.

## Run commands

Run these commands from the repository root after activating the project environment.

Lightweight smoke run on one dataset:

```bash
python stage1_domain_adapter/benchmark_suite/run_stage1_embedding_benchmark.py \
  --datasets tabula_sapiens_ear_cell_annotation \
  --methods rank_pca tfidf_svd \
  --split-families standard grouped \
  --max-cells-per-split 256
```

Full per-dataset stage-1 run with your adapter and the matched base model:

```bash
python stage1_domain_adapter/benchmark_suite/run_stage1_embedding_benchmark.py \
  --datasets tabula_sapiens_ear_cell_annotation \
  --methods rank_pca tfidf_svd c2s_base c2s_lora_features c2s_lora_projections \
  --split-families standard grouped \
  --batch-size 8 \
  --checkpoint-tag best
```

Optional scGPT run after placing a local checkpoint directory on disk:

```bash
python stage1_domain_adapter/benchmark_suite/run_stage1_embedding_benchmark.py \
  --datasets tabula_sapiens_ear_cell_annotation \
  --methods scgpt \
  --split-families standard grouped \
  --scgpt-model-dir /path/to/scgpt_whole_human
```

## Outputs

By default the runner writes to a timestamped directory under `benchmark_suite/results/`.

Artifacts include:

- `run_config.json`
- `benchmark_results.csv`
- `benchmark_results.json`
- per-dataset, per-family, per-method JSON summaries

## Notes

- The standard and grouped benchmarks are intentionally kept separate in the outputs. Do not average them together unless you explicitly want a mixed summary.
- The train split for each family is used as the transfer reference for that family only. There is no mixing between standard and grouped references.
- The default LoRA loader resolves `gemma_stage1_{dataset_name}` first, so each Tabula Sapiens dataset uses the adapter it was trained with.
