# Dataset Pipeline

Standalone CELLxGENE-backed dataset pipeline for stage-2 task-adapter training.

This folder is self-contained. It owns its own:

- dataset catalog
- CELLxGENE Census search and download logic
- metadata audit step
- normalization and task-example builders
- claim verifier for evidence, exclusions, and context
- stage-2 export path

## Purpose

The pipeline builds single-cell question-answer datasets for stage-2 supervised fine-tuning on top of the stage-1 domain adapter. The initial tasks are:

- cell annotation with rationale
- tissue identification with rationale
- cell state with rationale when auditable state metadata exists
- differential diagnosis with source-backed exclusions

## Hallucination policy

The pipeline treats `EVIDENCE`, `EXCLUSIONS`, and `CONTEXT` as claims, not prose.

- Evidence claims must come from approved marker sets and also be observed in the cell.
- Exclusion claims must come from plausible candidate labels and carry marker-grounded rejection reasons.
- Context claims must come from whitelisted metadata keys.
- Rows that fail verification are dropped.

The first marker-grounded implementation is wired into cell-annotation and differential-diagnosis builders.
Tissue and state builders remain conservative and should be expanded only after dedicated marker coverage is added.

This design prefers conservative correctness over aggressive dataset size.

## Folder layout

The package creates and uses local directories under this folder:

- `data/raw/`
- `data/normalized/`
- `data/exports/`
- `data/stage3_exports/`
- `data/gene_only_benchmarks/`
- `data/reports/`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run package commands from the repository root. Check the H5AD dependencies before downloading data:

```bash
python -m dataset_pipeline doctor
```

## Catalog

Approved dataset specs live in:

- `approved_datasets.json`

You can search CELLxGENE first, then add or refine approved dataset specs locally.
The starter catalog now includes multiple organ-specific Tabula Sapiens entries for cell-type annotation, in addition to the original audit seed.

## Commands

Run from this folder.

Search candidate datasets:

```bash
python -m dataset_pipeline search-census --collection-name-contains "Tabula Sapiens"
```

Check interpreter and H5AD runtime health before download, audit, build, or run-all:

```bash
python -m dataset_pipeline doctor
```

List approved datasets from the local catalog:

```bash
python -m dataset_pipeline list-approved
```

Download approved datasets:

```bash
python -m dataset_pipeline download
```

Audit downloaded metadata:

```bash
python -m dataset_pipeline audit
```

Build verified task datasets and export stage-2 JSONL:

```bash
python -m dataset_pipeline build
```

Build grouped stage-3 orchestration benchmark datasets from the existing grouped stage-2 exports:

```bash
python -m dataset_pipeline build-stage3
```

Build a gene-only benchmark dataset with GPT-4o-mini-generated natural questions and grounded answers:

```bash
python -m dataset_pipeline build-gene-benchmark -- --k 2 --splits test --levels cell cluster
```

You can also call the thin wrapper directly:

```bash
python dataset_pipeline/build_gene_only_benchmark_dataset.py --k 2 --splits test --levels cell cluster
```

Audit an existing gene-only benchmark to find answer gene mentions or identity labels that are not present in hidden grounding:

```bash
python dataset_pipeline/audit_gene_only_benchmark.py \
	--hidden-path data/gene_only_benchmarks/gpt4o_mini_benchmark_v1/benchmark_hidden.jsonl
```

For a smaller smoke test on one tissue export, cap each task family and split independently:

```bash
python -m dataset_pipeline build-stage3 \
	--dataset tabula_sapiens_bladder_cell_annotation \
	--limit-per-split 25
```

Create grouped train, val, and test splits that hold out full donor or specimen-like groups instead of individual cells:

```bash
python dataset_pipeline/create_grouped_dataset_splits.py \
	--exports-root dataset_pipeline/data/exports
```

This writes separate grouped outputs such as `*_grouped_train.jsonl`, plus `grouped_split_manifest.json` and `grouped_split_summary.json`, without overwriting the existing cell-level split files.

Run the full standalone flow:

```bash
python -m dataset_pipeline run-all
```

If `doctor` reports a broken H5AD runtime, fix the environment before running `download`, `audit`, `build`, or `run-all`.

## Output contract

The builder emits fixed-field answers under `structured_claim_v2`:

```text
LABEL: <label>
POSITIVE_MARKERS: <comma-separated grounded positive markers observed in the input gene set or none>
NEGATIVE_MARKERS: <comma-separated curated negative markers for the grounded label or none>
EVIDENCE: <comma-separated observed supporting genes or none>
EXCLUSIONS: <comma-separated verified exclusions or none>
CONTEXT: <semicolon-separated metadata claims or none>
CONFIDENCE: <high|medium|low>
FINAL: <label>
```

`EVIDENCE` remains the observed support in the current cell. `POSITIVE_MARKERS` is grounded to the observed intersection between the curated marker profile and the input gene set, while `NEGATIVE_MARKERS` remains the curated negative-marker panel used for grounded cell-type tasks.

## Export products

For each approved dataset, the pipeline writes:

- verified intermediate examples in `data/normalized/`
- task-specific stage-2 JSONL files in `data/exports/<dataset_name>/`
- grouped stage-3 benchmark JSONL files in `data/stage3_exports/`
- verification reports in `data/reports/`

The stage-2 export rows are compatible with the completion-style task adapter under:

- `../stage2_task_adapter/`

The stage-3 generator is downstream-only: it reads the grouped stage-2 exports, derives direct, cross-expert, and abstention-style benchmark rows, and writes per-family grouped train, val, and test files plus merged manifests.

The gene-only benchmark creator lives under `gene_only_benchmark_creator/`. It reads the verified stage-2 exports, keeps the public benchmark input limited to gene lists, uses hidden grounding fields from the export rows to prompt GPT-4o-mini, and writes:

- `benchmark.jsonl` with public benchmark rows
- `benchmark_hidden.jsonl` with hidden grounding for audit/evaluation
- `manifest.json` with counts and generation settings

## Wrapper scripts

Thin entry points are included for convenience:

- `download_cellxgene_datasets.py`
- `build_task_datasets.py`
- `verify_task_datasets.py`

They call the same CLI implementation in `cli.py`.