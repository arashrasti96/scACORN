# BioAgent

Code for **Agentic Context Engineering with Specialized Small Language Models for Single-Cell Reasoning**.

BioAgent builds reusable single-cell experts in three stages:

1. **Domain alignment** adapts a Cell2Sentence backbone with contrastive LoRA training.
2. **Expert specialization** trains task adapters while regularizing embedding geometry against the stage-1 expert.
3. **Textual orchestration** registers the experts as tools and optimizes an ACE playbook for routing and evidence synthesis.

## Repository layout

- `dataset_pipeline/`: CELLxGENE discovery, normalization, grounded task construction, grouped splits, and stage-3 benchmark generation.
- `stage1_domain_adapter/`: contrastive domain adaptation and embedding benchmarks.
- `stage2_task_adapter/`: task specialization, geometry-preserving training, inference, and held-out evaluation.
- `stage3_ace_orchestrator/`: expert registry, local inference runtime, benchmark runner, and playbook optimization.
- `ace/`: the exact ACE runtime used by the stage-3 experiments, vendored for reproducibility.
- `analysis/scripts/`: figure, table, and metric aggregation code used for the paper.
- `contrastive_learning.py`, `evaluate_baseline.py`, and `utils.py`: original contrastive baseline and shared utilities.

Training datasets, model checkpoints, run logs, and generated result payloads are intentionally excluded. They are generated locally by the included pipeline and entry points.

## Installation

Python 3.11 and CUDA-capable hardware are recommended. Create an environment and install the full research stack:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The default backbone is [`vandijklab/C2S-Scale-Gemma-2-2B`](https://huggingface.co/vandijklab/C2S-Scale-Gemma-2-2B). Scripts also accept a local model path through `--model-path` or `GEMMA_MODEL_PATH`.

Some dataset and stage-3 commands call hosted LLM APIs. Set credentials in the shell or in an untracked `.env` file based on `.env.example`. Never commit credentials.

## Data preparation

The approved public CELLxGENE datasets are declared in `dataset_pipeline/approved_datasets.json`.

```bash
python -m dataset_pipeline doctor
python -m dataset_pipeline list-approved
python -m dataset_pipeline download
python -m dataset_pipeline audit
python -m dataset_pipeline build
python dataset_pipeline/create_grouped_dataset_splits.py \
  --exports-root dataset_pipeline/data/exports
python -m dataset_pipeline build-stage3
```

The pipeline writes raw and derived artifacts below `dataset_pipeline/data/`; this directory is ignored by Git. See `dataset_pipeline/README.md` for schemas, verification rules, and additional commands.

## Stage 1: domain alignment

Train one contrastive adapter per dataset export:

```bash
python stage1_domain_adapter/train_stage1_one_adapter_per_dataset.py \
  --exports-root dataset_pipeline/data/exports \
  --output-root stage1_domain_adapter/outputs \
  --num-gpus 2 \
  -- \
  --epochs 20 \
  --batch 64 \
  --temperature 0.05 \
  --projection-dim 128 \
  --model-path vandijklab/C2S-Scale-Gemma-2-2B
```

For a single shared adapter, use `stage1_domain_adapter/train_stage1_domain_lora.py`. Embedding evaluations and classical baselines are under `stage1_domain_adapter/benchmark_suite/` and `stage1_domain_adapter/comparison/`.

## Stage 2: expert specialization

Train the geometry-preserving task adapter on a grouped dataset split:

```bash
deepspeed --num_gpus 2 \
  stage2_task_adapter/train_stage2_task_adapter_regularized_dataset.py \
  --train-data dataset_pipeline/data/exports/<dataset>/<task>_grouped_train.jsonl \
  --val-data dataset_pipeline/data/exports/<dataset>/<task>_grouped_val.jsonl \
  --dataset-type <dataset> \
  --exports-root dataset_pipeline/data/exports \
  --stage1-outputs-root stage1_domain_adapter/outputs \
  --output-dir stage2_task_adapter/outputs/<run> \
  --use-deepspeed \
  --embedding-anchor-weight 0.6 \
  --embedding-distribution-weight 0.6 \
  --embedding-relation-weight 0.6
```

`stage2_task_adapter/run_stage2_remaining_datasets.py` automates matched training and test evaluation across dataset directories. See `stage2_task_adapter/README.md` for the accepted JSONL schema and other variants.

## Stage 3: textual orchestration

Stage 3 discovers trained stage-2 experts from `stage2_task_adapter/outputs`, builds a tool registry, and uses the vendored ACE runtime to optimize a textual playbook.

Dry-run a short training sequence without API calls:

```bash
python stage3_ace_orchestrator/train_stage3_playbook.py \
  --samples dataset_pipeline/data/stage3_exports/<family>/grouped_train.jsonl \
  --max-samples 3 \
  --dry-run
```

Run a trained playbook on a benchmark:

```bash
python stage3_ace_orchestrator/run_stage3_benchmark.py --help
```

Model-provider credentials and model names are selected through the stage-3 command-line options and environment. The supplied JSON files capture the playbook configurations used during development; generated run payloads are excluded.

## Reproducing evaluations

The repository includes the evaluation implementations used for the three stages:

- Stage 1: `stage1_domain_adapter/benchmark_suite/run_stage1_embedding_benchmark.py`
- Stage 2: `stage2_task_adapter/evaluate_stage2_test_split.py`
- Stage 3: `stage3_ace_orchestrator/run_stage3_benchmark.py`

Paper figures and summary tables can be regenerated with the scripts under `analysis/scripts/` after the corresponding ignored run outputs have been produced. See `analysis/README.md` for the expected local artifact layout.

Exact numerical reproduction also requires the corresponding public dataset versions, generated splits, trained adapters, and provider model versions. These large or provider-dependent artifacts are not stored in Git.

## License

This repository is released under the MIT License. The C2S model weights are distributed separately under their own license; consult the linked Hugging Face model card before use.
