Stage-1 domain fitting with contrastive LoRA for Gemma lives here.

This stage trains on the train split only, evaluates retrieval on the validation split after each epoch, and keeps the best checkpoint according to validation projection Recall@10 by default.

Example:

```bash
deepspeed --num_gpus 2 train_stage1_domain_lora.py \
  --batch 64 \
  --grad-accum 1 \
  --epochs 20 \
  --lr 5e-4 \
  --warmup-steps 50 \
  --temperature 0.05 \
  --max-seq-len 512 \
  --projection-dim 128 \
  --top-genes 200 \
  --run-name gemma_stage1_domain_adapter
```

For the dataset-pipeline exports, use `train_stage1_contrastive_unsupervised.py`.
By default it discovers `../dataset_pipeline/data/exports/*/cell_annotation_rationale.jsonl`,
builds a stratified train/val/test split, and writes checkpoints and metrics under
the same `outputs/<run-name>/` layout.

If you want one stage-1 adapter per dataset instead of one shared adapter over all
exports, use `train_stage1_one_adapter_per_dataset.py`. It launches one separate
training job per dataset export directory and assigns each run its own run name
and output folder.

Outputs are written under `outputs/<run-name>/`:

- `metrics/`: per-evaluation JSON metric snapshots
- `checkpoints/best/`: best stage-1 checkpoint from validation retrieval
- `checkpoints/last/`: final checkpoint after the last epoch
- `artifacts/`: training history and summary files

Default stage-1 LoRA strategy:

- target modules: `q_proj,k_proj,v_proj,o_proj`
- target layers: upper half of Gemma blocks
- model selection metric: validation projection `Recall@10`