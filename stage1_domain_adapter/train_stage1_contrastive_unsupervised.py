#!/usr/bin/env python
"""
Stage-1 unsupervised contrastive training entrypoint.

This script intentionally reuses the canonical implementation in
train_stage1_domain_lora.py so that contrastive training, validation retrieval,
checkpoint selection, and artifact saving stay identical across both entrypoints.

Example command:
deepspeed --num_gpus 2 \
    stage1_domain_adapter/train_stage1_contrastive_unsupervised.py \
  --run-name gemma_stage1_contrastive_unsupervised \
  --batch 64 \
  --grad-accum 1 \
  --epochs 20 \
  --lr 5e-4 \
  --warmup-steps 50 \
  --temperature 0.05 \
  --projection-dim 128 \
  --max-seq-len 512 \
  --top-genes 200 \
  --seed 42
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    target_script = Path(__file__).with_name("train_stage1_domain_lora.py")
    default_data_root = Path(__file__).resolve().parent.parent / "dataset_pipeline" / "data" / "exports"
    argv = sys.argv[1:]
    dataset_flags = {"--train-data", "--val-data", "--test-data", "--data-root"}
    if not any(flag in argv for flag in dataset_flags):
        argv = ["--data-root", str(default_data_root), "--data-glob", "*/cell_annotation_rationale.jsonl", *argv]
    sys.argv = [sys.argv[0], *argv]
    runpy.run_path(str(target_script), run_name="__main__")


if __name__ == "__main__":
    main()