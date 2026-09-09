#!/usr/bin/env python
"""
Stage-1 domain fitting with SimCLR contrastive learning on Gemma 2B.

Key differences from the original contrastive_learning.py:
- trains only on the train split
- evaluates retrieval on the validation split after every epoch
- selects and saves the best checkpoint by validation retrieval metric
- persists per-evaluation metric snapshots and training summaries
- defaults to stage-1 LoRA scope: attention modules in the upper half of layers

Example command:
deepspeed --num_gpus 4 \
    stage1_domain_adapter/train_stage1_domain_lora.py \
  --run-name gemma_27B_stage1_domain_adapter_h100x2 \
  --batch 48 \
  --grad-accum 1 \
  --epochs 20 \
  --lr 5e-4 \
  --warmup-steps 50 \
  --temperature 0.05 \
  --projection-dim 128 \
  --max-seq-len 512 \
  --top-genes 200 \
  --eval-batch 8 \
  --lora-r 8 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --lora-target-modules q_proj,k_proj,v_proj,o_proj \
  --layer-strategy upper_half \
  --seed 42


"""

import argparse
import copy
import errno
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from utils import (  # noqa: E402
    ProjectionHead,
    build_c2s_prompt,
    extract_genes_from_question,
    get_hidden_states,
    print_recall_at_k,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1 contrastive domain fit with Gemma 2B")
    parser.add_argument("--train-data", type=str, default=None, help="Optional explicit train JSONL")
    parser.add_argument("--val-data", type=str, default=None, help="Optional explicit validation JSONL")
    parser.add_argument("--test-data", type=str, default=None, help="Optional explicit test JSONL")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Optional root directory of dataset_pipeline export files to discover and split.",
    )
    parser.add_argument(
        "--data-glob",
        type=str,
        default="*/cell_annotation_rationale.jsonl",
        help="Glob under --data-root used to discover export files.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio when splitting discovered export files")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio when splitting discovered export files")
    parser.add_argument("--label-key", type=str, default="cell_type", help="Metadata label key for export rows")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch", type=int, default=16, help="Per-GPU micro-batch size")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--temperature", type=float, default=0.07, help="NT-Xent temperature")
    parser.add_argument("--projection-dim", type=int, default=128, help="Projection head output dim")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Maximum prompt length")
    parser.add_argument("--top-genes", type=int, default=200, help="Genes to keep per sample")
    parser.add_argument("--eval-batch", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze LoRA/backbone and train projection only")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank injected by launcher")
    parser.add_argument("--run-name", type=str, default=None, help="Output run name; defaults to a timestamp")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--output-root", type=str, default=None, help="Base directory used to derive output-dir as <output-root>/<run-name>")
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.getenv("GEMMA_MODEL_PATH", "vandijklab/C2S-Scale-Gemma-2-27B"),
        help="Gemma base model path",
    )
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated LoRA target module names",
    )
    parser.add_argument(
        "--layer-strategy",
        type=str,
        choices=["upper_half", "last_third", "all"],
        default="upper_half",
        help="Default stage-1 layer coverage if explicit layer range is not provided",
    )
    parser.add_argument("--layer-start", type=int, default=None, help="Inclusive start layer override")
    parser.add_argument("--layer-end", type=int, default=None, help="Exclusive end layer override")
    parser.add_argument("--selection-k", type=int, default=10, help="Recall@K used for best-checkpoint selection")
    parser.add_argument(
        "--selection-space",
        type=str,
        choices=["projections", "features"],
        default="projections",
        help="Embedding space used for model selection",
    )
    parser.add_argument(
        "--startup-full-eval",
        action="store_true",
        help=(
            "Run the expensive train/val/test embedding pass before training starts. "
            "Disabled by default to avoid long cross-rank barrier waits on large models."
        ),
    )
    parser = deepspeed.add_config_arguments(parser)
    args, _ = parser.parse_known_args()
    return args


ARGS = parse_args()
USE_4BIT = not ARGS.no_4bit
ORGANISM = "Homo sapiens"

deepspeed.init_distributed()
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", getattr(ARGS, "local_rank", 0)))
WORLD_SIZE = dist.get_world_size()
GLOBAL_RANK = dist.get_rank()
IS_MAIN = GLOBAL_RANK == 0

torch.cuda.set_device(LOCAL_RANK)
DEVICE = torch.device("cuda", LOCAL_RANK)

random.seed(ARGS.seed + GLOBAL_RANK)
np.random.seed(ARGS.seed + GLOBAL_RANK)
torch.manual_seed(ARGS.seed + GLOBAL_RANK)
torch.cuda.manual_seed_all(ARGS.seed + GLOBAL_RANK)

run_name = ARGS.run_name or time.strftime("stage1_%Y%m%d_%H%M%S")
default_output_dir = SCRIPT_DIR / "outputs" / run_name
if ARGS.output_dir:
    OUTPUT_DIR = Path(ARGS.output_dir)
elif ARGS.output_root:
    OUTPUT_DIR = Path(ARGS.output_root) / run_name
else:
    OUTPUT_DIR = default_output_dir
METRICS_DIR = OUTPUT_DIR / "metrics"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"
for directory in (OUTPUT_DIR, METRICS_DIR, CHECKPOINTS_DIR, ARTIFACTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def is_main_process() -> bool:
    return IS_MAIN


def log_main(message: str) -> None:
    if is_main_process():
        print(message, flush=True)


def wait_for_main_only_section(section_name: str, work_fn) -> None:
    marker_dir = OUTPUT_DIR if OUTPUT_DIR.exists() else Path(tempfile.gettempdir())
    marker_path = marker_dir / f".{section_name}.done"
    if is_main_process() and marker_path.exists():
        marker_path.unlink()
    dist.barrier()
    if is_main_process():
        try:
            work_fn()
        finally:
            marker_path.touch()
    else:
        while not marker_path.exists():
            time.sleep(5)


def _is_disk_quota_error(error: OSError) -> bool:
    return error.errno in {errno.ENOSPC, errno.EDQUOT, 122}


def _warn_noncritical_write(path: Path, error: OSError) -> None:
    log_main(f"[Warn] Skipping write to {path} due to filesystem limit: {error}")


def write_json_best_effort(path: Path, payload: dict, *, critical: bool = False) -> bool:
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return True
    except OSError as error:
        if critical or not _is_disk_quota_error(error):
            raise
        _warn_noncritical_write(path, error)
        return False


def torch_save_best_effort(path: Path, payload: dict, *, critical: bool = False) -> bool:
    try:
        torch.save(payload, str(path))
        return True
    except OSError as error:
        if critical or not _is_disk_quota_error(error):
            raise
        _warn_noncritical_write(path, error)
        return False


DATASET_DIR = SCRIPT_DIR.parents[3] / "dataset" / "datasets" / "Sadra"
DATA_PATHS = {
    "train": DATASET_DIR / "immune1_celltype_train.jsonl",
    "val": DATASET_DIR / "immune1_celltype_val.jsonl",
    "test": DATASET_DIR / "immune1_celltype_test.jsonl",
}


def _parse_label_from_answer(answer_text: str | None) -> str | None:
    if not answer_text:
        return None
    for raw_line in str(answer_text).splitlines():
        line = raw_line.strip()
        if line.startswith("FINAL:") or line.startswith("LABEL:"):
            return line.split(":", 1)[1].strip()
    text = str(answer_text).strip()
    return text or None


def _load_records_from_jsonl(path: Path, top_genes: int, label_key: str) -> list[tuple[list[str], str]]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            genes = record.get("genes")
            if isinstance(genes, list):
                genes = [str(gene) for gene in genes][:top_genes]
            else:
                question_text = record.get("question") or record.get("question_text") or ""
                genes = extract_genes_from_question(question_text)[:top_genes]

            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            label = metadata.get(label_key) or metadata.get("cell_type") or record.get("label")
            if not label:
                label = _parse_label_from_answer(record.get("answer") or record.get("answer_text"))

            if not genes or not label:
                continue
            records.append((genes, str(label)))
    return records


def _records_to_split(records: list[tuple[list[str], str]]) -> tuple[list[list[str]], list[str]]:
    genes_list = [genes for genes, _ in records]
    labels_list = [label for _, label in records]
    return genes_list, labels_list


def _stratified_split_records(
    records: list[tuple[list[str], str]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[tuple[list[str], str]]]:
    by_label: dict[str, list[tuple[list[str], str]]] = {}
    for genes, label in records:
        by_label.setdefault(label, []).append((genes, label))

    rng = random.Random(seed)
    split_records = {"train": [], "val": [], "test": []}
    for label_records in by_label.values():
        rng.shuffle(label_records)
        count = len(label_records)
        train_count = max(1, int(count * train_ratio))
        val_count = 0 if count < 10 else max(1, int(count * val_ratio))
        if train_count + val_count > count:
            val_count = max(0, count - train_count)
        test_count = count - train_count - val_count
        if count >= 3 and test_count == 0 and train_count > 1:
            train_count -= 1
            test_count += 1
        if count >= 10 and val_count == 0 and train_count > 1:
            train_count -= 1
            val_count += 1

        split_records["train"].extend(label_records[:train_count])
        split_records["val"].extend(label_records[train_count : train_count + val_count])
        split_records["test"].extend(label_records[train_count + val_count : train_count + val_count + test_count])
    return split_records


def _resolve_split_records(args: argparse.Namespace) -> tuple[dict[str, list[tuple[list[str], str]]], dict[str, str]]:
    explicit_paths = {
        "train": Path(args.train_data) if args.train_data else None,
        "val": Path(args.val_data) if args.val_data else None,
        "test": Path(args.test_data) if args.test_data else None,
    }
    if any(path is not None for path in explicit_paths.values()):
        return (
            {
                split_name: _load_records_from_jsonl(path, args.top_genes, args.label_key) if path else []
                for split_name, path in explicit_paths.items()
            },
            {
                "mode": "explicit_split_files",
                "train": str(explicit_paths["train"]) if explicit_paths["train"] else "",
                "val": str(explicit_paths["val"]) if explicit_paths["val"] else "",
                "test": str(explicit_paths["test"]) if explicit_paths["test"] else "",
            },
        )

    if args.data_root:
        data_root = Path(args.data_root)
        discovered_files = sorted(data_root.glob(args.data_glob))
        if not discovered_files:
            raise FileNotFoundError(f"No JSONL files matched {args.data_glob} under {data_root}")
        merged_records = []
        for path in discovered_files:
            merged_records.extend(_load_records_from_jsonl(path, args.top_genes, args.label_key))
        return (
            _stratified_split_records(merged_records, args.train_ratio, args.val_ratio, args.seed),
            {
                "mode": "discovered_export_files",
                "data_root": str(data_root),
                "data_glob": args.data_glob,
                "file_count": str(len(discovered_files)),
            },
        )

    return (
        {split_name: _load_records_from_jsonl(path, args.top_genes, args.label_key) for split_name, path in DATA_PATHS.items()},
        {"mode": "legacy_sadra", **{split_name: str(path) for split_name, path in DATA_PATHS.items()}},
    )


resolved_split_records, data_source_info = _resolve_split_records(ARGS)
split_data: dict[str, dict[str, list]] = {}
for split_name in ("train", "val", "test"):
    genes, labels = _records_to_split(resolved_split_records.get(split_name, []))
    split_data[split_name] = {"genes": genes, "labels": labels}

all_labels = []
for split_name in ("train", "val", "test"):
    all_labels.extend(split_data[split_name]["labels"])

unique_labels = sorted(set(all_labels))
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
NUM_CLASSES = len(unique_labels)

log_main(f"World size     : {WORLD_SIZE}")
log_main(f"Base model     : {ARGS.model_path}")
log_main(f"Output dir     : {OUTPUT_DIR}")
log_main(f"Data source    : {data_source_info['mode']}")
log_main(f"4-bit QLoRA    : {USE_4BIT}")
log_main(f"Freeze backbone: {ARGS.freeze_backbone}")
log_main(f"Epochs: {ARGS.epochs}  LR: {ARGS.lr}  Micro-batch/GPU: {ARGS.batch}  GradAccum: {ARGS.grad_accum}")
log_main(f"Effective batch: {ARGS.batch * WORLD_SIZE * ARGS.grad_accum}")
log_main(f"Temperature: {ARGS.temperature}  Projection dim: {ARGS.projection_dim}")
log_main(f"Train samples  : {len(split_data['train']['genes'])}")
log_main(f"Val samples    : {len(split_data['val']['genes'])}")
log_main(f"Test samples   : {len(split_data['test']['genes'])}")
log_main(f"Unique labels  : {NUM_CLASSES}")
log_main(f"Startup full eval: {ARGS.startup_full_eval}")
if is_main_process():
    if data_source_info["mode"] == "discovered_export_files":
        print(f"Discovered export root : {data_source_info['data_root']}")
        print(f"Discovered export glob : {data_source_info['data_glob']}")
        print(f"Matched files          : {data_source_info['file_count']}")
    elif data_source_info["mode"] == "explicit_split_files":
        print(f"Train split path       : {data_source_info.get('train', '')}")
        print(f"Val split path         : {data_source_info.get('val', '')}")
        print(f"Test split path        : {data_source_info.get('test', '')}")
    else:
        print(f"Legacy train path      : {data_source_info['train']}")
        print(f"Legacy val path        : {data_source_info['val']}")
        print(f"Legacy test path       : {data_source_info['test']}")
    label_counts = Counter(split_data["train"]["labels"])
    for label, count in label_counts.most_common(5):
        print(f"  {count:5d}  {label}")
    print()


class GeneAugmentor:
    def __init__(
        self,
        dropout_range: tuple[float, float] = (0.10, 0.30),
        swap_prob: float = 0.10,
        subsample_range: tuple[int, int] = (120, 200),
    ):
        self.dropout_range = dropout_range
        self.swap_prob = swap_prob
        self.subsample_range = subsample_range

    def __call__(self, genes: list[str]) -> list[str]:
        augmented = list(genes)

        dropout_rate = random.uniform(*self.dropout_range)
        num_drop = int(len(augmented) * dropout_rate)
        if num_drop > 0:
            drop_indices = set(random.sample(range(len(augmented)), num_drop))
            augmented = [gene for idx, gene in enumerate(augmented) if idx not in drop_indices]

        for idx in range(len(augmented) - 1):
            if random.random() < self.swap_prob:
                augmented[idx], augmented[idx + 1] = augmented[idx + 1], augmented[idx]

        keep_count = random.randint(*self.subsample_range)
        augmented = augmented[: min(keep_count, len(augmented))]
        return augmented


class SingleCellContrastiveDataset(Dataset):
    def __init__(self, genes_list: list[list[str]], augmentor: GeneAugmentor):
        self.genes_list = genes_list
        self.augmentor = augmentor

    def __len__(self) -> int:
        return len(self.genes_list)

    def __getitem__(self, index: int) -> tuple[str, str]:
        genes = self.genes_list[index]
        prompt1 = build_c2s_prompt(self.augmentor(genes), organism=ORGANISM)
        prompt2 = build_c2s_prompt(self.augmentor(genes), organism=ORGANISM)
        return prompt1, prompt2


def collate_fn(batch: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    prompts1, prompts2 = zip(*batch)
    return list(prompts1), list(prompts2)


log_main("Loading tokenizer from base model ...")
tokenizer = AutoTokenizer.from_pretrained(ARGS.model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

log_main("Loading base model ...")
if USE_4BIT:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        ARGS.model_path,
        quantization_config=quant_config,
        device_map={"": LOCAL_RANK},
        attn_implementation="eager",
        trust_remote_code=True,
    )
    base_model = prepare_model_for_kbit_training(base_model)
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        ARGS.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": LOCAL_RANK},
        attn_implementation="eager",
        trust_remote_code=True,
    )


def resolve_target_layers(num_hidden_layers: int, args: argparse.Namespace) -> list[int] | None:
    if args.layer_start is not None or args.layer_end is not None:
        start = 0 if args.layer_start is None else args.layer_start
        end = num_hidden_layers if args.layer_end is None else args.layer_end
        start = max(0, start)
        end = min(num_hidden_layers, end)
        if start >= end:
            raise ValueError(f"Invalid layer range: start={start}, end={end}, total={num_hidden_layers}")
        return list(range(start, end))

    if args.layer_strategy == "all":
        return None
    if args.layer_strategy == "upper_half":
        return list(range(num_hidden_layers // 2, num_hidden_layers))
    if args.layer_strategy == "last_third":
        return list(range((2 * num_hidden_layers) // 3, num_hidden_layers))
    raise ValueError(f"Unsupported layer strategy: {args.layer_strategy}")


target_modules = [name.strip() for name in ARGS.lora_target_modules.split(",") if name.strip()]
num_hidden_layers = int(base_model.config.num_hidden_layers)
layers_to_transform = resolve_target_layers(num_hidden_layers, ARGS)

log_main("Initializing stage-1 LoRA adapter ...")
lora_kwargs = {
    "r": ARGS.lora_r,
    "lora_alpha": ARGS.lora_alpha,
    "target_modules": target_modules,
    "lora_dropout": ARGS.lora_dropout,
    "bias": "none",
    "task_type": TaskType.CAUSAL_LM,
}
if layers_to_transform is not None:
    lora_kwargs["layers_to_transform"] = layers_to_transform
    lora_kwargs["layers_pattern"] = "layers"
lora_config = LoraConfig(**lora_kwargs)

gemma_model = get_peft_model(base_model, lora_config)
gemma_model.config.use_cache = False

HIDDEN_SIZE = int(gemma_model.config.hidden_size)
log_main(f"Model loaded — hidden_size: {HIDDEN_SIZE}")
log_main(f"LoRA target modules: {target_modules}")
if layers_to_transform is None:
    log_main("LoRA layer scope  : all transformer blocks")
else:
    log_main(
        f"LoRA layer scope  : {len(layers_to_transform)} blocks "
        f"(min={layers_to_transform[0]}, max={layers_to_transform[-1]})"
    )
if is_main_process():
    gemma_model.print_trainable_parameters()
    print()

projection_head = ProjectionHead(
    input_dim=HIDDEN_SIZE,
    hidden_dim=HIDDEN_SIZE // 2,
    output_dim=ARGS.projection_dim,
).to(DEVICE)
log_main(f"Projection head: {HIDDEN_SIZE} -> {HIDDEN_SIZE // 2} -> {ARGS.projection_dim}")


class ContrastiveModel(nn.Module):
    def __init__(self, backbone, proj_head, tokenizer_obj, max_seq_len: int):
        super().__init__()
        self.backbone = backbone
        self.proj_head = proj_head
        self.tokenizer = tokenizer_obj
        self.max_seq_len = max_seq_len

    def forward(self, prompts1: list[str], prompts2: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        all_prompts = list(prompts1) + list(prompts2)
        hidden = get_hidden_states(self.backbone, self.tokenizer, all_prompts, self.max_seq_len)
        projected = self.proj_head(hidden)
        return projected.chunk(2, dim=0)


model = ContrastiveModel(gemma_model, projection_head, tokenizer, ARGS.max_seq_len)
if ARGS.freeze_backbone:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    log_main("Backbone frozen — training projection head only")
else:
    model.backbone.train()
    lora_parameters = sum(param.numel() for param in model.backbone.parameters() if param.requires_grad)
    projection_parameters = sum(param.numel() for param in model.proj_head.parameters())
    log_main(f"Training LoRA params ({lora_parameters:,}) + projection head ({projection_parameters:,})")

train_dataset = SingleCellContrastiveDataset(split_data["train"]["genes"], GeneAugmentor())
train_sampler = DistributedSampler(
    train_dataset,
    num_replicas=WORLD_SIZE,
    rank=GLOBAL_RANK,
    shuffle=True,
    seed=ARGS.seed,
    drop_last=True,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=ARGS.batch,
    sampler=train_sampler,
    drop_last=True,
    num_workers=0,
    collate_fn=collate_fn,
)
log_main(f"Train dataset size: {len(train_dataset)}  |  Batches/GPU/epoch: {len(train_loader)}")


def gather_from_all_gpus(tensor: torch.Tensor) -> torch.Tensor:
    if WORLD_SIZE == 1:
        return tensor

    gathered = [torch.zeros_like(tensor) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered, tensor.detach())
    gathered[GLOBAL_RANK] = tensor
    return torch.cat(gathered, dim=0)


class SimCLRLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        z_i_all = gather_from_all_gpus(z_i)
        z_j_all = gather_from_all_gpus(z_j)
        num_samples = z_i_all.shape[0]
        combined = torch.cat([z_i_all, z_j_all], dim=0)

        similarity = torch.mm(combined, combined.t()) / self.temperature
        self_mask = torch.eye(2 * num_samples, dtype=torch.bool, device=combined.device)
        similarity.masked_fill_(self_mask, -1e9)

        indices = torch.arange(num_samples, device=combined.device)
        positive_mask = torch.zeros(2 * num_samples, 2 * num_samples, dtype=torch.bool, device=combined.device)
        positive_mask[indices, indices + num_samples] = True
        positive_mask[indices + num_samples, indices] = True

        log_prob = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
        return -(log_prob * positive_mask.float()).sum(dim=1).mean()


criterion = SimCLRLoss(temperature=ARGS.temperature)
log_main(f"Loss: SimCLR NT-Xent (temperature={ARGS.temperature})")

total_steps = ARGS.epochs * max(1, len(train_loader) // ARGS.grad_accum)
ds_config = {
    "train_batch_size": ARGS.batch * WORLD_SIZE * ARGS.grad_accum,
    "train_micro_batch_size_per_gpu": ARGS.batch,
    "gradient_accumulation_steps": ARGS.grad_accum,
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "none"},
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
    },
    "gradient_clipping": 1.0,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": ARGS.lr,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
        },
    },
    "steps_per_print": 10,
    "wall_clock_breakdown": False,
}

engine, optimizer, _, _ = deepspeed.initialize(
    args=ARGS,
    model=model,
    model_parameters=[parameter for parameter in model.parameters() if parameter.requires_grad],
    config=ds_config,
)


class ManualWarmupCosineScheduler:
    def __init__(self, optimizer_obj, base_lr: float, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer_obj
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = max(total_steps, 1)
        self.step_count = 0
        self.last_lr = [base_lr]

    def _compute_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * float(step) / max(1.0, float(self.warmup_steps))
        progress = float(step - self.warmup_steps) / max(1.0, float(self.total_steps - self.warmup_steps))
        return max(0.0, self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress)))

    def step(self) -> None:
        self.step_count += 1
        lr_value = self._compute_lr(self.step_count)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr_value
        self.last_lr = [lr_value]

    def get_last_lr(self) -> list[float]:
        return self.last_lr


scheduler = ManualWarmupCosineScheduler(
    optimizer,
    base_lr=ARGS.lr,
    warmup_steps=ARGS.warmup_steps,
    total_steps=total_steps,
)
log_main(f"DeepSpeed engine initialized (ZeRO stage 2) with {total_steps} total steps")


def extract_embeddings_for_split(split_name: str) -> dict:
    backbone = engine.module.backbone
    proj_head = engine.module.proj_head
    backbone.eval()
    proj_head.eval()

    genes = split_data[split_name]["genes"]
    labels = split_data[split_name]["labels"]
    if not genes:
        return {
            "split": split_name,
            "n_samples": 0,
            "retrieval": {"features": None, "projections": None},
        }

    all_features = []
    all_projections = []
    with torch.no_grad():
        for start in range(0, len(genes), ARGS.eval_batch):
            batch_genes = genes[start : start + ARGS.eval_batch]
            prompts = [build_c2s_prompt(sample_genes, organism=ORGANISM) for sample_genes in batch_genes]
            hidden = get_hidden_states(backbone, tokenizer, prompts, ARGS.max_seq_len, enable_grad=False)
            projected = proj_head(hidden)
            all_features.append(hidden.float().cpu().numpy())
            all_projections.append(projected.float().cpu().numpy())

    features = np.concatenate(all_features, axis=0)
    projections = np.concatenate(all_projections, axis=0)
    labels_array = np.array(labels)
    mask = np.ones(len(labels_array), dtype=bool)

    feature_metrics = print_recall_at_k(
        f"{split_name.upper()} FEATURES",
        features,
        labels_array,
        mask,
        run_tag=split_name,
    )
    projection_metrics = print_recall_at_k(
        f"{split_name.upper()} PROJECTIONS",
        projections,
        labels_array,
        mask,
        run_tag=split_name,
    )
    return {
        "split": split_name,
        "n_samples": int(len(labels_array)),
        "features": features,
        "projections": projections,
        "labels": labels_array,
        "retrieval": {
            "features": feature_metrics,
            "projections": projection_metrics,
        },
    }


def save_embedding_artifacts(step_name: str, split_embeddings: dict[str, dict]) -> None:
    arrays_to_save = {}
    for split_name, split_result in split_embeddings.items():
        arrays_to_save[f"{split_name}_features"] = split_result["features"]
        arrays_to_save[f"{split_name}_projections"] = split_result["projections"]
        arrays_to_save[f"{split_name}_labels"] = split_result["labels"]
    output_path = ARTIFACTS_DIR / f"representations_{step_name}.npz"
    np.savez(str(output_path), **arrays_to_save)
    log_main(f"Saved representations to: {output_path}")


def plot_umap_embeddings(step_name: str, split_embeddings: dict[str, dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from umap import UMAP
    except ImportError:
        log_main("Skipping embedding plots because matplotlib or umap-learn is unavailable.")
        return

    plots_dir = ARTIFACTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    combined_splits = []
    for split_name in ["train", "val", "test"]:
        if split_name in split_embeddings:
            combined_splits.append(split_name)
    if not combined_splits:
        return

    def plot_embedding_kind(kind: str) -> None:
        combined_embeddings = []
        combined_labels = []
        combined_split_tags = []
        for split_name in combined_splits:
            split_result = split_embeddings[split_name]
            combined_embeddings.append(split_result[kind])
            combined_labels.append(split_result["labels"])
            combined_split_tags.append(np.full(split_result["n_samples"], split_name, dtype=object))

        embedding_matrix = np.concatenate(combined_embeddings, axis=0)
        labels_array = np.concatenate(combined_labels, axis=0)
        split_array = np.concatenate(combined_split_tags, axis=0)

        reducer = UMAP(n_components=2, random_state=ARGS.seed, n_neighbors=30, min_dist=0.3)
        reduced = reducer.fit_transform(embedding_matrix)

        def save_plot(mask: np.ndarray, split_tag: str) -> None:
            if not np.any(mask):
                return

            fig, ax = plt.subplots(figsize=(14, 10))
            subset_labels = labels_array[mask]
            subset_embedding = reduced[mask]
            unique_types = sorted(set(subset_labels))
            cmap = plt.cm.get_cmap("tab20", len(unique_types))

            for index, cell_type in enumerate(unique_types):
                label_mask = subset_labels == cell_type
                ax.scatter(
                    subset_embedding[label_mask, 0],
                    subset_embedding[label_mask, 1],
                    label=cell_type,
                    c=[cmap(index)],
                    s=8,
                    alpha=0.7,
                )

            title_name = split_tag.upper() if split_tag != "all" else "ALL"
            ax.set_title(f"UMAP {kind}: {title_name} [{step_name}]", fontsize=14)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7, markerscale=3)
            plt.tight_layout()
            output_path = plots_dir / f"umap_{step_name}_{kind}_{split_tag}.png"
            plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            log_main(f"Saved plot to: {output_path}")

        save_plot(np.ones(len(labels_array), dtype=bool), "all")
        for split_name in combined_splits:
            save_plot(split_array == split_name, split_name)

    plot_embedding_kind("features")
    plot_embedding_kind("projections")


def selection_score_from_metrics(metrics: dict) -> float:
    retrieval = metrics["splits"]["val"]["retrieval"][ARGS.selection_space]
    if retrieval is None:
        return float("-inf")
    key = f"R@{ARGS.selection_k}"
    score = retrieval["recall"].get(key)
    if score is None:
        return float("-inf")
    return float(score)


def evaluate(step_name: str, epoch: int | None, splits: tuple[str, ...], save_artifacts: bool = False) -> dict:
    metric_payload = {
        "step": step_name,
        "epoch": epoch,
        "selection_space": ARGS.selection_space,
        "selection_k": ARGS.selection_k,
        "splits": {},
    }
    split_embeddings = {}
    for split_name in splits:
        split_result = extract_embeddings_for_split(split_name)
        split_embeddings[split_name] = split_result
        metric_payload["splits"][split_name] = {
            "split": split_result["split"],
            "n_samples": split_result["n_samples"],
            "retrieval": split_result["retrieval"],
        }
    if "val" in metric_payload["splits"]:
        metric_payload["selection_score"] = selection_score_from_metrics(metric_payload)
    metrics_path = METRICS_DIR / f"metrics_{step_name}.json"
    if write_json_best_effort(metrics_path, metric_payload, critical=False):
        log_main(f"Saved metrics to: {metrics_path}")
    if save_artifacts:
        save_embedding_artifacts(step_name, split_embeddings)
        plot_umap_embeddings(step_name, split_embeddings)
    return metric_payload


def load_checkpoint_for_evaluation(tag: str, adapter_name: str) -> None:
    checkpoint_dir = CHECKPOINTS_DIR / tag
    projection_path = checkpoint_dir / "projection_head.pt"
    projection_payload = torch.load(str(projection_path), map_location="cpu")
    engine.module.proj_head.load_state_dict(projection_payload["state_dict"])

    if not ARGS.freeze_backbone:
        adapter_dir = checkpoint_dir / "gemma_lora_stage1"
        if adapter_name in gemma_model.peft_config:
            gemma_model.delete_adapter(adapter_name)
        gemma_model.load_adapter(str(adapter_dir), adapter_name=adapter_name, is_trainable=False)
        gemma_model.set_adapter(adapter_name)
    engine.module.backbone.eval()
    engine.module.proj_head.eval()


def save_checkpoint(tag: str, epoch: int, selection_score: float | None, history: dict, metrics_snapshot: dict | None) -> None:
    checkpoint_dir = CHECKPOINTS_DIR / tag
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    projection_path = checkpoint_dir / "projection_head.pt"
    torch_save_best_effort(
        projection_path,
        {
            "state_dict": engine.module.proj_head.state_dict(),
            "config": {
                "input_dim": HIDDEN_SIZE,
                "hidden_dim": HIDDEN_SIZE // 2,
                "output_dim": ARGS.projection_dim,
            },
        },
        critical=True,
    )

    if not ARGS.freeze_backbone:
        adapter_dir = checkpoint_dir / "gemma_lora_stage1"
        gemma_model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

    metadata = {
        "epoch": epoch,
        "selection_score": selection_score,
        "args": vars(ARGS),
        "history": history,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "hidden_size": HIDDEN_SIZE,
        "num_classes": NUM_CLASSES,
        "target_modules": target_modules,
        "layers_to_transform": layers_to_transform,
        "metrics_snapshot": metrics_snapshot,
    }
    write_json_best_effort(checkpoint_dir / "metadata.json", metadata, critical=False)
    log_main(f"Saved {tag} checkpoint to: {checkpoint_dir}")


history = {
    "epoch": [],
    "loss": [],
    "lr": [],
    "val_selection_score": [],
    "best_selection_score": [],
}
best_state = {
    "score": float("-inf"),
    "epoch": None,
}

def run_startup_validation() -> None:
    log_main("Running startup validation retrieval before training ...")
    pretrain_metrics = evaluate("pre_train_val", epoch=0, splits=("val",))
    if ARGS.startup_full_eval:
        log_main("Running full startup embedding/artifact pass ...")
        evaluate("pre_train_full", epoch=0, splits=("train", "val", "test"), save_artifacts=True)
    history["val_selection_score"].append(pretrain_metrics["selection_score"])
    history["best_selection_score"].append(pretrain_metrics["selection_score"])


wait_for_main_only_section("startup_validation", run_startup_validation)

log_main(f"Training for {ARGS.epochs} epochs")
log_main(f"Negatives per sample: {2 * ARGS.batch * WORLD_SIZE - 2}")
start_time = time.time()

for epoch in range(1, ARGS.epochs + 1):
    train_sampler.set_epoch(epoch)
    engine.train()
    epoch_loss = 0.0

    for step, (prompts1, prompts2) in enumerate(train_loader, start=1):
        z1, z2 = engine(prompts1, prompts2)
        loss = criterion(z1, z2)
        engine.backward(loss)
        engine.step()
        scheduler.step()
        epoch_loss += loss.item()

        if is_main_process() and step % 10 == 0:
            elapsed = time.time() - start_time
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f} "
                f"| LR: {current_lr:.2e} | Elapsed: {elapsed:.0f}s",
                flush=True,
            )

    avg_loss = epoch_loss / max(len(train_loader), 1)
    loss_tensor = torch.tensor(avg_loss, device=DEVICE)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    avg_loss_global = float(loss_tensor.item())
    current_lr = scheduler.get_last_lr()[0]

    history["epoch"].append(epoch)
    history["loss"].append(avg_loss_global)
    history["lr"].append(current_lr)
    log_main(f"Epoch {epoch:3d}/{ARGS.epochs} | Loss: {avg_loss_global:.4f} | LR: {current_lr:.2e}")

    dist.barrier()
    epoch_metrics = None
    if is_main_process():
        epoch_metrics = evaluate(f"epoch_{epoch:03d}_val", epoch=epoch, splits=("val",))
        selection_score = float(epoch_metrics["selection_score"])
        history["val_selection_score"].append(selection_score)

        if selection_score > best_state["score"]:
            best_state["score"] = selection_score
            best_state["epoch"] = epoch
            save_checkpoint("best", epoch, selection_score, copy.deepcopy(history), epoch_metrics)
            log_main(
                f"New best checkpoint at epoch {epoch} with "
                f"{ARGS.selection_space} Recall@{ARGS.selection_k} = {selection_score:.4f}"
            )
        history["best_selection_score"].append(best_state["score"])
    dist.barrier()

total_time = time.time() - start_time
log_main(f"Training complete in {total_time:.1f}s")

def run_final_evaluation() -> None:
    final_metrics = evaluate("last_full", epoch=ARGS.epochs, splits=("train", "val", "test"))
    save_checkpoint("last", ARGS.epochs, final_metrics.get("selection_score"), history, final_metrics)
    if best_state["epoch"] is not None:
        load_checkpoint_for_evaluation("best", adapter_name="best_eval")
        best_metrics = evaluate("best_full", epoch=best_state["epoch"], splits=("train", "val", "test"), save_artifacts=True)
    else:
        best_metrics = None

    summary = {
        "run_name": run_name,
        "output_dir": str(OUTPUT_DIR),
        "best_epoch": best_state["epoch"],
        "best_selection_score": best_state["score"],
        "best_full_metrics": best_metrics,
        "selection_metric": f"val_{ARGS.selection_space}_R@{ARGS.selection_k}",
        "total_time_seconds": total_time,
        "history": history,
    }
    write_json_best_effort(ARTIFACTS_DIR / "training_summary.json", summary, critical=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (loss_ax, metric_ax) = plt.subplots(1, 2, figsize=(14, 5))
    loss_ax.plot(history["epoch"], history["loss"], color="#1f77b4", linewidth=2)
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Contrastive Loss")
    loss_ax.set_title("Stage-1 Training Loss")
    loss_ax.grid(True, alpha=0.3)

    metric_epochs = list(range(0, len(history["val_selection_score"])))
    metric_ax.plot(metric_epochs, history["val_selection_score"], color="#d62728", linewidth=2, label="val score")
    metric_ax.plot(metric_epochs, history["best_selection_score"], color="#2ca02c", linewidth=2, label="best so far")
    metric_ax.set_xlabel("Evaluation Index")
    metric_ax.set_ylabel(f"Recall@{ARGS.selection_k}")
    metric_ax.set_title(f"Validation {ARGS.selection_space.title()} Retrieval")
    metric_ax.grid(True, alpha=0.3)
    metric_ax.legend()

    plt.tight_layout()
    try:
        plt.savefig(str(ARTIFACTS_DIR / "training_curves.png"), dpi=150, bbox_inches="tight")
    except OSError as error:
        if not _is_disk_quota_error(error):
            raise
        _warn_noncritical_write(ARTIFACTS_DIR / "training_curves.png", error)
    plt.close(figure)

    print("=" * 60)
    print(f"Best epoch: {best_state['epoch']}  |  Best score: {best_state['score']:.4f}")
    print(f"Artifacts saved to: {OUTPUT_DIR}")
    print("=" * 60)

wait_for_main_only_section("final_evaluation", run_final_evaluation)