#!/usr/bin/env python
"""
═══════════════════════════════════════════════════════════════════
  SimCLR Contrastive Learning (Self-Supervised) with Gemma 2B
  ── DeepSpeed multi-GPU version ──
═══════════════════════════════════════════════════════════════════

Launch with:
    deepspeed --num_gpus <N> contrastive_learning.py \
        --deepspeed ds_config.json \
        --batch 16 --epochs 10

DeepSpeed ZeRO-2 shards optimizer states + gradients across GPUs.
SimCLR negatives are gathered across all GPUs so the effective
negative pool is (2 * per-gpu-batch * world_size - 2).

═══════════════════════════════════════════════════════════════════

With 2 GPUs, you can run:
deepspeed --num_gpus 2 contrastive_learning.py \
    --batch 64 \
    --grad-accum 1 \
    --epochs 20 \
    --lr 5e-4 \
    --warmup-steps 50 \
    --temperature 0.05 \
    --max-seq-len 512 \
    --projection-dim 128 \
    --top-genes 200 \
    --seed 42

"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  1. Imports & Configuration                                  ║
# ╚══════════════════════════════════════════════════════════════╝

import json
import os
import random
import time
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import math

import deepspeed

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType
from utils import (
    ProjectionHead,
    build_c2s_prompt,
    extract_genes_from_question,
    get_hidden_states,
    print_recall_at_k,
)

# ── Paths ──────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = (
    SCRIPT_DIR.parents[2]   # benchmarks/
    / "dataset" / "datasets" / "Sadra"
)
DATA_PATHS = [
    DATASET_DIR / "immune1_celltype_train.jsonl",
    DATASET_DIR / "immune1_celltype_val.jsonl",
    DATASET_DIR / "immune1_celltype_test.jsonl",
]
BASE_MODEL_PATH = os.getenv("GEMMA_MODEL_PATH", "vandijklab/C2S-Scale-Gemma-2-2B")
SAVE_DIR = SCRIPT_DIR / "checkpoints"

# ── CLI ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="SimCLR contrastive learning with Gemma 2B (DeepSpeed)")
parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--batch", type=int, default=16, help="Per-GPU micro-batch size")
parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
parser.add_argument("--warmup-steps", type=int, default=100, help="LR warmup steps")
parser.add_argument("--temperature", type=float, default=0.07, help="NT-Xent temperature")
parser.add_argument("--projection-dim", type=int, default=128, help="Projection head output dim")
parser.add_argument("--max-seq-len", type=int, default=512, help="Max sequence length")
parser.add_argument("--top-genes", type=int, default=200, help="Number of top genes to keep")
parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantisation")
parser.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone, only train projection head")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
# Accept launcher-injected rank (torch.distributed / deepspeed pass --local_rank)
parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training (injected by launcher)")
# DeepSpeed adds its own args (--local_rank, --deepspeed, etc.)
parser = deepspeed.add_config_arguments(parser)
# Use parse_known_args to ignore any unexpected extra launcher args safely
args, _ = parser.parse_known_args()

USE_4BIT = not args.no_4bit
ORGANISM = "Homo sapiens"

# ── DeepSpeed / distributed init ──────────────────────────────
deepspeed.init_distributed()
# Prefer the environment variable set by the launcher, fallback to parsed arg
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", getattr(args, "local_rank", 0)))
WORLD_SIZE = dist.get_world_size()
GLOBAL_RANK = dist.get_rank()
IS_MAIN = GLOBAL_RANK == 0

torch.cuda.set_device(LOCAL_RANK)
DEVICE = torch.device("cuda", LOCAL_RANK)

# Reproducibility
random.seed(args.seed + GLOBAL_RANK)
np.random.seed(args.seed + GLOBAL_RANK)
torch.manual_seed(args.seed + GLOBAL_RANK)
torch.cuda.manual_seed_all(args.seed + GLOBAL_RANK)

SAVE_DIR.mkdir(parents=True, exist_ok=True)

if IS_MAIN:
    print(f"World size     : {WORLD_SIZE}")
    print(f"Base model     : {BASE_MODEL_PATH}")
    print(f"Data files     : {[p.name for p in DATA_PATHS]}")
    print(f"4-bit QLoRA    : {USE_4BIT}")
    print(f"Freeze backbone: {args.freeze_backbone}")
    print(f"Epochs: {args.epochs}  LR: {args.lr}  Micro-batch/GPU: {args.batch}  GradAccum: {args.grad_accum}")
    print(f"Effective batch: {args.batch * WORLD_SIZE * args.grad_accum}")
    print(f"Temperature: {args.temperature}  Projection dim: {args.projection_dim}")
    print()


# ╔══════════════════════════════════════════════════════════════╗
# ║  2. Load & Parse the Data                                    ║
# ╚══════════════════════════════════════════════════════════════╝

raw_data = []
split_tags = []  # track train/val/test
for data_path in DATA_PATHS:
    # Identify split from filename
    s_name = "other"
    if "train" in data_path.name:
        s_name = "train"
    elif "val" in data_path.name:
        s_name = "val"
    elif "test" in data_path.name:
        s_name = "test"

    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_data.append(json.loads(line))
                split_tags.append(s_name)

if IS_MAIN:
    print(f"Total samples loaded: {len(raw_data)}  (from {len(DATA_PATHS)} files)")


# Parse all samples
genes_list = []
labels_list = []
for record in raw_data:
    genes = extract_genes_from_question(record["question"])[:args.top_genes]
    genes_list.append(genes)
    labels_list.append(record["answer"])

# Label mapping (kept for evaluation only)
unique_labels = sorted(set(labels_list))
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
NUM_CLASSES = len(unique_labels)

if IS_MAIN:
    print(f"Unique cell types: {NUM_CLASSES}")
    label_counts = Counter(labels_list)
    for ct, count in label_counts.most_common(5):
        print(f"  {count:5d}  {ct}")
    print(f"  ... ({NUM_CLASSES - 5} more)")
    print()


# ╔══════════════════════════════════════════════════════════════╗
# ║  3. C2S Prompt Builder & Gene Augmentations                  ║
# ╚══════════════════════════════════════════════════════════════╝

class GeneAugmentor:
    """Stochastic augmentations for ordered gene lists."""

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
        genes = list(genes)

        # 1) Gene dropout
        dropout_rate = random.uniform(*self.dropout_range)
        num_drop = int(len(genes) * dropout_rate)
        if num_drop > 0:
            drop_indices = set(random.sample(range(len(genes)), num_drop))
            genes = [g for i, g in enumerate(genes) if i not in drop_indices]

        # 2) Rank perturbation
        for i in range(len(genes) - 1):
            if random.random() < self.swap_prob:
                genes[i], genes[i + 1] = genes[i + 1], genes[i]

        # 3) Random subsampling
        k = random.randint(*self.subsample_range)
        genes = genes[: min(k, len(genes))]

        return genes


augmentor = GeneAugmentor()


# ╔══════════════════════════════════════════════════════════════╗
# ║  4. Load Gemma 2B + fresh LoRA                               ║
# ╚══════════════════════════════════════════════════════════════╝

if IS_MAIN:
    print("Loading tokenizer from base model ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

if IS_MAIN:
    print("Loading base model ...")
if USE_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": LOCAL_RANK},
        attn_implementation="eager",
        trust_remote_code=True,
    )
    base_model = prepare_model_for_kbit_training(base_model)
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": LOCAL_RANK},
        attn_implementation="eager",
        trust_remote_code=True,
    )

if IS_MAIN:
    print("Initializing LoRA adapter from scratch ...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
gemma_model = get_peft_model(base_model, lora_config)
gemma_model.config.use_cache = False

HIDDEN_SIZE = gemma_model.config.hidden_size
if IS_MAIN:
    print(f"Model loaded — hidden_size: {HIDDEN_SIZE}")
    gemma_model.print_trainable_parameters()
    print()


# ╔══════════════════════════════════════════════════════════════╗
# ║  5. Projection Head                                          ║
# ╚══════════════════════════════════════════════════════════════╝

projection_head = ProjectionHead(
    input_dim=HIDDEN_SIZE,
    hidden_dim=HIDDEN_SIZE // 2,
    output_dim=args.projection_dim,
).to(DEVICE)

if IS_MAIN:
    print(f"Projection head: {HIDDEN_SIZE} -> {HIDDEN_SIZE // 2} -> {args.projection_dim}")
    print(f"  Parameters: {sum(p.numel() for p in projection_head.parameters()):,}")
    print()


# ╔══════════════════════════════════════════════════════════════╗
# ║  6. Extract Hidden State from Gemma                          ║
# ╚══════════════════════════════════════════════════════════════╝

# ╔══════════════════════════════════════════════════════════════╗
# ║  7. Dataset + Distributed Sampler                            ║
# ╚══════════════════════════════════════════════════════════════╝

class SingleCellContrastiveDataset(Dataset):
    """Returns two augmented C2S prompts per sample (no labels)."""

    def __init__(self, genes_list: list[list[str]], augmentor: GeneAugmentor):
        self.genes_list = genes_list
        self.augmentor = augmentor

    def __len__(self):
        return len(self.genes_list)

    def __getitem__(self, idx):
        genes = self.genes_list[idx]
        prompt1 = build_c2s_prompt(self.augmentor(genes), organism=ORGANISM)
        prompt2 = build_c2s_prompt(self.augmentor(genes), organism=ORGANISM)
        return prompt1, prompt2


def collate_fn(batch):
    prompts1, prompts2 = zip(*batch)
    return list(prompts1), list(prompts2)


dataset = SingleCellContrastiveDataset(genes_list=genes_list, augmentor=augmentor)

sampler = DistributedSampler(
    dataset,
    num_replicas=WORLD_SIZE,
    rank=GLOBAL_RANK,
    shuffle=True,
    seed=args.seed,
    drop_last=True,
)

dataloader = DataLoader(
    dataset,
    batch_size=args.batch,
    sampler=sampler,
    drop_last=True,
    num_workers=0,
    collate_fn=collate_fn,
)

if IS_MAIN:
    print(f"Dataset size: {len(dataset)}  |  Batches/GPU/epoch: {len(dataloader)}")
    print()


# ╔══════════════════════════════════════════════════════════════╗
# ║  8. SimCLR NT-Xent Loss with cross-GPU gathering             ║
# ╚══════════════════════════════════════════════════════════════╝


def gather_from_all_gpus(tensor: torch.Tensor) -> torch.Tensor:
    """ZeRO-safe all-gather: gathers from every GPU but only back-props
    through the local rank's slice, avoiding the double-reduction
    assertion that ZeRO-2 triggers with a custom autograd all_reduce."""
    if WORLD_SIZE == 1:
        return tensor

    tensors_gather = [torch.zeros_like(tensor) for _ in range(WORLD_SIZE)]
    dist.all_gather(tensors_gather, tensor.detach())   # no grad on remote slices
    # Replace local slice with the original (grad-carrying) tensor
    tensors_gather[GLOBAL_RANK] = tensor
    return torch.cat(tensors_gather, dim=0)


class SimCLRLoss(nn.Module):
    """NT-Xent with cross-GPU negative gathering.

    Gathers projections from all GPUs so the negative pool is
    (2 * batch * world_size - 2) per sample.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        # Gather across GPUs
        z_i_all = gather_from_all_gpus(z_i)   # [N, D]  N = B * world_size
        z_j_all = gather_from_all_gpus(z_j)   # [N, D]

        N = z_i_all.shape[0]
        z = torch.cat([z_i_all, z_j_all], dim=0)     # [2N, D]

        sim = torch.mm(z, z.t()) / self.temperature   # [2N, 2N]

        # Self-similarity mask
        self_mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim.masked_fill_(self_mask, -1e9)

        # Positive mask: (i, i+N) and (i+N, i) for each i
        idx = torch.arange(N, device=z.device)
        pos_mask = torch.zeros(2 * N, 2 * N, dtype=torch.bool, device=z.device)
        pos_mask[idx, idx + N] = True
        pos_mask[idx + N, idx] = True

        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        loss = -(log_prob * pos_mask.float()).sum(dim=1).mean()

        return loss


criterion = SimCLRLoss(temperature=args.temperature)
if IS_MAIN:
    print(f"Loss: SimCLR NT-Xent (temperature={args.temperature})  [cross-GPU gathering enabled]")


# ╔══════════════════════════════════════════════════════════════╗
# ║  9. Combine backbone + projection into one Module             ║
# ╚══════════════════════════════════════════════════════════════╝
# DeepSpeed wraps a single nn.Module.  We create a thin wrapper
# that holds both the Gemma backbone and the projection head.

class ContrastiveModel(nn.Module):
    """Wraps Gemma backbone + projection head for DeepSpeed."""

    def __init__(self, backbone, proj_head, tokenizer, max_seq_len: int):
        super().__init__()
        self.backbone = backbone
        self.proj_head = proj_head
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def forward(self, prompts1: list[str], prompts2: list[str]):
        # Concatenate both views into ONE forward pass through the backbone
        # so that each parameter is only reduced once by ZeRO-2.
        all_prompts = list(prompts1) + list(prompts2)
        h_all = get_hidden_states(self.backbone, self.tokenizer, all_prompts, self.max_seq_len)
        z_all = self.proj_head(h_all)
        z1, z2 = z_all.chunk(2, dim=0)
        return z1, z2


model = ContrastiveModel(gemma_model, projection_head, tokenizer, args.max_seq_len)

# Freeze backbone if requested
if args.freeze_backbone:
    for param in model.backbone.parameters():
        param.requires_grad = False
    if IS_MAIN:
        print("Backbone FROZEN — only training projection head")
else:
    model.backbone.train()
    if IS_MAIN:
        lora_p = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
        proj_p = sum(p.numel() for p in model.proj_head.parameters())
        print(f"Training LoRA params ({lora_p:,}) + projection head ({proj_p:,})")


# ╔══════════════════════════════════════════════════════════════╗
# ║  10. DeepSpeed Engine Init                                   ║
# ╚══════════════════════════════════════════════════════════════╝

# Compute total training steps for the scheduler
total_steps = args.epochs * (len(dataloader) // args.grad_accum)

ds_config = {
    "train_batch_size": args.batch * WORLD_SIZE * args.grad_accum,
    "train_micro_batch_size_per_gpu": args.batch,
    "gradient_accumulation_steps": args.grad_accum,
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
            "lr": args.lr,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
        },
    },
    "steps_per_print": 10,
    "wall_clock_breakdown": False,
}

# If user supplied --deepspeed <config.json>, that file's values will override
# the programmatic config above via deepspeed's merge logic.  We pass both.
engine, optimizer, _, _ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=[p for p in model.parameters() if p.requires_grad],
    config=ds_config,
)

# ── Manual warmup + cosine-decay LR helper (avoids PyTorch scheduler type-check) ──
class ManualWarmupCosineScheduler:
    """Directly sets param-group LRs on the DeepSpeed optimizer."""

    def __init__(self, optimizer, base_lr: float, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = max(total_steps, 1)
        self._step_count = 0
        self._last_lr = [base_lr]

    def _compute_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * float(step) / max(1.0, float(self.warmup_steps))
        progress = float(step - self.warmup_steps) / max(
            1.0, float(self.total_steps - self.warmup_steps)
        )
        return max(0.0, self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress)))

    def step(self):
        self._step_count += 1
        lr = self._compute_lr(self._step_count)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        self._last_lr = [lr]

    def get_last_lr(self) -> list[float]:
        return self._last_lr

scheduler = ManualWarmupCosineScheduler(
    optimizer, base_lr=args.lr,
    warmup_steps=args.warmup_steps,
    total_steps=total_steps,
)

if IS_MAIN:
    print(f"DeepSpeed engine initialized (ZeRO stage 2)")
    print(f"  Total training steps: {total_steps}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print()


def run_evaluation(step_name="post_train"):
    print(f"\n[{step_name.upper()}] Extracting representations for all cells ...")
    engine.eval()

    all_features = []
    all_proj = []
    all_labels_eval = []
    eval_metrics = {
        "step": step_name,
        "knn": {},
        "retrieval": {},
    }

    EVAL_BATCH = 8
    with torch.no_grad():
        for i in range(0, len(genes_list), EVAL_BATCH):
            batch_genes = genes_list[i : i + EVAL_BATCH]
            batch_labels = labels_list[i : i + EVAL_BATCH]
            prompts = [build_c2s_prompt(g, organism=ORGANISM) for g in batch_genes]

            h = get_hidden_states(gemma_model, tokenizer, prompts, args.max_seq_len)
            z = projection_head(h)

            all_features.append(h.float().cpu().numpy())
            all_proj.append(z.float().cpu().numpy())
            all_labels_eval.extend(batch_labels)

            if (i // EVAL_BATCH) % 50 == 0:
                print(f"  {i}/{len(genes_list)} ...")

    features = np.concatenate(all_features, axis=0)
    projections = np.concatenate(all_proj, axis=0)

    # Convert to numpy for easier indexing
    all_labels_eval = np.array(all_labels_eval)
    all_splits_eval = np.array(split_tags[:len(all_labels_eval)])

    print(f"[{step_name.upper()}] Extracted representations: {features.shape}")
    print(f"[{step_name.upper()}] Extracted projections:     {projections.shape}")

    np.savez(
        str(SAVE_DIR / f"representations_{step_name}.npz"),
        features=features,
        projections=projections,
        labels=all_labels_eval,
        splits=all_splits_eval
    )
    print(f"Representations saved to: {SAVE_DIR / f'representations_{step_name}.npz'}")

    # ── 1. UMAP Visualization (Global & Per-Split) ─────────────────
    try:
        from umap import UMAP
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "umap-learn", "-q"])
        from umap import UMAP

    print("Computing UMAP embedding (all samples) ...")
    reducer = UMAP(n_components=2, random_state=args.seed, n_neighbors=30, min_dist=0.3)
    embedding = reducer.fit_transform(features)

    def plot_umap(mask, title_suffix, filename_suffix):
        """Helper to plot and save UMAP for a subset of data."""
        n_samples = np.sum(mask)
        if n_samples == 0:
            return

        fig, ax = plt.subplots(figsize=(14, 10))
        subset_labels = all_labels_eval[mask]
        subset_embedding = embedding[mask]

        unique_types_sub = sorted(set(subset_labels))
        # Use a consistent colormap based on global labels if possible, or local
        cmap_eval = plt.cm.get_cmap("tab20", len(unique_types_sub))
        color_map = {ct: cmap_eval(i) for i, ct in enumerate(unique_types_sub)}

        for ct in unique_types_sub:
            sub_mask = (subset_labels == ct)
            ax.scatter(
                subset_embedding[sub_mask, 0],
                subset_embedding[sub_mask, 1],
                label=ct,
                c=[color_map[ct]],
                s=8,
                alpha=0.7
            )

        ax.set_title(f"UMAP: {title_suffix} ({n_samples} samples) [{step_name}]", fontsize=14)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        # Place legend outside
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7, markerscale=3)
        plt.tight_layout()

        out_path = SAVE_DIR / f"umap_{step_name}_{filename_suffix}.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        print(f"UMAP plot saved to: {out_path}")
        plt.close(fig)

    # 1) Plot ALL
    plot_umap(np.ones(len(features), dtype=bool), "All Samples", "all")

    # 2) Plot per Split
    for s_name in ["train", "val", "test"]:
        mask = (all_splits_eval == s_name)
        if np.any(mask):
            plot_umap(mask, f"{s_name.upper()} Split", s_name)

    # ── 2. k-NN Evaluation (Per Split) ─────────────────────────────
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import LabelEncoder

    def run_knn_eval(name, mask):
        n_samples = np.sum(mask)
        if n_samples < 5:
            return None

        result = {
            "n_samples": int(n_samples),
            "scores": {},
        }

        print(f"\n--- k-NN Evaluation [{step_name}]: {name} ({n_samples} samples) ---")
        sub_feats = features[mask]
        sub_labs = all_labels_eval[mask]

        if len(set(sub_labs)) < 2:
            print("  Skipping: < 2 classes.")
            result["skipped"] = "< 2 classes"
            return result

        le_local = LabelEncoder()
        y = le_local.fit_transform(sub_labs)

        for k in [5, 10, 20]:
            if n_samples <= k: continue
            knn = KNeighborsClassifier(n_neighbors=k)
            cv_k = 5 if n_samples > 50 else 3
            try:
                scores = cross_val_score(knn, sub_feats, y, cv=cv_k, scoring="accuracy")
                print(f"  k={k:2d}  |  {cv_k}-fold CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")
                result["scores"][f"k_{k}"] = {
                    "cv": int(cv_k),
                    "mean": float(scores.mean()),
                    "std": float(scores.std()),
                }
            except Exception as e:
                print(f"  k={k:2d}  |  Error: {e}")
                result["scores"][f"k_{k}"] = {"error": str(e)}

        return result

    # Run for ALL
    knn_all = run_knn_eval("ALL SAMPLES", np.ones(len(features), dtype=bool))
    if knn_all is not None:
        eval_metrics["knn"]["ALL_SAMPLES"] = knn_all

    # Run for each Split
    for s_name in ["train", "val", "test"]:
        mask = (all_splits_eval == s_name)
        if np.any(mask):
            split_key = f"SPLIT_{s_name.upper()}"
            knn_split = run_knn_eval(f"SPLIT: {s_name.upper()}", mask)
            if knn_split is not None:
                eval_metrics["knn"][split_key] = knn_split

    # ── 3. Retrieval Evaluation: Recall@K ────────────────────────
    ret_all_feat = print_recall_at_k("ALL FEATURES", features, all_labels_eval, np.ones(len(features), dtype=bool), run_tag=step_name)
    if ret_all_feat is not None:
        eval_metrics["retrieval"]["ALL_FEATURES"] = ret_all_feat
    ret_all_proj = print_recall_at_k("ALL PROJECTIONS", projections, all_labels_eval, np.ones(len(features), dtype=bool), run_tag=step_name)
    if ret_all_proj is not None:
        eval_metrics["retrieval"]["ALL_PROJECTIONS"] = ret_all_proj
    for s_name in ["train", "val", "test"]:
        mask = (all_splits_eval == s_name)
        if np.any(mask):
            split_feat = print_recall_at_k(f"{s_name.upper()} FEATURES", features, all_labels_eval, mask, run_tag=step_name)
            if split_feat is not None:
                eval_metrics["retrieval"][f"{s_name.upper()}_FEATURES"] = split_feat
            split_proj = print_recall_at_k(f"{s_name.upper()} PROJECTIONS", projections, all_labels_eval, mask, run_tag=step_name)
            if split_proj is not None:
                eval_metrics["retrieval"][f"{s_name.upper()}_PROJECTIONS"] = split_proj

    metrics_path = SAVE_DIR / f"metrics_{step_name}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)
    print(f"Evaluation metrics saved to: {metrics_path}")




# ╔══════════════════════════════════════════════════════════════╗
# ║  11. Training Loop                                           ║
# ╚══════════════════════════════════════════════════════════════╝

if IS_MAIN:
    # Run evaluation baseline BEFORE training
    run_evaluation(step_name="pre_train")

history = {"epoch": [], "loss": [], "lr": []}

if IS_MAIN:
    print(f"\nTraining for {args.epochs} epochs")
    print(f"  Per-GPU batch: {args.batch}  |  GPUs: {WORLD_SIZE}  |  GradAccum: {args.grad_accum}")
    print(f"  Effective batch: {args.batch * WORLD_SIZE * args.grad_accum}")
    print(f"  Negatives per sample: {2 * args.batch * WORLD_SIZE - 2}")
    print(f"  Batches/GPU/epoch: {len(dataloader)}")
    print("-" * 60)

start_time = time.time()

for epoch in range(1, args.epochs + 1):
    sampler.set_epoch(epoch)
    engine.train()

    epoch_loss = 0.0

    for step, (prompts1, prompts2) in enumerate(dataloader, 1):
        z1, z2 = engine(prompts1, prompts2)

        loss = criterion(z1, z2)

        engine.backward(loss)
        engine.step()
        scheduler.step()

        epoch_loss += loss.item()

        if IS_MAIN and step % 10 == 0:
            elapsed = time.time() - start_time
            frac = step / len(dataloader)
            est_total = elapsed / frac * args.epochs if frac > 0 else 0
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"  Epoch {epoch} | Step {step}/{len(dataloader)} "
                f"| Loss: {loss.item():.4f} | LR: {current_lr:.2e} "
                f"| Elapsed: {elapsed:.0f}s | ETA: {est_total - elapsed:.0f}s",
                flush=True,
            )

    avg_loss = epoch_loss / len(dataloader)

    # Reduce loss across ranks for logging
    loss_tensor = torch.tensor(avg_loss, device=DEVICE)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    avg_loss_global = loss_tensor.item()

    current_lr = scheduler.get_last_lr()[0]
    history["epoch"].append(epoch)
    history["loss"].append(avg_loss_global)
    history["lr"].append(current_lr)

    if IS_MAIN:
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {avg_loss_global:.4f} | LR: {current_lr:.2e} | Time: {elapsed:.1f}s")

total_time = time.time() - start_time
if IS_MAIN:
    print("-" * 60)
    print(f"Training complete in {total_time:.1f}s")
    print()


# ╔══════════════════════════════════════════════════════════════╗
# ║  12. Save Checkpoints (rank 0 only)                          ║
# ╚══════════════════════════════════════════════════════════════╝

if IS_MAIN:
    # Save projection head
    proj_path = SAVE_DIR / "projection_head.pt"
    torch.save({
        "state_dict": projection_head.state_dict(),
        "config": {
            "input_dim": HIDDEN_SIZE,
            "hidden_dim": HIDDEN_SIZE // 2,
            "output_dim": args.projection_dim,
        },
    }, str(proj_path))
    print(f"Projection head saved to: {proj_path}")

    # Save LoRA adapter
    if not args.freeze_backbone:
        adapter_save_path = SAVE_DIR / "gemma_lora_contrastive"
        gemma_model.save_pretrained(str(adapter_save_path))
        tokenizer.save_pretrained(str(adapter_save_path))
        print(f"LoRA adapter saved to: {adapter_save_path}")

    # Save training metadata
    meta_path = SAVE_DIR / "training_meta.pt"
    torch.save({
        "history": history,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "args": vars(args),
        "hidden_size": HIDDEN_SIZE,
        "num_classes": NUM_CLASSES,
    }, str(meta_path))
    print(f"Training metadata saved to: {meta_path}")
    print()

dist.barrier()


# ╔══════════════════════════════════════════════════════════════╗
# ║  13. Training Curves (rank 0 only)                           ║
# ╚══════════════════════════════════════════════════════════════╝

if IS_MAIN:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["epoch"], history["loss"], color="#2196F3", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Contrastive Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["epoch"], history["lr"], color="#FF9800", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Cosine Annealing LR Schedule")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    curves_path = SAVE_DIR / "training_curves.png"
    plt.savefig(str(curves_path), dpi=150, bbox_inches="tight")
    print(f"Training curves saved to: {curves_path}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  14. Evaluate: UMAP + k-NN (rank 0 only)                    ║
# ╚══════════════════════════════════════════════════════════════╝

if IS_MAIN:
    run_evaluation(step_name="post_train")


# ╔══════════════════════════════════════════════════════════════╗
# ║  Done                                                        ║
# ╚══════════════════════════════════════════════════════════════╝

dist.barrier()

if IS_MAIN:
    print("\n" + "=" * 60)
    print("  DONE — All artifacts saved to:", SAVE_DIR)
    print("=" * 60)
    print(f"\nTo load the contrastive-trained encoder for downstream use:")
    print(f"  base = AutoModelForCausalLM.from_pretrained('{BASE_MODEL_PATH}')")
    if not args.freeze_backbone:
        print(f"  model = PeftModel.from_pretrained(base, '{SAVE_DIR / 'gemma_lora_contrastive'}')")
    print(f"  # Then extract hidden states and use the projection head or raw representations")
