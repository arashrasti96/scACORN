import os
import sys
import time
import json
import math
import random
import subprocess
from pathlib import Path
from collections import Counter
from argparse import ArgumentParser

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# HF ecosystem
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    PeftModel,
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import deepspeed
from utils import (
    ProjectionHead,
    build_c2s_prompt,
    extract_genes_from_question,
    get_hidden_states,
    print_recall_at_k,
)

# ╔══════════════════════════════════════════════════════════════╗
# ║  1. Config & Setup                                           ║
# ╚══════════════════════════════════════════════════════════════╝

BASE_MODEL_PATH = os.getenv("GEMMA_MODEL_PATH", "vandijklab/C2S-Scale-Gemma-2-2B")
# Base DIR for context and relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DATASET_DIR = SCRIPT_DIR.parents[2] / "dataset" / "datasets" / "Sadra"

DATA_PATHS = {
    "train": DATASET_DIR / "immune1_celltype_train.jsonl",
    "val": DATASET_DIR / "immune1_celltype_val.jsonl",
    "test": DATASET_DIR / "immune1_celltype_test.jsonl",
}

CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
EVAL_OUT_DIR = SCRIPT_DIR / "eval_output"
EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

parser = ArgumentParser(description="Baseline Evaluation for Cell-Type Annotation")
parser.add_argument("--top-genes", type=int, default=200, help="Number of genes to keep per cell")
parser.add_argument("--batch", type=int, default=16, help="Eval Batch Size")
parser.add_argument("--max-seq-len", type=int, default=1024, help="Max sequence length for C2S prompts")
parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
parser.add_argument("--projection-dim", type=int, default=128, help="SimCLR projection head dimension")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")

args, _ = parser.parse_known_args()
USE_4BIT = not args.no_4bit
ORGANISM = "Homo sapiens"

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", getattr(args, "local_rank", 0)))
IS_MAIN = LOCAL_RANK == 0

torch.cuda.set_device(LOCAL_RANK)
DEVICE = torch.device("cuda", LOCAL_RANK)

random.seed(args.seed + LOCAL_RANK)
np.random.seed(args.seed + LOCAL_RANK)
torch.manual_seed(args.seed + LOCAL_RANK)
torch.cuda.manual_seed_all(args.seed + LOCAL_RANK)

# removed mkdir

# ╔══════════════════════════════════════════════════════════════╗
# ║  2. Load & Parse the Data                                    ║
# ╚══════════════════════════════════════════════════════════════╝

raw_data = []
split_tags = []
for split_name, data_path in DATA_PATHS.items():
    if not data_path.exists():
        if IS_MAIN: print(f"WARNING: Data split not found at {data_path}")
        continue
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_data.append(json.loads(line))
                split_tags.append(split_name)

genes_list = []
labels_list = []
for record in raw_data:
    genes = extract_genes_from_question(record["question"])[:args.top_genes]
    genes_list.append(genes)
    labels_list.append(record["answer"])

unique_labels = sorted(set(labels_list))
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
NUM_CLASSES = len(unique_labels)

# ╔══════════════════════════════════════════════════════════════╗
# ║  3. Prompt Builder                                           ║
# ╚══════════════════════════════════════════════════════════════╝

# ╔══════════════════════════════════════════════════════════════╗
# ║  4. Load Gemma 2B (Baseline)                                 ║
# ╚══════════════════════════════════════════════════════════════╝

if IS_MAIN: print("Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

if IS_MAIN: print("Loading base model ...")
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
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": LOCAL_RANK},
        attn_implementation="eager",
        trust_remote_code=True,
    )

base_model.config.use_cache = False
HIDDEN_SIZE = base_model.config.hidden_size

projection_head = ProjectionHead(
    input_dim=HIDDEN_SIZE,
    hidden_dim=HIDDEN_SIZE // 2,
    output_dim=args.projection_dim,
).to(DEVICE)
projection_head.eval()

# ╔══════════════════════════════════════════════════════════════╗
# ║  5. Extraction Evaluation Logic                              ║
# ╚══════════════════════════════════════════════════════════════╝

def evaluate_model(model, proj_head, step_name):
    print(f"\n[{step_name.upper()}] Extracting representations for all cells ...")
    model.eval()
    proj_head.eval()

    all_features = []
    all_proj = []
    all_labels_eval = []
    eval_metrics = {
        "step": step_name,
        "knn": {},
        "retrieval": {},
    }

    EVAL_BATCH = args.batch
    with torch.no_grad():
        for i in range(0, len(genes_list), EVAL_BATCH):
            batch_genes = genes_list[i : i + EVAL_BATCH]
            batch_labels = labels_list[i : i + EVAL_BATCH]
            prompts = [build_c2s_prompt(g, organism=ORGANISM) for g in batch_genes]

            h = get_hidden_states(model, tokenizer, prompts, args.max_seq_len, enable_grad=False)
            z = proj_head(h)

            all_features.append(h.float().cpu().numpy())
            all_proj.append(z.float().cpu().numpy())
            all_labels_eval.extend(batch_labels)

            if (i // EVAL_BATCH) % 50 == 0:
                print(f"  {i}/{len(genes_list)} ...")

    if len(all_features) == 0:
        print(f"[{step_name.upper()}] No samples were loaded. Skipping evaluation.")
        return

    features = np.concatenate(all_features, axis=0)
    projections = np.concatenate(all_proj, axis=0)

    all_labels_eval_arr = np.array(all_labels_eval)
    all_splits_eval = np.array(split_tags[:len(all_labels_eval)])

    print(f"[{step_name.upper()}] Extracted representations: {features.shape}")
    print(f"[{step_name.upper()}] Extracted projections:     {projections.shape}")

    np.savez(
        str(EVAL_OUT_DIR / f"representations_{step_name}.npz"),
        features=features,
        projections=projections,
        labels=all_labels_eval_arr,
        splits=all_splits_eval
    )
    print(f"Representations saved to: {EVAL_OUT_DIR / f'representations_{step_name}.npz'}")

    try:
        from umap import UMAP
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "umap-learn", "-q"])
        from umap import UMAP

    print(f"[{step_name.upper()}] Computing UMAP embedding (all samples) ...")
    reducer = UMAP(n_components=2, random_state=args.seed, n_neighbors=30, min_dist=0.3)
    embedding = reducer.fit_transform(features)

    def plot_umap(mask, title_suffix, filename_suffix):
        n_samples = np.sum(mask)
        if n_samples == 0:
            return

        fig, ax = plt.subplots(figsize=(14, 10))
        subset_labels = all_labels_eval_arr[mask]
        subset_embedding = embedding[mask]

        unique_types_sub = sorted(set(subset_labels))
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

        ax.set_title(f"UMAP: {title_suffix} ({n_samples} samples) [{step_name.upper()}]", fontsize=14)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7, markerscale=3)
        plt.tight_layout()

        out_path = EVAL_OUT_DIR / f"umap_{step_name}_{filename_suffix}.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        print(f"UMAP plot saved to: {out_path}")
        plt.close(fig)

    plot_umap(np.ones(len(features), dtype=bool), "All Samples", "all")
    for s_name in ["train", "val", "test"]:
        mask = (all_splits_eval == s_name)
        if np.any(mask):
            plot_umap(mask, f"{s_name.upper()} Split", s_name)

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

        print(f"\n--- k-NN Evaluation [{step_name.upper()}]: {name} ({n_samples} samples) ---")
        sub_feats = features[mask]
        sub_labs = all_labels_eval_arr[mask]

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

    knn_all = run_knn_eval("ALL SAMPLES", np.ones(len(features), dtype=bool))
    if knn_all is not None:
        eval_metrics["knn"]["ALL_SAMPLES"] = knn_all
    for s_name in ["train", "val", "test"]:
        mask = (all_splits_eval == s_name)
        if np.any(mask):
            split_key = f"SPLIT_{s_name.upper()}"
            knn_split = run_knn_eval(f"SPLIT: {s_name.upper()}", mask)
            if knn_split is not None:
                eval_metrics["knn"][split_key] = knn_split

    ret_all_feat = print_recall_at_k("ALL FEATURES", features, all_labels_eval_arr, np.ones(len(features), dtype=bool), run_tag=step_name.upper())
    if ret_all_feat is not None:
        eval_metrics["retrieval"]["ALL_FEATURES"] = ret_all_feat
    ret_all_proj = print_recall_at_k("ALL PROJECTIONS", projections, all_labels_eval_arr, np.ones(len(features), dtype=bool), run_tag=step_name.upper())
    if ret_all_proj is not None:
        eval_metrics["retrieval"]["ALL_PROJECTIONS"] = ret_all_proj
    for s_name in ["train", "val", "test"]:
        mask = (all_splits_eval == s_name)
        if np.any(mask):
            split_feat = print_recall_at_k(f"{s_name.upper()} FEATURES", features, all_labels_eval_arr, mask, run_tag=step_name.upper())
            if split_feat is not None:
                eval_metrics["retrieval"][f"{s_name.upper()}_FEATURES"] = split_feat
            split_proj = print_recall_at_k(f"{s_name.upper()} PROJECTIONS", projections, all_labels_eval_arr, mask, run_tag=step_name.upper())
            if split_proj is not None:
                eval_metrics["retrieval"][f"{s_name.upper()}_PROJECTIONS"] = split_proj

    metrics_path = EVAL_OUT_DIR / f"metrics_{step_name}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)
    print(f"Evaluation metrics saved to: {metrics_path}")

if IS_MAIN:
    print("\n" + "="*80)
    print(" 1. EVALUATING BASELINE (Untrained base model + random projection)")
    print("="*80)
    evaluate_model(base_model, projection_head, "baseline")

    print("\n" + "="*80)
    print(" 2. EVALUATING TRAINED MODEL")
    print("="*80)

    # Load LoRA adapter
    lora_path = CHECKPOINT_DIR / "gemma_lora_contrastive"
    if lora_path.exists():
        print(f"Loading trained LoRA adapter from {lora_path} ...")
        trained_model = PeftModel.from_pretrained(base_model, str(lora_path))
        trained_model.eval()
    else:
        print(f"WARNING: LoRA path {lora_path} does not exist! Skipping trained evaluation.")
        sys.exit(0)

    # Load projection head
    proj_path = CHECKPOINT_DIR / "projection_head.pt"
    if proj_path.exists():
        print(f"Loading trained projection head from {proj_path} ...")
        checkpoint = torch.load(str(proj_path), map_location=DEVICE)
        projection_head.load_state_dict(checkpoint["state_dict"])
    else:
        print(f"WARNING: Projection head {proj_path} does not exist!")

    evaluate_model(trained_model, projection_head, "trained")

    print("\nDONE: Both evaluations complete. All outputs in:", EVAL_OUT_DIR)
