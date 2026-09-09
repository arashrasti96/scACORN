#!/usr/bin/env python
"""
Stage-2 task adapter training for C2S-Scale-Gemma-2-2B.

This script keeps the setup completion-style, loads the stage-1 domain adapter
for initialization, and trains a separate late-layer LoRA for curated QnA tasks.

How to run (example):
deepspeed --num_gpus 4 \
    stage2_task_adapter/train_stage2_task_adapter.py \
    --train-data dataset_pipeline/data/exports/example/task_train.jsonl \
    --val-data dataset_pipeline/data/exports/example/task_val.jsonl \
    --stage1-adapter stage1_domain_adapter/outputs/example/checkpoints/best/gemma_lora_stage1 \
    --output-dir stage2_task_adapter/outputs/example \
  --use-deepspeed \
  --zero-stage 2 \
  --epochs 50 \
  --lr 1e-4 \
  --batch 4 \
  --eval-batch 4 \
  --grad-accum 4 \
  --max-seq-len 2048 \
  --layer-strategy last_third \
  --replay-ratio 0.0 \
  --replay-val-ratio 0.0 \
  --no-4bit

"""

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.utils import SAFETENSORS_WEIGHTS_NAME, get_peft_model_state_dict
from safetensors.torch import save_file as save_safetensors
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from utils import (  # noqa: E402
    STRUCTURED_REASONING_ANSWER_FORMAT,
    STRUCTURED_REASONING_QUESTION_TEXT,
    build_c2s_prompt,
    clean_reasoning_dataset_sample,
    extract_genes_from_question,
    is_reasoning_dataset_sample,
    print_recall_at_k,
)


TASK_DESCRIPTIONS = {
    "cell_type": "Predict the most likely cell type from the expression profile.",
    "marker_evidence": "Identify the gene-expression evidence supporting the annotation.",
    "differential_diagnosis": "Choose the best diagnosis among candidate cell identities.",
    "state_classification": "Infer the most likely biological state from the expression profile.",
    "tissue_inference": "Infer the most likely tissue or compartment from the expression profile.",
    "perturbation_inference": "Infer the most likely perturbation or pathway activity from the expression profile.",
    "cluster_caption": "Write a short grounded biological description of the cell profile.",
    "freeform_qa": "Answer the grounded biology question using the cell sentence.",
    "domain_replay": "Predict the most likely cell type from the expression profile.",
}

STAGE1_ADAPTER_NAME = "stage1_domain"
STAGE2_ADAPTER_NAME = "default"

def strip_supporting_evidence_section(answer_text: str) -> str:
    lines = []
    for raw_line in str(answer_text).splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("EVIDENCE:") or stripped.startswith("POSITIVE_MARKERS:"):
            continue
        lines.append(raw_line)
    return "\n".join(lines).strip()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stage-2 completion-style task adapter")
    parser.add_argument("--train-data", type=str, required=True, help="Stage-2 train JSONL")
    parser.add_argument("--val-data", type=str, required=True, help="Stage-2 val JSONL")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--stage1-adapter", type=str, default=None, help="Path to the stage-1 adapter directory")
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.getenv("GEMMA_MODEL_PATH", "vandijklab/C2S-Scale-Gemma-2-2B"),
        help="Gemma base model path",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch", type=int, default=4, help="Per-device train batch size")
    parser.add_argument("--eval-batch", type=int, default=4, help="Per-device eval batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--top-genes", type=int, default=200, help="Genes to keep per sample")
    parser.add_argument("--lora-r", type=int, default=16, help="Stage-2 LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="Stage-2 LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="Stage-2 LoRA dropout")
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated target modules",
    )
    parser.add_argument(
        "--layer-strategy",
        type=str,
        choices=["last_third", "upper_half", "all"],
        default="last_third",
        help="Default stage-2 layer coverage",
    )
    parser.add_argument("--layer-start", type=int, default=None, help="Inclusive start layer override")
    parser.add_argument("--layer-end", type=int, default=None, help="Exclusive end layer override")
    parser.add_argument("--replay-ratio", type=float, default=0.0, help="Replay examples as fraction of task train size")
    parser.add_argument("--replay-val-ratio", type=float, default=0.0, help="Replay examples as fraction of task val size")
    parser.add_argument(
        "--domain-train-path",
        type=str,
        default=None,
        help="Optional override for replay train dataset path",
    )
    parser.add_argument(
        "--domain-val-path",
        type=str,
        default=None,
        help="Optional override for replay val dataset path",
    )
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument(
        "--exclude-evidence-in-ground-truth",
        "--exclude-positive-markers-in-ground-truth",
        dest="exclude_evidence_in_ground_truth",
        action="store_true",
        help="Strip EVIDENCE lines, and any legacy POSITIVE_MARKERS lines, from structured target answers before SFT formatting",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank injected by launcher")
    parser.add_argument("--use-deepspeed", action="store_true", help="Enable DeepSpeed training")
    parser.add_argument("--zero-stage", type=int, choices=[0, 1, 2, 3], default=2, help="DeepSpeed ZeRO stage")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


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


def build_deepspeed_config(args: argparse.Namespace, world_size: int) -> dict:
    return {
        "train_micro_batch_size_per_gpu": args.batch,
        "train_batch_size": args.batch * max(1, world_size) * args.grad_accum,
        "gradient_accumulation_steps": args.grad_accum,
        "gradient_clipping": 1.0,
        "bf16": {"enabled": True},
        "fp16": {"enabled": False},
        "zero_optimization": {
            "stage": args.zero_stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e7,
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e5,
        },
    }


def load_base_model(model_path: str, use_4bit: bool, use_deepspeed: bool, local_rank: int):
    model_load_kwargs = {
        "attn_implementation": "eager",
        "trust_remote_code": True,
    }
    if use_deepspeed:
        model_load_kwargs["device_map"] = {"": local_rank}
    else:
        model_load_kwargs["device_map"] = "auto"

    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            **model_load_kwargs,
        )
        model = prepare_model_for_kbit_training(model)
        return model

    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        **model_load_kwargs,
    )


def build_stage2_peft_model(base_model, lora_config: LoraConfig, mixed: bool):
    # Plain PeftModel already supports loading and activating multiple LoRA adapters.
    # Avoid PeftMixedModel here because this environment's PEFT build rejects LoRA
    # mixed-mode construction on 4-bit / 8-bit base models.
    return get_peft_model(base_model, lora_config, adapter_name=STAGE2_ADAPTER_NAME, mixed=False)


def activate_adapters(model, adapter_names: str | list[str]) -> None:
    if isinstance(adapter_names, str):
        model.set_adapter(adapter_names)
        return

    base_model = getattr(model, "base_model", None)
    if base_model is None or not hasattr(base_model, "set_adapter"):
        raise TypeError("This PEFT model does not support activating multiple adapters")
    base_model.set_adapter(adapter_names)


def compose_stage_adapters(model, args: argparse.Namespace):
    composition_mode = "stage2_only"
    if not args.stage1_adapter:
        activate_adapters(model, STAGE2_ADAPTER_NAME)
        return model, composition_mode

    print("Loading frozen stage-1 adapter …")
    model.load_adapter(args.stage1_adapter, adapter_name=STAGE1_ADAPTER_NAME, is_trainable=False)
    activate_adapters(model, get_active_adapter_names(include_stage1=True))
    return model, "mixed"


def save_stage2_adapter(model, adapter_dir: Path) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    state_dict = get_peft_model_state_dict(model, adapter_name=STAGE2_ADAPTER_NAME)
    save_safetensors(state_dict, str(adapter_dir / SAFETENSORS_WEIGHTS_NAME))
    model.peft_config[STAGE2_ADAPTER_NAME].save_pretrained(str(adapter_dir))


def load_jsonl(path: Path) -> list[dict]:
    records = []
    dropped_invalid_reasoning = 0
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                record = json.loads(line)
                cleaned_record = clean_reasoning_dataset_sample(record)
                if cleaned_record is None:
                    dropped_invalid_reasoning += 1
                    continue
                records.append(cleaned_record)
    if dropped_invalid_reasoning:
        print(f"[Clean] Dropped {dropped_invalid_reasoning} invalid reasoning samples from {path}")
    return records


def load_retrieval_split(path: Path, top_genes: int) -> dict[str, list]:
    genes_list = []
    labels_list = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            genes = extract_genes_from_question(record["question"])[:top_genes]
            if not genes:
                continue
            genes_list.append(genes)
            labels_list.append(str(record["answer"]).strip())
    return {"genes": genes_list, "labels": labels_list}


def get_last_token_hidden_states(model, tokenizer, prompts: list[str], max_len: int) -> torch.Tensor:
    device = getattr(model, "device", None)
    if device is None and hasattr(model, "module"):
        device = getattr(model.module, "device", None)
    if device is None:
        device = next(model.parameters()).device

    model_config = getattr(model, "config", None)
    if model_config is None and hasattr(model, "module"):
        model_config = getattr(model.module, "config", None)

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    ).to(device)

    forward_kwargs = {
        "output_hidden_states": True,
    }
    model_type = getattr(model_config, "model_type", None)
    if model_type in {"gemma2", "gemma"}:
        forward_kwargs["logits_to_keep"] = 1

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(**inputs, **forward_kwargs)

    last_hidden = outputs.hidden_states[-1]
    attention_mask = inputs["attention_mask"]
    last_token_pos = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(last_hidden.size(0), device=last_hidden.device)
    return last_hidden[batch_indices, last_token_pos]


class RetrievalEvalSFTTrainer(SFTTrainer):
    def __init__(
        self,
        *args,
        retrieval_eval_data: dict[str, list] | None = None,
        retrieval_output_dir: Path | None = None,
        retrieval_max_seq_len: int = 2048,
        retrieval_batch_size: int = 4,
        retrieval_ks: tuple[int, ...] = (1, 5, 10),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.retrieval_eval_data = retrieval_eval_data
        self.retrieval_output_dir = retrieval_output_dir
        self.retrieval_max_seq_len = retrieval_max_seq_len
        self.retrieval_batch_size = retrieval_batch_size
        self.retrieval_ks = retrieval_ks
        self.retrieval_history: list[dict] = []
        if self.retrieval_output_dir is not None:
            self.retrieval_output_dir.mkdir(parents=True, exist_ok=True)

    def _compute_domain_retrieval_metrics(self, split_name: str = "val") -> dict | None:
        if not self.retrieval_eval_data or not self.retrieval_eval_data.get("genes"):
            return None
        if not self.is_world_process_zero():
            return None

        wrapped_model = self.model_wrapped if self.model_wrapped is not None else self.model
        tokenizer = self.processing_class
        was_training = wrapped_model.training
        wrapped_model.eval()

        genes = self.retrieval_eval_data["genes"]
        labels = np.array(self.retrieval_eval_data["labels"])
        all_features = []

        for start in range(0, len(genes), self.retrieval_batch_size):
            batch_genes = genes[start : start + self.retrieval_batch_size]
            prompts = [build_c2s_prompt(sample_genes) for sample_genes in batch_genes]
            hidden = get_last_token_hidden_states(wrapped_model, tokenizer, prompts, self.retrieval_max_seq_len)
            all_features.append(hidden.float().cpu().numpy())

        if was_training:
            wrapped_model.train()

        features = np.concatenate(all_features, axis=0)
        mask = np.ones(len(labels), dtype=bool)
        retrieval = print_recall_at_k(
            f"{split_name.upper()} HIDDEN",
            features,
            labels,
            mask,
            run_tag=f"stage2_{split_name}",
            ks=self.retrieval_ks,
        )
        if retrieval is None:
            return None

        record = {
            "epoch": None if self.state.epoch is None else float(self.state.epoch),
            "global_step": int(self.state.global_step),
            "split": split_name,
            "n_samples": int(len(labels)),
            "retrieval": retrieval,
        }
        return record

    def run_domain_retrieval_eval(self, tag: str) -> dict[str, float]:
        record = self._compute_domain_retrieval_metrics(split_name="val")
        if record is None:
            return {}

        record["tag"] = tag
        self.retrieval_history.append(record)

        if self.retrieval_output_dir is not None:
            safe_tag = tag.replace("/", "_")
            with open(self.retrieval_output_dir / f"{safe_tag}.json", "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2)
            with open(self.retrieval_output_dir / "history.json", "w", encoding="utf-8") as handle:
                json.dump(self.retrieval_history, handle, indent=2)

        flat_metrics = {}
        for key, value in record["retrieval"]["recall"].items():
            if value is not None:
                flat_metrics[f"eval_hidden_{key}"] = float(value)
        if flat_metrics:
            self.log(flat_metrics)
        return flat_metrics

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)
        epoch_value = 0.0 if self.state.epoch is None else float(self.state.epoch)
        tag = f"epoch_{epoch_value:06.2f}_step_{int(self.state.global_step)}"
        retrieval_metrics = self.run_domain_retrieval_eval(tag=tag)
        metrics.update(retrieval_metrics)
        return metrics


def extract_genes(sample: dict, top_genes: int) -> list[str]:
    if "genes" in sample and isinstance(sample["genes"], list):
        genes = [str(gene) for gene in sample["genes"]]
    elif "cell_sentence" in sample and isinstance(sample["cell_sentence"], str):
        genes = sample["cell_sentence"].strip().split()
    elif "question" in sample and isinstance(sample["question"], str):
        genes = extract_genes_from_question(sample["question"])
    else:
        raise ValueError("Each sample must define genes, cell_sentence, or question")
    genes = [gene for gene in genes if gene][:top_genes]
    if not genes:
        raise ValueError("Sample does not contain a usable gene sequence")
    return genes


def build_general_task_prompt(genes: list[str], task_type: str, question_text: str, sample: dict, organism: str) -> str:
    cell_sentence = " ".join(genes)
    num_genes = len(genes)
    task_instruction = sample.get("task_instruction") or TASK_DESCRIPTIONS[task_type]
    reasoning_sample = task_type == "freeform_qa" and is_reasoning_dataset_sample(sample)
    prompt_parts = [
        (
            f"The following is a list of {num_genes} gene names ordered by descending expression level "
            f"in a {organism} cell."
        ),
        f"Cell sentence: {cell_sentence}.",
        f"Task: {task_instruction}",
    ]

    if sample.get("candidate_labels"):
        prompt_parts.append("Candidate labels: " + ", ".join(sample["candidate_labels"]))
    if sample.get("predicted_label"):
        prompt_parts.append(f"Predicted label: {sample['predicted_label']}")
    if sample.get("evidence_genes"):
        prompt_parts.append("Relevant genes: " + ", ".join(sample["evidence_genes"]))
    if sample.get("context"):
        prompt_parts.append(f"Context: {sample['context']}")
    if reasoning_sample:
        prompt_parts.append("Answer format:\n" + sample.get("answer_format", STRUCTURED_REASONING_ANSWER_FORMAT))

    prompt_parts.append(f"Question: {question_text}")
    prompt_parts.append("Answer:")
    return "\n".join(prompt_parts)


def format_task_sample(
    sample: dict,
    top_genes: int,
    organism: str,
    exclude_evidence_in_ground_truth: bool = False,
) -> dict:
    task_type = sample.get("task_type", "freeform_qa")
    if task_type not in TASK_DESCRIPTIONS:
        raise ValueError(f"Unsupported task_type: {task_type}")

    genes = extract_genes(sample, top_genes)
    answer = str(sample["answer"]).strip()
    if exclude_evidence_in_ground_truth:
        answer = strip_supporting_evidence_section(answer)
    if not answer:
        raise ValueError("Sample answer is empty")

    if task_type in {"cell_type", "domain_replay"}:
        prompt = build_c2s_prompt(genes)
    else:
        default_questions = {
            "marker_evidence": "Which evidence in the gene expression profile supports the annotation?",
            "differential_diagnosis": "Which candidate label best matches the cell sentence?",
            "state_classification": "What is the most likely biological state of this cell?",
            "tissue_inference": "What tissue or compartment is most consistent with this expression profile?",
            "perturbation_inference": "What perturbation or pathway activity is most consistent with this expression profile?",
            "cluster_caption": "Provide a short biological description of this cell profile.",
            "freeform_qa": STRUCTURED_REASONING_QUESTION_TEXT,
            "domain_replay": "What is the most likely cell type for this cell?",
        }
        question_text = sample.get("question_text") or default_questions[task_type]
        prompt = build_general_task_prompt(genes, task_type, question_text, sample, organism)

    return {
        "prompt": prompt,
        "completion": f" {answer}",
        "task_type": task_type,
        "sample_id": sample.get("sample_id"),
    }


def build_replay_samples(path: Path, sample_count: int, top_genes: int, seed: int) -> list[dict]:
    if sample_count <= 0:
        return []
    raw = load_jsonl(path)
    generator = random.Random(seed)
    if len(raw) > sample_count:
        raw = generator.sample(raw, sample_count)
    replay_samples = []
    for record in raw:
        replay_samples.append(
            {
                "task_type": "domain_replay",
                "question": record["question"],
                "answer": record["answer"],
                "sample_id": record.get("id"),
            }
        )
    return replay_samples


def configure_stage2_trainability(model) -> None:
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            parameter.requires_grad = STAGE2_ADAPTER_NAME in name


def get_active_adapter_names(include_stage1: bool) -> list[str]:
    if include_stage1:
        # PEFT best-checkpoint reload targets the first active adapter.
        return [STAGE2_ADAPTER_NAME, STAGE1_ADAPTER_NAME]
    return [STAGE2_ADAPTER_NAME]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", getattr(args, "local_rank", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_deepspeed = bool(args.use_deepspeed)
    if use_deepspeed and world_size < 1:
        world_size = 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    organism = "Homo sapiens"
    use_4bit = not args.no_4bit

    dataset_dir = SCRIPT_DIR.parents[3] / "dataset" / "datasets" / "Sadra"
    domain_train_path = Path(args.domain_train_path) if args.domain_train_path else dataset_dir / "immune1_celltype_train.jsonl"
    domain_val_path = Path(args.domain_val_path) if args.domain_val_path else dataset_dir / "immune1_celltype_val.jsonl"

    train_samples = load_jsonl(Path(args.train_data))
    val_samples = load_jsonl(Path(args.val_data))
    retrieval_val_split = load_retrieval_split(domain_val_path, args.top_genes)

    replay_train_count = int(round(len(train_samples) * max(0.0, args.replay_ratio)))
    replay_val_count = int(round(len(val_samples) * max(0.0, args.replay_val_ratio)))
    train_samples.extend(build_replay_samples(domain_train_path, replay_train_count, args.top_genes, args.seed))
    val_samples.extend(build_replay_samples(domain_val_path, replay_val_count, args.top_genes, args.seed + 1))

    formatted_train = [format_task_sample(sample, args.top_genes, organism) for sample in train_samples]
    formatted_train = [
        format_task_sample(
            sample,
            args.top_genes,
            organism,
            exclude_evidence_in_ground_truth=args.exclude_evidence_in_ground_truth,
        )
        for sample in train_samples
    ]
    formatted_val = [
        format_task_sample(
            sample,
            args.top_genes,
            organism,
            exclude_evidence_in_ground_truth=args.exclude_evidence_in_ground_truth,
        )
        for sample in val_samples
    ]
    random.Random(args.seed).shuffle(formatted_train)

    # Save sample examples for inspection
    if is_main_process():
        example_samples = formatted_train[:20]
        examples_path = output_dir / "sample_completions.json"
        with open(examples_path, "w", encoding="utf-8") as handle:
            json.dump(example_samples, handle, indent=2, ensure_ascii=False)
        print(f"[Debug] Saved {len(example_samples)} sample completions to {examples_path}")

    train_dataset = Dataset.from_list(formatted_train)
    val_dataset = Dataset.from_list(formatted_val)

    task_counts = Counter(sample["task_type"] for sample in formatted_train)
    val_task_counts = Counter(sample["task_type"] for sample in formatted_val)

    if is_main_process():
        print(f"[Config] Model        : {args.model_path}")
        print(f"[Config] Train data   : {args.train_data}")
        print(f"[Config] Val data     : {args.val_data}")
        print(f"[Config] Output       : {output_dir}")
        print(f"[Config] Stage1       : {args.stage1_adapter}")
        print(f"[Config] DeepSpeed    : {use_deepspeed}")
        print(f"[Config] World size   : {world_size}")
        print(f"[Config] Replay ratio : train={args.replay_ratio} val={args.replay_val_ratio}")
        print(f"[Config] Retrieval val: {domain_val_path} ({len(retrieval_val_split['genes'])} samples)")
        print(f"[Config] Epochs       : {args.epochs}  LR: {args.lr}")
        print(f"[Config] Batch×Accum  : {args.batch} × {args.grad_accum}")
        print(f"[Config] Max seq len  : {args.max_seq_len}")
        print(f"[Config] 4-bit QLoRA  : {use_4bit}")
        print(f"[Config] Train size   : {len(train_dataset)}")
        print(f"[Config] Val size     : {len(val_dataset)}")
        print(f"[Config] Train tasks  : {dict(task_counts)}")
        print(f"[Config] Val tasks    : {dict(val_task_counts)}")
        print()

    print("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading model …")
    model = load_base_model(args.model_path, use_4bit, use_deepspeed, local_rank)

    num_hidden_layers = int(model.config.num_hidden_layers)
    layers_to_transform = resolve_target_layers(num_hidden_layers, args)
    target_modules = [name.strip() for name in args.lora_target_modules.split(",") if name.strip()]

    lora_kwargs = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": target_modules,
    }
    if layers_to_transform is not None:
        lora_kwargs["layers_to_transform"] = layers_to_transform
        lora_kwargs["layers_pattern"] = "layers"
    lora_config = LoraConfig(**lora_kwargs)

    print("Attaching stage-2 task adapter …")
    model = build_stage2_peft_model(model, lora_config, mixed=bool(args.stage1_adapter))
    model, composition_mode = compose_stage_adapters(model, args)
    configure_stage2_trainability(model)
    model.config.use_cache = False
    trainable, total = model.get_nb_trainable_parameters()
    if is_main_process():
        print(f"Trainable params: {trainable:,} ({100 * trainable / total:.2f}% of {total:,})")
        print(f"Target modules : {target_modules}")
        if layers_to_transform is None:
            print("Target layers  : all")
        else:
            print(f"Target layers  : {layers_to_transform[0]}..{layers_to_transform[-1]} ({len(layers_to_transform)} blocks)")
        print()
        print(f"Active adapters: {model.active_adapters}")
        print(f"Adapter composition : {composition_mode}")
        print(f"Stage-2 trainable adapter: {STAGE2_ADAPTER_NAME}")
        if args.stage1_adapter:
            print(f"Frozen stage-1 adapter  : {STAGE1_ADAPTER_NAME}")
        print()

    optimizer_name = "adamw_torch" if use_deepspeed else "paged_adamw_8bit"
    deepspeed_config = build_deepspeed_config(args, world_size) if use_deepspeed else None

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.eval_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=False,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=optimizer_name,
        report_to="none",
        seed=args.seed,
        dataloader_pin_memory=True,
        remove_unused_columns=True,
        max_length=args.max_seq_len,
        completion_only_loss=True,
        packing=False,
        ddp_find_unused_parameters=False,
        deepspeed=deepspeed_config,
    )

    retrieval_output_dir = output_dir / "retrieval_metrics"

    trainer = RetrievalEvalSFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        processing_class=tokenizer,
        retrieval_eval_data=retrieval_val_split,
        retrieval_output_dir=retrieval_output_dir,
        retrieval_max_seq_len=args.max_seq_len,
        retrieval_batch_size=args.eval_batch,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    if is_main_process():
        trainer.run_domain_retrieval_eval(tag="pre_train")

    print("Training stage-2 task adapter …")
    train_result = trainer.train()
    print("Training complete.")

    adapter_dir = output_dir / "adapter"
    if is_main_process():
        save_stage2_adapter(trainer.model, adapter_dir)
        tokenizer.save_pretrained(str(adapter_dir))

    train_metrics = train_result.metrics
    eval_metrics = trainer.evaluate()
    trainer.save_state()

    manifest = {
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "train_task_counts": dict(task_counts),
        "val_task_counts": dict(val_task_counts),
        "replay_train_count": replay_train_count,
        "replay_val_count": replay_val_count,
        "retrieval_val_path": str(domain_val_path),
        "retrieval_val_samples": len(retrieval_val_split["genes"]),
        "target_modules": target_modules,
        "layers_to_transform": layers_to_transform,
        "stage1_adapter": args.stage1_adapter,
        "stage1_adapter_name": STAGE1_ADAPTER_NAME if args.stage1_adapter else None,
        "stage2_adapter_name": STAGE2_ADAPTER_NAME,
        "adapter_composition": composition_mode,
        "separate_adapter_training": True,
        "args": vars(args),
    }
    if is_main_process():
        with open(output_dir / "task_manifest.json", "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    summary = {
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "active_adapters": trainer.model.active_adapters,
        "domain_retrieval_history": trainer.retrieval_history,
        "log_history": trainer.state.log_history,
    }
    if is_main_process():
        with open(output_dir / "training_summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    if is_main_process() and len(val_dataset) > 0:
        prompt = val_dataset[0]["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_seq_len).to(model.device)
        model.eval()
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=192, do_sample=False)
        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        sanity_payload = {
            "prompt": prompt,
            "target_completion": val_dataset[0]["completion"],
            "generated_completion": generated,
            "task_type": val_dataset[0]["task_type"],
        }
        with open(output_dir / "sanity_generation.json", "w", encoding="utf-8") as handle:
            json.dump(sanity_payload, handle, indent=2)

    if is_main_process():
        print(f"Adapter saved to: {adapter_dir}")
        print(f"Best checkpoint : {trainer.state.best_model_checkpoint}")


if __name__ == "__main__":
    main()