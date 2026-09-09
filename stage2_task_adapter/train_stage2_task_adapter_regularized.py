#!/usr/bin/env python
"""
Stage-2 task adapter training with stage-1 embedding-preservation regularization.

This variant keeps the completion-style stage-2 objective, but adds a hidden-state
regularizer that anchors the prompt embedding geometry to the frozen stage-1 adapter.
The goal is to preserve the retrieval-style separation learned in stage 1 while the
stage-2 late-layer adapter learns downstream reasoning tasks.

How to run (example):
deepspeed --num_gpus 4 \
    stage2_task_adapter/train_stage2_task_adapter_regularized.py \
    --train-data dataset_pipeline/data/exports/example/task_train.jsonl \
    --val-data dataset_pipeline/data/exports/example/task_val.jsonl \
    --stage1-adapter stage1_domain_adapter/outputs/example/checkpoints/best/gemma_lora_stage1 \
    --output-dir stage2_task_adapter/outputs/example_regularized \
  --use-deepspeed \
  --zero-stage 2 \
  --epochs 50 \
  --lr 1e-4 \
  --batch 4 \
  --eval-batch 4 \
  --grad-accum 4 \
  --max-seq-len 2048 \
  --layer-strategy last_third \
  --embedding-anchor-weight 0.6 \
  --embedding-distribution-weight 0.6 \
  --embedding-relation-weight 0.6 \
  --replay-ratio 0.0 \
  --replay-val-ratio 0.0 \
  --no-4bit
"""

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, EarlyStoppingCallback
from trl import SFTConfig

from train_stage2_task_adapter import (
    STAGE1_ADAPTER_NAME,
    STAGE2_ADAPTER_NAME,
    STRUCTURED_REASONING_QUESTION_TEXT,
    TASK_DESCRIPTIONS,
    activate_adapters,
    RetrievalEvalSFTTrainer,
    build_c2s_prompt,
    build_deepspeed_config,
    build_general_task_prompt,
    build_replay_samples,
    build_stage2_peft_model,
    compose_stage_adapters,
    configure_stage2_trainability,
    extract_genes,
    get_active_adapter_names,
    is_main_process,
    load_base_model,
    load_jsonl,
    load_retrieval_split,
    resolve_target_layers,
    save_stage2_adapter,
    set_seed,
    strip_supporting_evidence_section,
)
from peft import LoraConfig


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stage-2 task adapter with stage-1 embedding regularization")
    parser.add_argument("--train-data", type=str, required=True, help="Stage-2 train JSONL")
    parser.add_argument("--val-data", type=str, required=True, help="Stage-2 val JSONL")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--stage1-adapter",
        type=str,
        required=True,
        help="Path to the stage-1 adapter directory used as the frozen reference space",
    )
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
    parser.add_argument("--embedding-anchor-weight", type=float, default=0.05, help="Weight for per-sample cosine anchor loss")
    parser.add_argument(
        "--embedding-distribution-weight",
        type=float,
        default=0.05,
        help="Weight for batch mean/variance alignment in embedding space",
    )
    parser.add_argument(
        "--embedding-relation-weight",
        type=float,
        default=0.05,
        help="Weight for pairwise similarity-structure preservation",
    )
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank injected by launcher")
    parser.add_argument("--use-deepspeed", action="store_true", help="Enable DeepSpeed training")
    parser.add_argument("--zero-stage", type=int, choices=[0, 1, 2, 3], default=2, help="DeepSpeed ZeRO stage")
    return parser.parse_args()


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


class Stage1EmbeddingRegularizedTrainer(RetrievalEvalSFTTrainer):
    def __init__(
        self,
        *args,
        embedding_anchor_weight: float,
        embedding_distribution_weight: float,
        embedding_relation_weight: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.embedding_anchor_weight = embedding_anchor_weight
        self.embedding_distribution_weight = embedding_distribution_weight
        self.embedding_relation_weight = embedding_relation_weight
        self._last_reg_log_step = -1

    @staticmethod
    def _filter_model_inputs(inputs: dict, include_labels: bool) -> dict:
        allowed = {"input_ids", "attention_mask", "position_ids"}
        if include_labels:
            allowed.add("labels")
        return {
            key: value
            for key, value in inputs.items()
            if key in allowed and torch.is_tensor(value)
        }

    @staticmethod
    def _prompt_hidden_from_outputs(outputs, labels: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = outputs.hidden_states[-1]
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_token_positions = attention_mask.sum(dim=1) - 1

        if labels is None:
            return hidden_states[batch_indices, last_token_positions]

        completion_mask = labels.ne(-100)
        has_completion = completion_mask.any(dim=1)
        first_completion_positions = completion_mask.int().argmax(dim=1)
        prompt_positions = torch.where(
            has_completion & (first_completion_positions > 0),
            first_completion_positions - 1,
            last_token_positions,
        )
        return hidden_states[batch_indices, prompt_positions]

    @staticmethod
    def _pairwise_relation_loss(current_embeddings: torch.Tensor, reference_embeddings: torch.Tensor) -> torch.Tensor:
        if current_embeddings.size(0) < 2:
            return current_embeddings.new_zeros(())

        current_sim = current_embeddings @ current_embeddings.T
        reference_sim = reference_embeddings @ reference_embeddings.T
        mask = ~torch.eye(current_embeddings.size(0), dtype=torch.bool, device=current_embeddings.device)
        return F.mse_loss(current_sim[mask], reference_sim[mask])

    @staticmethod
    def _adapter_model(model):
        return model.module if hasattr(model, "module") else model

    def _set_reference_adapter(self, model) -> None:
        peft_model = self._adapter_model(model)
        activate_adapters(peft_model, STAGE1_ADAPTER_NAME)
        configure_stage2_trainability(peft_model)

    def _set_training_adapter(self, model) -> None:
        peft_model = self._adapter_model(model)
        activate_adapters(peft_model, get_active_adapter_names(include_stage1=True))
        configure_stage2_trainability(peft_model)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("Regularized stage-2 training expects tokenized labels in each batch")

        training_inputs = self._filter_model_inputs(inputs, include_labels=True)
        reference_inputs = self._filter_model_inputs(inputs, include_labels=False)
        attention_mask = training_inputs["attention_mask"]

        self._set_training_adapter(model)
        outputs = model(**training_inputs, output_hidden_states=True)
        lm_loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        if not lm_loss.requires_grad and model.training:
            raise RuntimeError("Stage-2 LM loss is detached; active adapter path is not trainable")

        current_hidden = self._prompt_hidden_from_outputs(outputs, labels=labels, attention_mask=attention_mask)
        current_hidden = F.normalize(current_hidden.float(), dim=-1)

        was_training = model.training
        if was_training:
            model.eval()
        self._set_reference_adapter(model)
        reference_forward_kwargs = {
            **reference_inputs,
            "output_hidden_states": True,
        }
        model_type = getattr(getattr(model, "config", None), "model_type", None)
        if model_type in {"gemma2", "gemma"}:
            reference_forward_kwargs["logits_to_keep"] = 1
        with torch.no_grad():
            reference_outputs = model(**reference_forward_kwargs)
        self._set_training_adapter(model)
        if was_training:
            model.train()

        reference_hidden = self._prompt_hidden_from_outputs(
            reference_outputs,
            labels=labels,
            attention_mask=attention_mask,
        )
        reference_hidden = F.normalize(reference_hidden.float(), dim=-1)

        anchor_loss = 1.0 - F.cosine_similarity(current_hidden, reference_hidden, dim=-1).mean()
        current_mean = current_hidden.mean(dim=0)
        reference_mean = reference_hidden.mean(dim=0)
        current_var = current_hidden.var(dim=0, unbiased=False)
        reference_var = reference_hidden.var(dim=0, unbiased=False)
        distribution_loss = F.mse_loss(current_mean, reference_mean) + F.mse_loss(current_var, reference_var)
        relation_loss = self._pairwise_relation_loss(current_hidden, reference_hidden)

        total_loss = lm_loss
        total_loss = total_loss + self.embedding_anchor_weight * anchor_loss
        total_loss = total_loss + self.embedding_distribution_weight * distribution_loss
        total_loss = total_loss + self.embedding_relation_weight * relation_loss

        if self.model.training and self.state.global_step != self._last_reg_log_step:
            self.log(
                {
                    "train_lm_loss": float(lm_loss.detach().cpu()),
                    "train_embedding_anchor_loss": float(anchor_loss.detach().cpu()),
                    "train_embedding_distribution_loss": float(distribution_loss.detach().cpu()),
                    "train_embedding_relation_loss": float(relation_loss.detach().cpu()),
                    "train_embedding_regularized_loss": float(total_loss.detach().cpu()),
                }
            )
            self._last_reg_log_step = self.state.global_step

        if return_outputs:
            return total_loss, outputs
        return total_loss


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
    formatted_val = [format_task_sample(sample, args.top_genes, organism) for sample in val_samples]
    random.Random(args.seed).shuffle(formatted_train)

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
        print(
            "[Config] Reg weights  : "
            f"anchor={args.embedding_anchor_weight} "
            f"distribution={args.embedding_distribution_weight} "
            f"relation={args.embedding_relation_weight}"
        )
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
    model = build_stage2_peft_model(model, lora_config, mixed=True)
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

    trainer = Stage1EmbeddingRegularizedTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        processing_class=tokenizer,
        retrieval_eval_data=retrieval_val_split,
        retrieval_output_dir=retrieval_output_dir,
        retrieval_max_seq_len=args.max_seq_len,
        retrieval_batch_size=args.eval_batch,
        embedding_anchor_weight=args.embedding_anchor_weight,
        embedding_distribution_weight=args.embedding_distribution_weight,
        embedding_relation_weight=args.embedding_relation_weight,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    if is_main_process():
        trainer.run_domain_retrieval_eval(tag="pre_train")

    print("Training stage-2 task adapter with embedding regularization …")
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
        "stage1_adapter_name": STAGE1_ADAPTER_NAME,
        "stage2_adapter_name": STAGE2_ADAPTER_NAME,
        "adapter_composition": composition_mode,
        "separate_adapter_training": True,
        "regularization": {
            "anchor_weight": args.embedding_anchor_weight,
            "distribution_weight": args.embedding_distribution_weight,
            "relation_weight": args.embedding_relation_weight,
            "reference_adapter": STAGE1_ADAPTER_NAME,
            "reference_space": "prompt_hidden_last_token_before_completion",
        },
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
        "regularization": manifest["regularization"],
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