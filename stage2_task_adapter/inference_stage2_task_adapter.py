#!/usr/bin/env python
"""Inference for stage-2 task adapters on top of a stage-1 initialized C2S Gemma model."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from utils import (  # noqa: E402
    STRUCTURED_REASONING_ANSWER_FORMAT,
    STRUCTURED_REASONING_QUESTION_TEXT,
    build_c2s_prompt,
    extract_genes_from_question,
    is_reasoning_dataset_sample,
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
}

STAGE1_ADAPTER_NAME = "stage1_domain"
STAGE2_ADAPTER_NAME = "default"


def activate_adapters(model, adapter_names: str | list[str]) -> None:
    if isinstance(adapter_names, str):
        model.set_adapter(adapter_names)
        return

    base_model = getattr(model, "base_model", None)
    if base_model is None or not hasattr(base_model, "set_adapter"):
        raise TypeError("This PEFT model does not support activating multiple adapters")
    base_model.set_adapter(adapter_names)


def load_base_model(base_model: str, use_4bit: bool):
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        return AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quant_config,
            device_map="auto",
            attn_implementation="eager",
            trust_remote_code=True,
        )

    return AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with stage-1 + stage-2 adapters")
    parser.add_argument("--dataset", type=str, required=True, help="Stage-2 JSONL dataset")
    parser.add_argument("--stage2-adapter", type=str, required=True, help="Stage-2 task adapter directory")
    parser.add_argument("--stage1-adapter", type=str, default=None, help="Optional stage-1 adapter directory")
    parser.add_argument(
        "--base-model",
        type=str,
        default=os.getenv("GEMMA_MODEL_PATH", "vandijklab/C2S-Scale-Gemma-2-2B"),
        help="Gemma base model path",
    )
    parser.add_argument("--top-genes", type=int, default=200, help="Genes to keep")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum generated tokens")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Maximum prompt length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--num-samples", type=int, default=None, help="Optional evaluation subset")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_genes(sample: dict, top_genes: int) -> list[str]:
    if "genes" in sample and isinstance(sample["genes"], list):
        genes = [str(gene) for gene in sample["genes"]]
    elif "cell_sentence" in sample and isinstance(sample["cell_sentence"], str):
        genes = sample["cell_sentence"].strip().split()
    elif "question" in sample and isinstance(sample["question"], str):
        genes = extract_genes_from_question(sample["question"])
    else:
        raise ValueError("Each sample must define genes, cell_sentence, or question")
    return [gene for gene in genes if gene][:top_genes]


def build_general_task_prompt(genes: list[str], task_type: str, sample: dict) -> str:
    organism = "Homo sapiens"
    cell_sentence = " ".join(genes)
    task_instruction = sample.get("task_instruction") or TASK_DESCRIPTIONS[task_type]
    reasoning_sample = task_type == "freeform_qa" and is_reasoning_dataset_sample(sample)
    question_text = sample.get("question_text") or (
        STRUCTURED_REASONING_QUESTION_TEXT if reasoning_sample else "Answer the grounded biology question for this cell sentence."
    )
    prompt_parts = [
        (
            f"The following is a list of {len(genes)} gene names ordered by descending expression level "
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


def build_prompt(sample: dict, top_genes: int) -> str:
    task_type = sample.get("task_type", "freeform_qa")
    genes = extract_genes(sample, top_genes)
    if task_type in {"cell_type", "domain_replay"}:
        return build_c2s_prompt(genes)
    return build_general_task_prompt(genes, task_type, sample)


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    output_path = Path(args.output) if args.output else SCRIPT_DIR / "outputs" / f"stage2_inference_{dataset_path.stem}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading dataset …")
    samples = load_jsonl(dataset_path)
    if args.num_samples is not None:
        samples = samples[: args.num_samples]
    print(f"Loaded {len(samples)} samples")

    print("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(args.stage2_adapter, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    use_4bit = not args.no_4bit
    print("Loading base model …")
    model = load_base_model(args.base_model, use_4bit)

    print("Applying stage-2 adapter …")
    if args.stage1_adapter:
        model = PeftModel.from_pretrained(model, args.stage2_adapter, adapter_name=STAGE2_ADAPTER_NAME, is_trainable=False)
        print("Loading frozen stage-1 adapter …")
        model.load_adapter(args.stage1_adapter, adapter_name=STAGE1_ADAPTER_NAME, is_trainable=False)
        activate_adapters(model, [STAGE2_ADAPTER_NAME, STAGE1_ADAPTER_NAME])
    else:
        model = PeftModel.from_pretrained(model, args.stage2_adapter, adapter_name=STAGE2_ADAPTER_NAME)
        activate_adapters(model, STAGE2_ADAPTER_NAME)
    model.eval()

    start_time = time.time()
    with open(output_path, "w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples, start=1):
            prompt = build_prompt(sample, args.top_genes)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_seq_len).to(model.device)
            with torch.no_grad():
                if args.temperature == 0.0:
                    output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                else:
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                    )
            generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            result = {
                "sample_id": sample.get("sample_id"),
                "task_type": sample.get("task_type", "freeform_qa"),
                "prompt": prompt,
                "answer": sample.get("answer"),
                "generated": generated,
            }
            handle.write(json.dumps(result) + "\n")
            print(f"[{index}/{len(samples)}] {result['task_type']} done", flush=True)

    elapsed = time.time() - start_time
    print(f"Results saved to: {output_path}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()