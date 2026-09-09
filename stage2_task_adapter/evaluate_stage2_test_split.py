#!/usr/bin/env python
"""Run stage-2 test-set generation for a training output directory.

Example:
  python evaluate_stage2_test_split.py \
      --run-dir outputs/bladder_stage2_from_split_l40s4

Optional:
  python evaluate_stage2_test_split.py \
      --run-dir outputs/heart_stage2_from_split_l40s4 \
      --num-samples 32 \
      --temperature 0.0
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from inference_stage2_task_adapter import (
    STAGE1_ADAPTER_NAME,
    STAGE2_ADAPTER_NAME,
    activate_adapters,
    build_prompt,
    load_base_model,
    load_jsonl,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run test-set inference for a stage-2 output directory")
    parser.add_argument("--run-dir", type=str, required=True, help="Stage-2 output directory")
    parser.add_argument("--manifest", type=str, default=None, help="Path to task_manifest.json")
    parser.add_argument("--training-summary", type=str, default=None, help="Path to training_summary.json")
    parser.add_argument("--test-data", type=str, default=None, help="Task test JSONL path")
    parser.add_argument("--stage2-adapter", type=str, default=None, help="Stage-2 adapter directory")
    parser.add_argument("--stage1-adapter", type=str, default=None, help="Optional stage-1 adapter override")
    parser.add_argument("--base-model", type=str, default=None, help="Base model path")
    parser.add_argument("--top-genes", type=int, default=None, help="Genes to keep from each sample")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Maximum prompt length")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum generated tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--num-samples", type=int, default=None, help="Optional evaluation subset")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--output-jsonl", type=str, default=None, help="Per-sample prediction output path")
    parser.add_argument("--output-summary", type=str, default=None, help="Summary metrics output path")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_test_data(manifest: dict, run_dir: Path) -> Path:
    args = manifest.get("args") or {}
    dataset_context = manifest.get("dataset_context") or {}
    candidates: list[Path] = []

    for key in ("train_data", "val_data"):
        raw_path = args.get(key)
        if not raw_path:
            continue
        path = Path(raw_path)
        name = path.name
        if name.endswith("_train.jsonl"):
            candidates.append(path.with_name(name.replace("_train.jsonl", "_test.jsonl")))
        if name.endswith("_val.jsonl"):
            candidates.append(path.with_name(name.replace("_val.jsonl", "_test.jsonl")))

    dataset_dir = dataset_context.get("dataset_dir")
    if dataset_dir:
        dataset_dir_path = Path(dataset_dir)
        candidates.append(dataset_dir_path / "cell_annotation_rationale_test.jsonl")
        candidates.append(dataset_dir_path / "test.jsonl")

    split_config = dataset_context.get("stage1_split_config") or {}
    split_paths = split_config.get("split_paths") or {}
    if split_paths.get("test"):
        candidates.append(Path(split_paths["test"]))

    candidates.append(run_dir / "test.jsonl")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not infer a test dataset path from the run manifest")


def normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().strip("\"'")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.lower() or None


def extract_structured_field(text: str | None, field_name: str) -> str | None:
    if not text:
        return None
    prefix = f"{field_name}:"
    for line in text.splitlines():
        if line.strip().upper().startswith(prefix.upper()):
            return line.split(":", 1)[1].strip() or None
    return None


def extract_target_label(sample: dict) -> str | None:
    answer = sample.get("answer")
    label = extract_structured_field(answer, "FINAL") or extract_structured_field(answer, "LABEL")
    if label:
        return label
    metadata = sample.get("metadata") or {}
    return metadata.get("cell_type") or sample.get("label")


def extract_canonical_label(sample: dict) -> str | None:
    metadata = sample.get("metadata") or {}
    provenance = metadata.get("provenance") or {}
    return provenance.get("canonical_label_for_grounding") or metadata.get("free_annotation")


def normalize_gene_token(value: str) -> str:
    cleaned = value.strip().strip("\"'")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.rstrip(".;:")
    return cleaned.upper()


def extract_ground_truth_evidence_genes(sample: dict) -> list[str]:
    explicit = sample.get("evidence_genes")
    if isinstance(explicit, list) and explicit:
        return [normalize_gene_token(gene) for gene in explicit if str(gene).strip()]

    answer = sample.get("answer")
    evidence_line = extract_structured_field(answer, "EVIDENCE")
    if not evidence_line or evidence_line.lower() == "none":
        return []
    return [normalize_gene_token(part) for part in evidence_line.split(",") if part.strip()]


def extract_generated_evidence_genes(text: str | None) -> list[str]:
    evidence_line = extract_structured_field(text, "EVIDENCE")
    if not evidence_line or evidence_line.lower() == "none":
        return []
    return [normalize_gene_token(part) for part in evidence_line.split(",") if part.strip()]


def build_result_record(sample: dict, prompt: str, generated: str, elapsed_sec: float) -> dict:
    target_label = extract_target_label(sample)
    generated_label = extract_structured_field(generated, "FINAL") or extract_structured_field(generated, "LABEL")
    canonical_target = extract_canonical_label(sample)
    input_genes = [normalize_gene_token(gene) for gene in sample.get("genes") or [] if str(gene).strip()]
    input_gene_set = set(input_genes)
    ground_truth_evidence_genes = extract_ground_truth_evidence_genes(sample)
    generated_evidence_genes = extract_generated_evidence_genes(generated)
    ground_truth_evidence_set = set(ground_truth_evidence_genes)
    evidence_in_input_count = sum(1 for gene in generated_evidence_genes if gene in input_gene_set)
    evidence_overlap_count = sum(1 for gene in generated_evidence_genes if gene in ground_truth_evidence_set)
    return {
        "sample_id": sample.get("sample_id"),
        "task_type": sample.get("task_type", "freeform_qa"),
        "prompt": prompt,
        "target_answer": sample.get("answer"),
        "target_label": target_label,
        "canonical_target_label": canonical_target,
        "input_genes": input_genes,
        "ground_truth_evidence_genes": ground_truth_evidence_genes,
        "generated": generated,
        "generated_label": generated_label,
        "generated_evidence_genes": generated_evidence_genes,
        "exact_match": normalize_text(generated_label) == normalize_text(target_label),
        "canonical_match": normalize_text(generated_label) == normalize_text(canonical_target),
        "evidence_in_input_count": evidence_in_input_count,
        "evidence_in_input_rate": (
            evidence_in_input_count / len(generated_evidence_genes)
            if generated_evidence_genes
            else None
        ),
        "evidence_ground_truth_match_count": evidence_overlap_count,
        "evidence_ground_truth_match_rate": (
            evidence_overlap_count / len(ground_truth_evidence_genes)
            if ground_truth_evidence_genes
            else None
        ),
        "elapsed_sec": elapsed_sec,
    }


def summarize_results(results: list[dict], config: dict) -> dict:
    total = len(results)
    exact_matches = sum(1 for row in results if row.get("exact_match"))
    canonical_matches = sum(1 for row in results if row.get("canonical_match"))
    missing_generated_label = sum(1 for row in results if not row.get("generated_label"))
    total_generated_evidence = sum(len(row.get("generated_evidence_genes") or []) for row in results)
    total_ground_truth_evidence = sum(len(row.get("ground_truth_evidence_genes") or []) for row in results)
    total_evidence_in_input = sum(int(row.get("evidence_in_input_count") or 0) for row in results)
    total_evidence_matches = sum(int(row.get("evidence_ground_truth_match_count") or 0) for row in results)
    task_counts = Counter(row.get("task_type", "unknown") for row in results)
    predicted_counts = Counter(row["generated_label"] for row in results if row.get("generated_label"))
    latency_values = [float(row["elapsed_sec"]) for row in results]
    total_elapsed = sum(latency_values)
    return {
        "config": config,
        "num_samples": total,
        "task_counts": dict(task_counts),
        "exact_match_count": exact_matches,
        "exact_match_rate": (exact_matches / total) if total else None,
        "canonical_match_count": canonical_matches,
        "canonical_match_rate": (canonical_matches / total) if total else None,
        "missing_generated_label_count": missing_generated_label,
        "evidence_in_input_count": total_evidence_in_input,
        "evidence_in_input_rate": (
            total_evidence_in_input / total_generated_evidence
            if total_generated_evidence
            else None
        ),
        "evidence_ground_truth_match_count": total_evidence_matches,
        "evidence_ground_truth_match_rate": (
            total_evidence_matches / total_ground_truth_evidence
            if total_ground_truth_evidence
            else None
        ),
        "generated_evidence_gene_count": total_generated_evidence,
        "ground_truth_evidence_gene_count": total_ground_truth_evidence,
        "total_elapsed_sec": total_elapsed,
        "mean_elapsed_sec": (total_elapsed / total) if total else None,
        "top_generated_labels": predicted_counts.most_common(25),
    }


def resolve_runtime_config(args: argparse.Namespace) -> dict:
    run_dir = Path(args.run_dir).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else run_dir / "task_manifest.json"
    summary_path = Path(args.training_summary).resolve() if args.training_summary else run_dir / "training_summary.json"

    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    summary = load_json(summary_path) if summary_path.exists() else {}
    manifest_args = manifest.get("args") or {}

    stage2_adapter = Path(args.stage2_adapter).resolve() if args.stage2_adapter else run_dir / "adapter"
    stage1_adapter = args.stage1_adapter or manifest.get("stage1_adapter") or manifest_args.get("stage1_adapter")
    base_model = args.base_model or manifest_args.get("model_path")
    top_genes = args.top_genes if args.top_genes is not None else int(manifest_args.get("top_genes") or 200)
    max_seq_len = args.max_seq_len if args.max_seq_len is not None else int(manifest_args.get("max_seq_len") or 2048)
    test_data = Path(args.test_data).resolve() if args.test_data else infer_test_data(manifest, run_dir)
    output_jsonl = Path(args.output_jsonl).resolve() if args.output_jsonl else run_dir / "test_predictions.jsonl"
    output_summary = Path(args.output_summary).resolve() if args.output_summary else run_dir / "test_summary.json"

    if not stage2_adapter.exists():
        raise FileNotFoundError(f"Stage-2 adapter directory not found: {stage2_adapter}")
    if not test_data.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_data}")
    if not base_model:
        raise ValueError("Base model path could not be resolved from arguments or task_manifest.json")

    return {
        "run_dir": run_dir,
        "manifest_path": manifest_path,
        "summary_path": summary_path,
        "training_summary": summary,
        "stage2_adapter": stage2_adapter,
        "stage1_adapter": stage1_adapter,
        "base_model": base_model,
        "top_genes": top_genes,
        "max_seq_len": max_seq_len,
        "test_data": test_data,
        "output_jsonl": output_jsonl,
        "output_summary": output_summary,
    }


def main() -> None:
    args = parse_args()
    runtime = resolve_runtime_config(args)

    print(f"[Config] Run dir       : {runtime['run_dir']}")
    print(f"[Config] Test data     : {runtime['test_data']}")
    print(f"[Config] Stage-2       : {runtime['stage2_adapter']}")
    print(f"[Config] Stage-1       : {runtime['stage1_adapter']}")
    print(f"[Config] Base model    : {runtime['base_model']}")
    print(f"[Config] Output JSONL  : {runtime['output_jsonl']}")
    print(f"[Config] Output summary: {runtime['output_summary']}")

    print("Loading test dataset ...")
    samples = load_jsonl(runtime["test_data"])
    if args.num_samples is not None:
        samples = samples[: args.num_samples]
    print(f"Loaded {len(samples)} samples")

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(str(runtime["stage2_adapter"]), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    use_4bit = not args.no_4bit
    print("Loading base model ...")
    model = load_base_model(runtime["base_model"], use_4bit)

    print("Loading adapters ...")
    if runtime["stage1_adapter"]:
        model = PeftModel.from_pretrained(
            model,
            str(runtime["stage2_adapter"]),
            adapter_name=STAGE2_ADAPTER_NAME,
            is_trainable=False,
        )
        model.load_adapter(str(runtime["stage1_adapter"]), adapter_name=STAGE1_ADAPTER_NAME, is_trainable=False)
        activate_adapters(model, [STAGE2_ADAPTER_NAME, STAGE1_ADAPTER_NAME])
    else:
        model = PeftModel.from_pretrained(
            model,
            str(runtime["stage2_adapter"]),
            adapter_name=STAGE2_ADAPTER_NAME,
            is_trainable=False,
        )
        activate_adapters(model, STAGE2_ADAPTER_NAME)
    model.eval()

    runtime["output_jsonl"].parent.mkdir(parents=True, exist_ok=True)
    runtime["output_summary"].parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    wall_start = time.time()
    with open(runtime["output_jsonl"], "w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples, start=1):
            prompt = build_prompt(sample, runtime["top_genes"])
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=runtime["max_seq_len"],
            ).to(model.device)
            sample_start = time.time()
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
            sample_elapsed = time.time() - sample_start
            generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            result = build_result_record(sample, prompt, generated, sample_elapsed)
            results.append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                f"[{index}/{len(samples)}] {result['task_type']} exact={result['exact_match']} canonical={result['canonical_match']}",
                flush=True,
            )

    config_summary = {
        "script_dir": str(SCRIPT_DIR),
        "run_dir": str(runtime["run_dir"]),
        "manifest_path": str(runtime["manifest_path"]),
        "summary_path": str(runtime["summary_path"]),
        "test_data": str(runtime["test_data"]),
        "stage2_adapter": str(runtime["stage2_adapter"]),
        "stage1_adapter": runtime["stage1_adapter"],
        "base_model": runtime["base_model"],
        "top_genes": runtime["top_genes"],
        "max_seq_len": runtime["max_seq_len"],
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "num_samples_requested": args.num_samples,
        "used_4bit": use_4bit,
        "wall_elapsed_sec": time.time() - wall_start,
        "best_model_checkpoint": runtime["training_summary"].get("best_model_checkpoint"),
        "best_metric": runtime["training_summary"].get("best_metric"),
    }
    summary = summarize_results(results, config_summary)

    with open(runtime["output_summary"], "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Saved predictions to: {runtime['output_jsonl']}")
    print(f"Saved summary to    : {runtime['output_summary']}")


if __name__ == "__main__":
    main()