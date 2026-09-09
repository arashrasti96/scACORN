#!/usr/bin/env python3

'''
 python stage1_domain_adapter/benchmark_suite/run_stage1_embedding_benchmark.py --datasets tabula_sapiens_bladder_cell_annotation tabula_sapiens_ear_cell_annotation tabula_sapiens_eye_cell_annotation tabula_sapiens_heart_cell_annotation tabula_sapiens_ovary_cell_annotation tabula_sapiens_pancreas_cell_annotation tabula_sapiens_prostate_cell_annotation tabula_sapiens_salivary_gland_cell_annotation tabula_sapiens_small_intestine_cell_annotation tabula_sapiens_spleen_cell_annotation tabula_sapiens_stomach_cell_annotation tabula_sapiens_trachea_cell_annotation --output-dir stage1_domain_adapter/benchmark_suite/results/all_tabula_stage1
'''
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

for env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(env_var, "1")

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from data import filter_dataset_dirs, load_dataset_splits, load_grouped_split_summary, normalize_split_family, discover_dataset_dirs
from metrics import clustering_metrics, knn_label_transfer_metrics, self_retrieval_metrics
from methods import DEFAULT_METHOD_KEYS, DEFAULT_STAGE1_OUTPUTS, available_method_keys, build_method


DEFAULT_EXPORTS_ROOT = SCRIPT_DIR.parent.parent / "dataset_pipeline" / "data" / "exports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-1 cell embedding benchmarks with standard and grouped splits")
    parser.add_argument("--exports-root", type=str, default=str(DEFAULT_EXPORTS_ROOT), help="Dataset export root with per-dataset JSONL directories")
    parser.add_argument("--outputs-root", type=str, default=str(DEFAULT_STAGE1_OUTPUTS), help="Stage-1 adapter outputs root")
    parser.add_argument("--datasets", nargs="+", default=None, help="Optional dataset name substrings to include")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHOD_KEYS, choices=available_method_keys(), help="Methods to benchmark")
    parser.add_argument("--split-families", nargs="+", default=["standard", "grouped"], help="Split families to benchmark separately")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Within-family splits to evaluate")
    parser.add_argument("--reference-split", type=str, default="train", choices=["train", "val", "test"], help="Reference split used for train-to-query transfer metrics")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for benchmark outputs. Defaults to a timestamped results folder.")
    parser.add_argument("--top-genes", type=int, default=200, help="Maximum number of genes per cell to keep")
    parser.add_argument("--label-key", type=str, default="cell_type", help="Metadata label field")
    parser.add_argument("--max-cells-per-split", type=int, default=None, help="Optional cap for smoke tests")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for clustering metrics")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for model-backed methods")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Max prompt sequence length for Gemma methods")
    parser.add_argument("--model-path", type=str, default=None, help="Optional explicit Gemma base model path")
    parser.add_argument("--sentence-transformer-model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--scgpt-model-dir", type=str, default=None, help="Local scGPT checkpoint directory")
    parser.add_argument("--geneformer-model", type=str, default="ctheodoris/Geneformer", help="Geneformer model path or HF id")
    parser.add_argument("--checkpoint-tag", type=str, default="best", choices=["best", "last"], help="Which stage-1 checkpoint to load")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading for Gemma methods")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_families = [normalize_split_family(family) for family in args.split_families]
    dataset_dirs = filter_dataset_dirs(discover_dataset_dirs(args.exports_root), args.datasets)
    if not dataset_dirs:
        raise SystemExit("No datasets matched the requested filters")

    output_dir = build_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(output_dir / "run_config.json", args, dataset_dirs, split_families)

    csv_rows: list[dict] = []

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        print(f"\n{'=' * 80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 80}")
        grouped_summary = load_grouped_split_summary(dataset_dir)

        for method_key in args.methods:
            print(f"\nMethod: {method_key}", flush=True)
            method = None
            try:
                method = build_method(
                    method_key=method_key,
                    dataset_name=dataset_name,
                    outputs_root=args.outputs_root,
                    model_path=args.model_path,
                    batch_size=args.batch_size,
                    max_seq_len=args.max_seq_len,
                    use_4bit=not args.no_4bit,
                    sentence_transformer_model=args.sentence_transformer_model,
                    scgpt_model_dir=args.scgpt_model_dir,
                    geneformer_model=args.geneformer_model,
                    checkpoint_tag=args.checkpoint_tag,
                )
                method_payload = method.describe()

                for split_family in split_families:
                    family_start = time.time()
                    print(f"  Split family: {split_family}", flush=True)
                    split_bundle = load_dataset_splits(
                        dataset_dir=dataset_dir,
                        split_family=split_family,
                        splits=args.splits,
                        top_genes=args.top_genes,
                        label_key=args.label_key,
                        max_cells_per_split=args.max_cells_per_split,
                    )
                    reference_data = split_bundle[args.reference_split]

                    fit_start = time.time()
                    method.fit(reference_data.genes_lists)
                    fit_seconds = round(time.time() - fit_start, 4)

                    split_embeddings: dict[str, np.ndarray] = {}
                    embed_seconds_by_split: dict[str, float] = {}
                    for split_name in args.splits:
                        split_data = split_bundle[split_name]
                        embed_start = time.time()
                        split_embeddings[split_name] = method.embed(split_data.genes_lists)
                        embed_seconds_by_split[split_name] = round(time.time() - embed_start, 4)
                        print(
                            f"    {split_name}: n={split_data.n_samples} labels={split_data.n_labels} embed_s={embed_seconds_by_split[split_name]:.2f}",
                            flush=True,
                        )

                    reference_embeddings = split_embeddings[args.reference_split]
                    reference_labels = np.asarray(reference_data.labels)

                    family_results = {
                        "dataset": dataset_name,
                        "split_family": split_family,
                        "grouped_split_summary": grouped_summary if split_family == "grouped" else None,
                        "method": method_payload,
                        "fit_seconds": fit_seconds,
                        "splits": {},
                        "total_family_seconds": None,
                    }

                    for split_name in args.splits:
                        split_data = split_bundle[split_name]
                        split_embeddings_array = split_embeddings[split_name]
                        labels = np.asarray(split_data.labels)
                        self_metrics = self_retrieval_metrics(split_embeddings_array, labels)
                        self_metrics.update(clustering_metrics(split_embeddings_array, labels, seed=args.seed))
                        transfer_metrics = knn_label_transfer_metrics(
                            reference_embeddings=reference_embeddings,
                            reference_labels=reference_labels,
                            query_embeddings=split_embeddings_array,
                            query_labels=labels,
                            same_set=(split_name == args.reference_split),
                        )

                        family_results["splits"][split_name] = {
                            "jsonl_path": str(split_data.jsonl_path),
                            "n_samples": split_data.n_samples,
                            "n_labels": split_data.n_labels,
                            "embed_seconds": embed_seconds_by_split[split_name],
                            "self_metrics": self_metrics,
                            "transfer_metrics": transfer_metrics,
                        }

                        row = {
                            "dataset": dataset_name,
                            "split_family": split_family,
                            "split": split_name,
                            "method_key": method_key,
                            "method_name": method_payload.get("display_name", method_key),
                            "reference_split": args.reference_split,
                            "n_samples": split_data.n_samples,
                            "n_labels": split_data.n_labels,
                            "fit_seconds": fit_seconds,
                            "embed_seconds": embed_seconds_by_split[split_name],
                            "status": "ok",
                            "error": None,
                        }
                        row.update(prefix_metrics("self", self_metrics))
                        row.update(prefix_metrics("transfer", transfer_metrics))
                        csv_rows.append(row)

                    family_results["total_family_seconds"] = round(time.time() - family_start, 4)
                    save_family_result(output_dir, dataset_name, split_family, method_key, family_results)
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                for split_family in split_families:
                    for split_name in args.splits:
                        csv_rows.append(
                            {
                                "dataset": dataset_name,
                                "split_family": split_family,
                                "split": split_name,
                                "method_key": method_key,
                                "method_name": method_key,
                                "reference_split": args.reference_split,
                                "n_samples": None,
                                "n_labels": None,
                                "fit_seconds": None,
                                "embed_seconds": None,
                                "status": "error",
                                "error": error_text,
                            }
                        )
            finally:
                if method is not None:
                    method.close()

    csv_path = output_dir / "benchmark_results.csv"
    write_csv(csv_path, csv_rows)
    with open(output_dir / "benchmark_results.json", "w", encoding="utf-8") as handle:
        json.dump(csv_rows, handle, indent=2)
    print(f"\nSaved benchmark outputs to {output_dir}")


def build_output_dir(output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return SCRIPT_DIR / "results" / timestamp


def write_run_metadata(output_path: Path, args: argparse.Namespace, dataset_dirs: list[Path], split_families: list[str]) -> None:
    payload = {
        "args": vars(args),
        "datasets": [dataset_dir.name for dataset_dir in dataset_dirs],
        "split_families": split_families,
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_family_result(output_dir: Path, dataset_name: str, split_family: str, method_key: str, payload: dict) -> None:
    target_dir = output_dir / dataset_name / split_family
    target_dir.mkdir(parents=True, exist_ok=True)
    with open(target_dir / f"{method_key}.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def prefix_metrics(prefix: str, metrics: dict[str, float | None]) -> dict[str, float | None]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def write_csv(csv_path: Path, rows: list[dict]) -> None:
    fieldnames = sorted({field for row in rows for field in row})
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
