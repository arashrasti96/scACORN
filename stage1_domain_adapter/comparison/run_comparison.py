#!/usr/bin/env python3
"""Unified comparison runner for single-cell embedding methods.

Usage examples
--------------
  # Run all methods on all datasets:
  python run_comparison.py

  # Run specific methods on specific datasets:
  python run_comparison.py --methods pca tfidf c2s_lora --datasets heart pancreas

  # Specify a custom output directory:
  python run_comparison.py --output-dir /scratch/comparison_results

  # Use a specific LoRA checkpoint:
  python run_comparison.py --lora-run gemma_stage1_domain_adapter_h100x2

  # Use a specific scGPT model directory:
  python run_comparison.py --scgpt-model-dir /path/to/scGPT_human
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Prevent BLAS/LAPACK threading deadlocks on HPC nodes.
# Must be set before numpy/sklearn are imported.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

from data_loader import discover_datasets, load_and_split
from metrics import evaluate_embeddings

# ---------------------------------------------------------------------------
# Method registry — maps short names to (module_name, display_name) tuples.
# ---------------------------------------------------------------------------

METHOD_REGISTRY: dict[str, str] = {
    "pca": "baseline_pca",
    "tfidf": "baseline_tfidf_svd",
    "sentence_transformer": "baseline_sentence_transformer",
    "c2s_base": "method_c2s_base",
    "scgpt": "method_scgpt",
    "geneformer": "method_geneformer",
    "c2s_lora": "method_c2s_lora",
    "c2s_lora_proj": "method_c2s_lora",  # projection space variant
}

# Methods to run by default (scgpt/geneformer excluded — need extra install)
_DEFAULT_METHODS = [
    "pca", "tfidf", "sentence_transformer",
    "c2s_base", "c2s_lora", "c2s_lora_proj",
]


def _load_method(method_key: str):
    """Dynamically import a method module and return it."""
    import importlib

    module_name = METHOD_REGISTRY[method_key]
    return importlib.import_module(module_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run embedding comparison across methods and datasets."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=_DEFAULT_METHODS,
        choices=list(METHOD_REGISTRY.keys()),
        help="Methods to run (default: all).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset short names to evaluate (e.g. 'heart pancreas'). "
        "Matched as substrings against export directory names. Default: all.",
    )
    parser.add_argument(
        "--exports-root",
        type=str,
        default=None,
        help="Path to the JSONL exports root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
        help="Directory to save results CSV and per-run JSONs.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--top-genes",
        type=int,
        default=200,
        help="Max genes per cell.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    # Method-specific arguments
    parser.add_argument(
        "--lora-run",
        type=str,
        default=None,
        help="Run name for LoRA checkpoint (auto-discover if not set).",
    )
    parser.add_argument(
        "--lora-checkpoint-dir",
        type=str,
        default=None,
        help="Explicit LoRA checkpoint directory.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Base Gemma model path (overrides env var).",
    )
    parser.add_argument(
        "--scgpt-model-dir",
        type=str,
        default="scGPT_human",
        help="scGPT pre-trained model directory.",
    )
    parser.add_argument(
        "--geneformer-model",
        type=str,
        default="ctheodoris/Geneformer",
        help="Geneformer model name or path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for GPU methods.",
    )
    return parser.parse_args()


def _filter_datasets(
    all_datasets: list[tuple[str, Path]],
    patterns: list[str] | None,
) -> list[tuple[str, Path]]:
    """Keep only datasets whose names match any of the given patterns."""
    if patterns is None:
        return all_datasets
    filtered = []
    for ds_name, ds_path in all_datasets:
        for pat in patterns:
            if pat.lower() in ds_name.lower():
                filtered.append((ds_name, ds_path))
                break
    return filtered


def _build_method_kwargs(
    args: argparse.Namespace, method_key: str, dataset_name: str = "",
) -> dict:
    """Build method-specific keyword arguments."""
    kw: dict = {"batch_size": args.batch_size}
    if args.model_path:
        kw["model_path"] = args.model_path
    if method_key in ("c2s_lora", "c2s_lora_proj"):
        kw["return_projections"] = (method_key == "c2s_lora_proj")
        kw["dataset_name"] = dataset_name
        if args.lora_checkpoint_dir:
            kw["checkpoint_dir"] = args.lora_checkpoint_dir
        elif args.lora_run:
            kw["run_name"] = args.lora_run
    elif method_key == "scgpt":
        kw["model_dir"] = args.scgpt_model_dir
    elif method_key == "geneformer":
        kw["model_name"] = args.geneformer_model
    return kw


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_datasets = discover_datasets(args.exports_root)
    datasets = _filter_datasets(all_datasets, args.datasets)
    if not datasets:
        print("ERROR: No datasets found. Check --exports-root or --datasets.")
        sys.exit(1)

    print(f"Datasets ({len(datasets)}): {[d[0] for d in datasets]}")
    print(f"Methods  ({len(args.methods)}): {args.methods}")
    print(f"Split: {args.split}")
    print(f"Output: {output_dir}\n")

    csv_path = output_dir / "comparison_results.csv"
    csv_fields = [
        "dataset",
        "method",
        "n_samples",
        "R@1",
        "R@5",
        "R@10",
        "NMI",
        "ARI",
        "Silhouette",
        "time_s",
    ]
    csv_rows: list[dict] = []

    for ds_name, ds_path in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        split_data = load_and_split(
            ds_path,
            top_genes=args.top_genes,
            seed=args.seed,
        )
        genes_lists, labels = split_data[args.split]
        if not genes_lists:
            print(f"  SKIP: no {args.split} data.")
            continue
        labels_arr = np.array(labels)
        print(f"  {args.split} split: {len(genes_lists)} cells, "
              f"{len(np.unique(labels_arr))} types", flush=True)

        for method_key in args.methods:
            print(f"\n  --- {method_key} ---", flush=True)
            try:
                mod = _load_method(method_key)
                method_name = mod.name()
                print(f"    loaded: {method_name}", flush=True)
                if method_key == "c2s_lora_proj":
                    method_name += " [projections]"

                kw = _build_method_kwargs(args, method_key, dataset_name=ds_name)
                t0 = time.time()
                embeddings = mod.embed(genes_lists, **kw)
                embed_time = time.time() - t0
                print(f"    embed: {embed_time:.1f}s  shape={embeddings.shape}", flush=True)

                t1 = time.time()
                metrics = evaluate_embeddings(embeddings, labels_arr)
                eval_time = time.time() - t1
                elapsed = embed_time
                metrics["time_s"] = round(elapsed, 2)

                row = {
                    "dataset": ds_name,
                    "method": method_name,
                    "n_samples": len(genes_lists),
                    **{k: metrics.get(k) for k in csv_fields[3:]},
                }
                csv_rows.append(row)

                # Print summary
                r1 = metrics.get("R@1")
                r5 = metrics.get("R@5")
                r10 = metrics.get("R@10")
                nmi = metrics.get("NMI")
                print(f"    R@1={r1:.4f}  R@5={r5:.4f}  R@10={r10:.4f}  "
                      f"NMI={nmi:.4f}  (embed={embed_time:.1f}s  eval={eval_time:.1f}s)"
                      if r1 is not None else "    N/A", flush=True)

                # Save per-run JSON
                run_json = output_dir / f"{ds_name}__{method_key}.json"
                with open(run_json, "w") as f:
                    json.dump(
                        {"dataset": ds_name, "method": method_name, "metrics": metrics},
                        f,
                        indent=2,
                    )

            except Exception as exc:
                print(f"    ERROR: {exc}", flush=True)
                traceback.print_exc()
                csv_rows.append({
                    "dataset": ds_name,
                    "method": method_key,
                    "n_samples": len(genes_lists),
                    **{k: None for k in csv_fields[3:]},
                })
            finally:
                # Free GPU memory between methods
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    # Write aggregate CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n{'='*60}")
    print(f"Results saved to {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
