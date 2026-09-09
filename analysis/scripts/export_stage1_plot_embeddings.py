"""Export full-sample Stage 1 embeddings for manuscript UMAP figures.

Example GPU run:
python analysis/scripts/export_stage1_plot_embeddings.py \
    --split-family grouped \
    --datasets tabula_sapiens_prostate_cell_annotation tabula_sapiens_stomach_cell_annotation \
    --methods rank_pca tfidf_svd c2s_base c2s_lora_features c2s_lora_projections \
    --splits train val test \
    --batch-size 16 \
    --max-seq-len 512 \
    --use-4bit
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CONTRASTIVE_ROOT = ROOT
BENCHMARK_SUITE_SRC = CONTRASTIVE_ROOT / "stage1_domain_adapter" / "benchmark_suite"
DATA_EXPORTS_ROOT = CONTRASTIVE_ROOT / "dataset_pipeline" / "data" / "exports"
STAGE1_OUTPUTS_ROOT = CONTRASTIVE_ROOT / "stage1_domain_adapter" / "outputs"
DEFAULT_OUTPUT_ROOT = ROOT / "analysis" / "artifacts" / "stage1_embedding_exports"

DEFAULT_DATASETS = [
    "tabula_sapiens_prostate_cell_annotation",
    "tabula_sapiens_stomach_cell_annotation",
]
DEFAULT_METHODS = [
    "rank_pca",
    "tfidf_svd",
    "c2s_base",
    "c2s_lora_features",
    "c2s_lora_projections",
]
DEFAULT_SPLITS = ["train", "val", "test"]


def ensure_benchmark_imports() -> None:
    for path in (BENCHMARK_SUITE_SRC, CONTRASTIVE_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def import_benchmark_module(module_name: str):
    ensure_benchmark_imports()
    return importlib.import_module(module_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export full-sample Stage 1 embeddings for manuscript UMAP figures."
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--split-family", default="grouped", choices=["grouped", "standard"])
    parser.add_argument("--top-genes", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint-tag", default="best")
    parser.add_argument("--sentence-transformer-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--geneformer-model", default="ctheodoris/Geneformer")
    parser.add_argument("--scgpt-model-dir", default=None)
    parser.add_argument("--label-key", default="cell_type")
    parser.add_argument("--use-4bit", action="store_true", default=False)
    return parser.parse_args()


def dataset_output_path(output_root: Path, split_family: str, dataset: str, method_key: str) -> Path:
    return output_root / split_family / dataset / f"{method_key}.npz"


def export_method_embeddings(args: argparse.Namespace, dataset: str, method_key: str) -> None:
    job_started = time.perf_counter()
    print(f"Preparing {dataset} :: {method_key}", flush=True)
    data_module = import_benchmark_module("data")
    methods_module = import_benchmark_module("methods")
    load_dataset_splits = data_module.load_dataset_splits
    build_method = methods_module.build_method

    dataset_dir = DATA_EXPORTS_ROOT / dataset
    split_map = load_dataset_splits(
        dataset_dir=dataset_dir,
        split_family=args.split_family,
        splits=args.splits,
        top_genes=args.top_genes,
        label_key=args.label_key,
    )
    split_counts = ", ".join(
        f"{split_name}={split_map[split_name].n_samples}" for split_name in args.splits
    )
    print(f"Loaded splits for {dataset} :: {method_key} ({split_counts})", flush=True)
    method = build_method(
        method_key=method_key,
        dataset_name=dataset,
        outputs_root=STAGE1_OUTPUTS_ROOT,
        model_path=None,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        use_4bit=args.use_4bit,
        sentence_transformer_model=args.sentence_transformer_model,
        scgpt_model_dir=args.scgpt_model_dir,
        geneformer_model=args.geneformer_model,
        checkpoint_tag=args.checkpoint_tag,
    )
    print(
        f"Built method {getattr(method, 'display_name', method_key)} for {dataset} :: {method_key}",
        flush=True,
    )

    if hasattr(method, "fit"):
        fit_started = time.perf_counter()
        print(f"Fitting {dataset} :: {method_key} on train split", flush=True)
        method.fit(split_map["train"].genes_lists)
        print(
            f"Finished fit for {dataset} :: {method_key} in {time.perf_counter() - fit_started:.1f}s",
            flush=True,
        )

    export_payload: dict[str, object] = {
        "dataset": dataset,
        "method_key": method_key,
        "method_name": getattr(method, "display_name", method_key),
        "split_family": args.split_family,
        "splits": np.asarray(args.splits, dtype=object),
    }
    manifest: dict[str, object] = {
        "dataset": dataset,
        "method_key": method_key,
        "method_name": getattr(method, "display_name", method_key),
        "split_family": args.split_family,
        "top_genes": args.top_genes,
        "batch_size": args.batch_size,
        "max_seq_len": args.max_seq_len,
        "splits": {},
    }

    for split_name in args.splits:
        split_data = split_map[split_name]
        split_started = time.perf_counter()
        print(f"Embedding {dataset} :: {method_key} :: {split_name} ({split_data.n_samples} cells)", flush=True)
        embeddings = method.embed(split_data.genes_lists)
        print(
            f"Finished {dataset} :: {method_key} :: {split_name} in {time.perf_counter() - split_started:.1f}s",
            flush=True,
        )
        export_payload[f"{split_name}_embeddings"] = embeddings.astype(np.float32)
        export_payload[f"{split_name}_labels"] = np.asarray(split_data.labels, dtype=object)
        export_payload[f"{split_name}_sample_ids"] = np.asarray(split_data.sample_ids, dtype=object)
        manifest["splits"][split_name] = {
            "n_samples": split_data.n_samples,
            "n_labels": split_data.n_labels,
            "embedding_dim": int(embeddings.shape[1]),
        }

    output_path = dataset_output_path(args.output_root, args.split_family, dataset, method_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **export_payload)
    output_path.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Wrote {output_path} in {time.perf_counter() - job_started:.1f}s total",
        flush=True,
    )

    if hasattr(method, "close"):
        method.close()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        for method_key in args.methods:
            export_method_embeddings(args, dataset, method_key)


if __name__ == "__main__":
    main()