#!/usr/bin/env python
"""Train stage-2 adapters for the remaining datasets, then run test inference.

Example:
  python run_stage2_remaining_datasets.py

Subset example:
    python run_stage2_remaining_datasets.py --include tabula_sapiens_heart_cell_annotation tabula_sapiens_pancreas_cell_annotation

Legacy cell-level split example:
    python run_stage2_remaining_datasets.py --split-tag '' --run-suffix stage2_from_split_l40s4
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EXPORTS_ROOT = SCRIPT_DIR.parent / "dataset_pipeline" / "data" / "exports"
DEFAULT_STAGE1_OUTPUTS_ROOT = SCRIPT_DIR.parent / "stage1_domain_adapter" / "outputs"
DEFAULT_TRAINER = SCRIPT_DIR / "train_stage2_task_adapter_regularized_dataset.py"
DEFAULT_EVALUATOR = SCRIPT_DIR / "evaluate_stage2_test_split.py"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs"
DEFAULT_MODEL_PATH = os.getenv("GEMMA_MODEL_PATH", "vandijklab/C2S-Scale-Gemma-2-2B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate stage-2 adapters for the remaining datasets")
    parser.add_argument("--exports-root", type=str, default=str(DEFAULT_EXPORTS_ROOT), help="Root containing dataset export folders")
    parser.add_argument(
        "--stage1-outputs-root",
        type=str,
        default=str(DEFAULT_STAGE1_OUTPUTS_ROOT),
        help="Root containing per-dataset stage-1 outputs",
    )
    parser.add_argument("--trainer-script", type=str, default=str(DEFAULT_TRAINER), help="Stage-2 trainer entrypoint")
    parser.add_argument("--evaluator-script", type=str, default=str(DEFAULT_EVALUATOR), help="Stage-2 evaluator entrypoint")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="Root directory for stage-2 outputs")
    parser.add_argument("--deepspeed-bin", type=str, default="deepspeed", help="DeepSpeed executable or command name")
    parser.add_argument("--python-bin", type=str, default=sys.executable, help="Python executable used for evaluation")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs per training run")
    parser.add_argument("--use-deepspeed", action="store_true", help="Launch training with DeepSpeed instead of plain Python")
    parser.add_argument("--include", type=str, nargs="*", default=None, help="Optional dataset directory names to include")
    parser.add_argument("--exclude", type=str, nargs="*", default=None, help="Optional dataset directory names to exclude")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching them")
    parser.add_argument("--data-stem", type=str, default="cell_annotation_rationale", help="Task JSONL file stem inside each dataset export dir")
    parser.add_argument(
        "--split-tag",
        type=str,
        default="grouped",
        help="Optional split tag inserted before train/val/test (for example 'grouped' -> stem_grouped_train.jsonl). Use empty string for legacy stem_train.jsonl paths.",
    )
    parser.add_argument(
        "--run-suffix",
        type=str,
        default="stage2_from_grouped_split_l40s4",
        help="Suffix used when generating output directory names",
    )
    parser.add_argument("--stage1-run-name-prefix", type=str, default="gemma_stage1", help="Prefix used for stage-1 output directories")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Base model path")
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs for each stage-2 run")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch", type=int, default=4, help="Per-device train batch size")
    parser.add_argument("--eval-batch", type=int, default=4, help="Per-device eval batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--top-genes", type=int, default=200, help="Top genes to keep")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora-target-modules", type=str, default="q_proj,k_proj,v_proj,o_proj", help="Comma-separated LoRA target modules")
    parser.add_argument("--layer-strategy", type=str, default="last_third", help="Layer selection strategy")
    parser.add_argument("--embedding-anchor-weight", type=float, default=0.6, help="Anchor regularization weight")
    parser.add_argument("--embedding-distribution-weight", type=float, default=0.6, help="Distribution regularization weight")
    parser.add_argument("--embedding-relation-weight", type=float, default=0.6, help="Relation regularization weight")
    parser.add_argument("--replay-ratio", type=float, default=0.0, help="Replay ratio for domain train split")
    parser.add_argument("--replay-val-ratio", type=float, default=0.0, help="Replay ratio for domain val split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--zero-stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum generated tokens during test inference")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature during test inference")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading for both training and inference")
    parser.add_argument(
        "--include-evidence-in-ground-truth",
        "--include-positive-markers-in-ground-truth",
        dest="include_evidence_in_ground_truth",
        action="store_true",
        help="Keep EVIDENCE lines, and any legacy POSITIVE_MARKERS lines, in structured target answers instead of stripping them during training",
    )
    return parser.parse_args()


def resolve_deepspeed_command(deepspeed_bin: str) -> list[str]:
    if shutil.which(deepspeed_bin) is not None:
        return [deepspeed_bin]
    if deepspeed_bin == "deepspeed" and importlib.util.find_spec("deepspeed") is not None:
        return [sys.executable, "-m", "deepspeed"]
    raise FileNotFoundError(f"DeepSpeed executable not found: {deepspeed_bin}")


def dataset_short_name(dataset_name: str) -> str:
    short_name = dataset_name
    if short_name.startswith("tabula_sapiens_"):
        short_name = short_name[len("tabula_sapiens_") :]
    if short_name.endswith("_cell_annotation"):
        short_name = short_name[: -len("_cell_annotation")]
    return short_name


def required_split_paths(dataset_dir: Path, data_stem: str) -> dict[str, Path]:
    return {
        "train": dataset_dir / f"{data_stem}_train.jsonl",
        "val": dataset_dir / f"{data_stem}_val.jsonl",
        "test": dataset_dir / f"{data_stem}_test.jsonl",
    }


def resolve_split_stem(data_stem: str, split_tag: str) -> str:
    clean_split_tag = split_tag.strip()
    if not clean_split_tag:
        return data_stem
    return f"{data_stem}_{clean_split_tag}"


def required_split_paths_for_tag(dataset_dir: Path, data_stem: str, split_tag: str) -> dict[str, Path]:
    split_stem = resolve_split_stem(data_stem, split_tag)
    return {
        "train": dataset_dir / f"{split_stem}_train.jsonl",
        "val": dataset_dir / f"{split_stem}_val.jsonl",
        "test": dataset_dir / f"{split_stem}_test.jsonl",
    }


def discover_dataset_dirs(exports_root: Path) -> list[Path]:
    return sorted(path for path in exports_root.iterdir() if path.is_dir())


def build_output_dir(args: argparse.Namespace, dataset_dir: Path) -> Path:
    return Path(args.output_root).resolve() / f"{dataset_short_name(dataset_dir.name)}_{args.run_suffix}"


def build_stage1_run_dir(args: argparse.Namespace, dataset_dir: Path) -> Path:
    return Path(args.stage1_outputs_root).resolve() / f"{args.stage1_run_name_prefix}_{dataset_dir.name}"


def build_training_command(args: argparse.Namespace, dataset_dir: Path, output_dir: Path) -> list[str]:
    split_paths = required_split_paths_for_tag(dataset_dir, args.data_stem, args.split_tag)
    launcher_command = (
        [
            *resolve_deepspeed_command(args.deepspeed_bin),
            "--num_gpus",
            str(args.num_gpus),
            str(Path(args.trainer_script).resolve()),
        ]
        if args.use_deepspeed
        else [args.python_bin, str(Path(args.trainer_script).resolve())]
    )
    return [
        *launcher_command,
        "--train-data",
        str(split_paths["train"]),
        "--val-data",
        str(split_paths["val"]),
        "--dataset-type",
        dataset_dir.name,
        "--exports-root",
        str(Path(args.exports_root).resolve()),
        "--stage1-outputs-root",
        str(Path(args.stage1_outputs_root).resolve()),
        "--stage1-run-name-prefix",
        args.stage1_run_name_prefix,
        "--output-dir",
        str(output_dir),
        "--model-path",
        args.model_path,
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--batch",
        str(args.batch),
        "--eval-batch",
        str(args.eval_batch),
        "--grad-accum",
        str(args.grad_accum),
        "--max-seq-len",
        str(args.max_seq_len),
        "--top-genes",
        str(args.top_genes),
        "--lora-r",
        str(args.lora_r),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
        "--lora-target-modules",
        args.lora_target_modules,
        "--layer-strategy",
        args.layer_strategy,
        "--embedding-anchor-weight",
        str(args.embedding_anchor_weight),
        "--embedding-distribution-weight",
        str(args.embedding_distribution_weight),
        "--embedding-relation-weight",
        str(args.embedding_relation_weight),
        "--replay-ratio",
        str(args.replay_ratio),
        "--replay-val-ratio",
        str(args.replay_val_ratio),
        "--seed",
        str(args.seed),
        "--patience",
        str(args.patience),
        *(["--use-deepspeed", "--zero-stage", str(args.zero_stage)] if args.use_deepspeed else []),
        *([] if args.include_evidence_in_ground_truth else ["--exclude-evidence-in-ground-truth"]),
        *( ["--no-4bit"] if args.no_4bit else [] ),
    ]


def build_evaluation_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    return [
        args.python_bin,
        str(Path(args.evaluator_script).resolve()),
        "--run-dir",
        str(output_dir),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        *( ["--no-4bit"] if args.no_4bit else [] ),
    ]


def print_command(label: str, command: list[str]) -> None:
    print(label, flush=True)
    print(" ".join(shlex.quote(part) for part in command), flush=True)


def validate_dataset(args: argparse.Namespace, dataset_dir: Path) -> str | None:
    split_paths = required_split_paths_for_tag(dataset_dir, args.data_stem, args.split_tag)
    missing = [name for name, path in split_paths.items() if not path.exists()]
    if missing:
        return f"missing split files: {', '.join(missing)}"

    stage1_run_dir = build_stage1_run_dir(args, dataset_dir)
    stage1_adapter_dir = stage1_run_dir / "checkpoints" / "best" / "gemma_lora_stage1"
    if not stage1_adapter_dir.exists():
        return f"missing stage-1 adapter: {stage1_adapter_dir}"
    return None


def main() -> None:
    args = parse_args()
    exports_root = Path(args.exports_root).resolve()
    trainer_script = Path(args.trainer_script).resolve()
    evaluator_script = Path(args.evaluator_script).resolve()
    output_root = Path(args.output_root).resolve()

    if not exports_root.exists():
        raise FileNotFoundError(f"Exports root does not exist: {exports_root}")
    if not trainer_script.exists():
        raise FileNotFoundError(f"Trainer script does not exist: {trainer_script}")
    if not evaluator_script.exists():
        raise FileNotFoundError(f"Evaluator script does not exist: {evaluator_script}")
    if args.num_gpus < 1:
        raise ValueError("--num-gpus must be at least 1")
    if args.num_gpus > 1 and not args.use_deepspeed:
        raise ValueError("Multi-GPU stage-2 runs require --use-deepspeed")
    if args.use_deepspeed:
        resolve_deepspeed_command(args.deepspeed_bin)

    dataset_dirs = discover_dataset_dirs(exports_root)
    if args.include:
        include_set = set(args.include)
        dataset_dirs = [path for path in dataset_dirs if path.name in include_set]
    if args.exclude:
        exclude_set = set(args.exclude)
        dataset_dirs = [path for path in dataset_dirs if path.name not in exclude_set]

    if not dataset_dirs:
        raise RuntimeError(f"No dataset directories found under {exports_root}")

    valid_datasets: list[Path] = []
    for dataset_dir in dataset_dirs:
        invalid_reason = validate_dataset(args, dataset_dir)
        if invalid_reason is not None:
            print(f"[skip] {dataset_dir.name}: {invalid_reason}", flush=True)
            continue
        valid_datasets.append(dataset_dir)

    if not valid_datasets:
        raise RuntimeError("No datasets remain after validating split files and stage-1 adapters")

    print(f"Discovered {len(valid_datasets)} runnable datasets under {exports_root}", flush=True)
    output_root.mkdir(parents=True, exist_ok=True)

    for dataset_dir in valid_datasets:
        output_dir = build_output_dir(args, dataset_dir)
        adapter_dir = output_dir / "adapter"
        print(f"\n[{dataset_dir.name}]", flush=True)
        if adapter_dir.exists():
            print(f"Skipping existing run because {adapter_dir} already exists", flush=True)
            continue

        train_command = build_training_command(args, dataset_dir, output_dir)
        eval_command = build_evaluation_command(args, output_dir)
        print_command("Training command:", train_command)
        print_command("Evaluation command:", eval_command)

        if args.dry_run:
            continue

        train_result = subprocess.run(train_command, check=False, cwd=str(SCRIPT_DIR))
        if train_result.returncode != 0:
            raise subprocess.CalledProcessError(train_result.returncode, train_command)

        eval_result = subprocess.run(eval_command, check=False, cwd=str(SCRIPT_DIR))
        if eval_result.returncode != 0:
            raise subprocess.CalledProcessError(eval_result.returncode, eval_command)


if __name__ == "__main__":
    main()