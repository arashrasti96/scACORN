#!/usr/bin/env python
"""
Launch one stage-1 contrastive adapter training run per dataset export directory.

This script discovers dataset directories under dataset_pipeline/data/exports and
launches train_stage1_domain_lora.py once for each directory, so every dataset
gets its own stage-1 adapter and output folder.

Example:
python stage1_domain_adapter/train_stage1_one_adapter_per_dataset.py \
    --exports-root dataset_pipeline/data/exports \
    --run-name-prefix gemma_stage1 \
    --num-gpus 4 \
    --output-root stage1_domain_adapter/outputs \
    --continue-on-error \
    --batch 64 \
    --grad-accum 1 \
    --epochs 6 \
    --lr 5e-4 \
    --warmup-steps 50 \
    --temperature 0.05 \
    --projection-dim 128 \
    --max-seq-len 512 \
    --top-genes 200 \
    --seed 42 \
    --model-path vandijklab/C2S-Scale-Gemma-2-2B
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
DEFAULT_TRAINER = SCRIPT_DIR / "train_stage1_domain_lora.py"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Train one stage-1 adapter per dataset export directory")
    parser.add_argument("--exports-root", type=str, default=str(DEFAULT_EXPORTS_ROOT), help="Root directory containing dataset export folders")
    parser.add_argument(
        "--data-glob",
        type=str,
        default="cell_annotation_rationale.jsonl",
        help="File name or glob expected inside each dataset export directory",
    )
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to pass to deepspeed per dataset run")
    parser.add_argument("--run-name-prefix", type=str, default="stage1_per_dataset", help="Prefix used when generating per-dataset run names")
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Base directory under which the launcher creates one output directory per dataset run",
    )
    parser.add_argument("--trainer-script", type=str, default=str(DEFAULT_TRAINER), help="Stage-1 trainer entrypoint to invoke")
    parser.add_argument("--deepspeed-bin", type=str, default="deepspeed", help="DeepSpeed executable or command name")
    parser.add_argument("--include", type=str, nargs="*", default=None, help="Optional dataset directory names to include")
    parser.add_argument("--exclude", type=str, nargs="*", default=None, help="Optional dataset directory names to exclude")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep launching remaining datasets if one dataset run fails")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching them")
    args, trainer_args = parser.parse_known_args()
    if trainer_args and trainer_args[0] == "--":
        trainer_args = trainer_args[1:]
    return args, trainer_args


def discover_dataset_dirs(exports_root: Path, data_glob: str) -> list[Path]:
    dataset_dirs = []
    for child in sorted(path for path in exports_root.iterdir() if path.is_dir()):
        if any(child.glob(data_glob)):
            dataset_dirs.append(child)
    return dataset_dirs


def trainer_args_include_flag(trainer_args: list[str], flag: str) -> bool:
    for arg in trainer_args:
        if arg == flag or arg.startswith(f"{flag}="):
            return True
    return False


def build_command(args: argparse.Namespace, dataset_dir: Path, trainer_args: list[str]) -> list[str]:
    dataset_name = dataset_dir.name
    run_name = f"{args.run_name_prefix}_{dataset_name}"
    deepspeed_command = resolve_deepspeed_command(args.deepspeed_bin)
    run_trainer_args = list(trainer_args)
    if args.output_root and not trainer_args_include_flag(run_trainer_args, "--output-dir"):
        output_dir = Path(args.output_root).expanduser() / run_name
        run_trainer_args.extend(["--output-dir", str(output_dir)])
    command = [
        *deepspeed_command,
        "--num_gpus",
        str(args.num_gpus),
        args.trainer_script,
        "--data-root",
        str(dataset_dir),
        "--data-glob",
        args.data_glob,
        "--run-name",
        run_name,
        *run_trainer_args,
    ]
    return command


def resolve_deepspeed_command(deepspeed_bin: str) -> list[str]:
    if shutil.which(deepspeed_bin) is not None:
        return [deepspeed_bin]
    if deepspeed_bin == "deepspeed" and importlib.util.find_spec("deepspeed") is not None:
        return [sys.executable, "-m", "deepspeed"]
    raise FileNotFoundError(f"DeepSpeed executable not found: {deepspeed_bin}")


def main() -> None:
    args, trainer_args = parse_args()
    exports_root = Path(args.exports_root).resolve()
    trainer_script = Path(args.trainer_script).resolve()

    if not exports_root.exists():
        raise FileNotFoundError(f"Exports root does not exist: {exports_root}")
    if not trainer_script.exists():
        raise FileNotFoundError(f"Trainer script does not exist: {trainer_script}")
    resolve_deepspeed_command(args.deepspeed_bin)

    dataset_dirs = discover_dataset_dirs(exports_root, args.data_glob)
    if args.include:
        include_set = set(args.include)
        dataset_dirs = [path for path in dataset_dirs if path.name in include_set]
    if args.exclude:
        exclude_set = set(args.exclude)
        dataset_dirs = [path for path in dataset_dirs if path.name not in exclude_set]

    if not dataset_dirs:
        raise RuntimeError(f"No dataset directories with {args.data_glob} found under {exports_root}")

    print(f"Discovered {len(dataset_dirs)} dataset directories under {exports_root}", flush=True)
    failures: list[tuple[str, int]] = []
    for dataset_dir in dataset_dirs:
        command = build_command(args, dataset_dir, trainer_args)
        print(f"\n[{dataset_dir.name}]", flush=True)
        print(" ".join(shlex.quote(part) for part in command), flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, check=False, cwd=str(SCRIPT_DIR))
        if result.returncode == 0:
            continue
        failures.append((dataset_dir.name, result.returncode))
        if not args.continue_on_error:
            raise subprocess.CalledProcessError(result.returncode, command)

    if failures:
        summary = ", ".join(f"{name} (exit {code})" for name, code in failures)
        raise RuntimeError(f"One or more dataset runs failed: {summary}")


if __name__ == "__main__":
    main()