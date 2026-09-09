#!/usr/bin/env python
"""
Create deterministic train/val/test splits for every export dataset folder.

For each dataset directory under the exports root, this script uses
cell_annotation_rationale.jsonl as the anchor file to build a stratified split
by label, then applies the resulting sample_id -> split assignment consistently
across every JSONL file in that same dataset directory.

Outputs are written next to the source files with suffixes:
  *_train.jsonl
  *_val.jsonl
  *_test.jsonl

It also writes split manifests:
  split_manifest.json
  split_summary.json

Example:
python dataset_pipeline/create_consistent_dataset_splits.py \
    --exports-root dataset_pipeline/data/exports \
  --seed 42 \
  --train-ratio 0.8 \
  --val-ratio 0.1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EXPORTS_ROOT = SCRIPT_DIR / "data" / "exports"
ANCHOR_FILENAME = "cell_annotation_rationale.jsonl"
SPLIT_NAMES = ("train", "val", "test")
INVALID_JSON_ESCAPE_RE = re.compile(r'\\([^"\\/bfnrtu])')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic train/val/test splits for export datasets")
    parser.add_argument(
        "--exports-root",
        type=str,
        default=str(DEFAULT_EXPORTS_ROOT),
        help="Root directory containing dataset export folders",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for stratified splitting")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio")
    parser.add_argument(
        "--label-key",
        type=str,
        default="cell_type",
        help="Metadata label key used for stratified anchor splitting",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.jsonl",
        help="JSONL glob to split inside each dataset directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing split files and manifests if they already exist",
    )
    return parser.parse_args()


def parse_label_from_answer(answer_text: str | None) -> str | None:
    if not answer_text:
        return None
    for raw_line in str(answer_text).splitlines():
        line = raw_line.strip()
        if line.startswith("FINAL:") or line.startswith("LABEL:"):
            return line.split(":", 1)[1].strip()
    text = str(answer_text).strip()
    return text or None


def discover_dataset_dirs(exports_root: Path, glob_pattern: str) -> list[Path]:
    dataset_dirs = []
    for child in sorted(path for path in exports_root.iterdir() if path.is_dir()):
        if any(child.glob(glob_pattern)):
            dataset_dirs.append(child)
    return dataset_dirs


def load_jsonl_records(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    repaired_line = INVALID_JSON_ESCAPE_RE.sub(r"\1", line)
                    try:
                        records.append(json.loads(repaired_line))
                    except json.JSONDecodeError as error:
                        raise ValueError(f"Failed to parse JSONL record in {path}:{line_number}: {error}") from error
    return records


def build_anchor_entries(path: Path, label_key: str) -> list[tuple[str, str]]:
    entries = []
    seen_sample_ids = set()
    for record in load_jsonl_records(path):
        sample_id = record.get("sample_id")
        if not sample_id or sample_id in seen_sample_ids:
            continue
        seen_sample_ids.add(sample_id)
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        label = metadata.get(label_key) or metadata.get("cell_type") or record.get("label")
        if not label:
            label = parse_label_from_answer(record.get("answer") or record.get("answer_text"))
        if not label:
            continue
        entries.append((str(sample_id), str(label)))
    return entries


def stratified_split_sample_ids(
    entries: list[tuple[str, str]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, str]:
    by_label: dict[str, list[str]] = {}
    for sample_id, label in entries:
        by_label.setdefault(label, []).append(sample_id)

    rng = random.Random(seed)
    split_map: dict[str, str] = {}
    for label_sample_ids in by_label.values():
        rng.shuffle(label_sample_ids)
        count = len(label_sample_ids)
        train_count = max(1, int(count * train_ratio))
        val_count = 0 if count < 10 else max(1, int(count * val_ratio))
        if train_count + val_count > count:
            val_count = max(0, count - train_count)
        test_count = count - train_count - val_count
        if count >= 3 and test_count == 0 and train_count > 1:
            train_count -= 1
            test_count += 1
        if count >= 10 and val_count == 0 and train_count > 1:
            train_count -= 1
            val_count += 1

        for sample_id in label_sample_ids[:train_count]:
            split_map[sample_id] = "train"
        for sample_id in label_sample_ids[train_count : train_count + val_count]:
            split_map[sample_id] = "val"
        for sample_id in label_sample_ids[train_count + val_count : train_count + val_count + test_count]:
            split_map[sample_id] = "test"
    return split_map


def stable_fallback_split(sample_id: str, train_ratio: float, val_ratio: float) -> str:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    fraction = int(digest[:12], 16) / float(16**12 - 1)
    if fraction < train_ratio:
        return "train"
    if fraction < train_ratio + val_ratio:
        return "val"
    return "test"


def write_jsonl(path: Path, records: list[dict], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def output_split_path(source_path: Path, split_name: str) -> Path:
    return source_path.with_name(f"{source_path.stem}_{split_name}.jsonl")


def split_dataset_dir(dataset_dir: Path, args: argparse.Namespace) -> dict:
    anchor_path = dataset_dir / ANCHOR_FILENAME
    if not anchor_path.exists():
        raise FileNotFoundError(f"Anchor file not found: {anchor_path}")

    anchor_entries = build_anchor_entries(anchor_path, args.label_key)
    if not anchor_entries:
        raise RuntimeError(f"No valid anchor entries found in {anchor_path}")

    split_map = stratified_split_sample_ids(
        anchor_entries,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    source_files = sorted(
        path for path in dataset_dir.glob(args.glob) if path.is_file() and not path.stem.endswith(("_train", "_val", "_test"))
    )
    dataset_summary = {
        "dataset_dir": str(dataset_dir),
        "anchor_path": str(anchor_path),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "anchor_unique_sample_ids": len(split_map),
        "anchor_split_counts": dict(Counter(split_map.values())),
        "files": {},
    }

    for source_path in source_files:
        records = load_jsonl_records(source_path)
        split_records = {split_name: [] for split_name in SPLIT_NAMES}
        fallback_count = 0
        missing_sample_id_count = 0
        for record in records:
            sample_id = record.get("sample_id")
            if not sample_id:
                missing_sample_id_count += 1
                continue
            split_name = split_map.get(sample_id)
            if split_name is None:
                split_name = stable_fallback_split(str(sample_id), args.train_ratio, args.val_ratio)
                fallback_count += 1
            split_records[split_name].append(record)

        file_summary = {
            "source": str(source_path),
            "total_records": len(records),
            "written_records": sum(len(split_records[split_name]) for split_name in SPLIT_NAMES),
            "missing_sample_id_records": missing_sample_id_count,
            "fallback_sample_assignments": fallback_count,
            "splits": {},
        }
        for split_name in SPLIT_NAMES:
            split_path = output_split_path(source_path, split_name)
            write_jsonl(split_path, split_records[split_name], overwrite=args.overwrite)
            file_summary["splits"][split_name] = {
                "path": str(split_path),
                "count": len(split_records[split_name]),
            }
        dataset_summary["files"][source_path.name] = file_summary

    manifest_path = dataset_dir / "split_manifest.json"
    summary_path = dataset_dir / "split_summary.json"
    write_jsonl_compatible = {"manifest_path": str(manifest_path), "summary_path": str(summary_path)}
    if (manifest_path.exists() or summary_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing split manifest in {dataset_dir} without --overwrite"
        )
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "sample_id_to_split": split_map,
                "config": {
                    "seed": args.seed,
                    "train_ratio": args.train_ratio,
                    "val_ratio": args.val_ratio,
                    "label_key": args.label_key,
                    "glob": args.glob,
                },
            },
            handle,
            indent=2,
        )
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(dataset_summary, handle, indent=2)
    dataset_summary.update(write_jsonl_compatible)
    return dataset_summary


def main() -> None:
    args = parse_args()
    exports_root = Path(args.exports_root).resolve()
    if not exports_root.exists():
        raise FileNotFoundError(f"Exports root does not exist: {exports_root}")

    dataset_dirs = discover_dataset_dirs(exports_root, args.glob)
    if not dataset_dirs:
        raise RuntimeError(f"No dataset directories with {args.glob} found under {exports_root}")

    all_summaries = []
    for dataset_dir in dataset_dirs:
        summary = split_dataset_dir(dataset_dir, args)
        all_summaries.append(summary)
        print(
            f"[Split] {dataset_dir.name}: "
            f"anchor={summary['anchor_unique_sample_ids']} "
            f"train={summary['anchor_split_counts'].get('train', 0)} "
            f"val={summary['anchor_split_counts'].get('val', 0)} "
            f"test={summary['anchor_split_counts'].get('test', 0)}",
            flush=True,
        )

    aggregate_path = exports_root / "split_generation_summary.json"
    if aggregate_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {aggregate_path}")
    with open(aggregate_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "exports_root": str(exports_root),
                "dataset_count": len(all_summaries),
                "datasets": all_summaries,
            },
            handle,
            indent=2,
        )
    print(f"[Done] Wrote summary to {aggregate_path}", flush=True)


if __name__ == "__main__":
    main()