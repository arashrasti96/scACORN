#!/usr/bin/env python
"""
Create deterministic grouped train/val/test splits for every export dataset folder.

Unlike create_consistent_dataset_splits.py, this script does not split at the
individual-cell level when stronger RNA-seq evaluation is desired. It resolves
each cell into a higher-level group such as donor, specimen, assay, or batch,
then assigns the entire group to one split.

The split is approximate-stratified by label at the group level, which is more
defensible for single-cell RNA-seq than random cell-level splitting because it
reduces leakage across highly correlated cells from the same donor or library.

Outputs are written next to the source files with suffixes:
  *_grouped_train.jsonl
  *_grouped_val.jsonl
  *_grouped_test.jsonl

It also writes grouped split manifests:
  grouped_split_manifest.json
  grouped_split_summary.json

Example:
python dataset_pipeline/create_grouped_dataset_splits.py \
    --exports-root dataset_pipeline/data/exports \
  --seed 42 \
  --train-ratio 0.8 \
  --val-ratio 0.1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EXPORTS_ROOT = SCRIPT_DIR / "data" / "exports"
ANCHOR_FILENAME = "cell_annotation_rationale.jsonl"
SPLIT_NAMES = ("train", "val", "test")
DEFAULT_GROUP_KEYS = (
    "donor_id",
    "specimen_id",
    "donor_tissue_assay",
    "donor_assay",
    "donor_tissue",
    "library_id",
    "10X_run",
    "sample_number",
    "batch",
    "_scvi_batch",
    "replicate",
    "assay",
    "method",
)
INVALID_JSON_ESCAPE_RE = re.compile(r'\\([^"\\/bfnrtu])')


@dataclass(frozen=True)
class GroupResolution:
    group_id: str
    group_key: str
    group_value: str


@dataclass(frozen=True)
class GroupStrategy:
    strategy_name: str
    active_keys: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic grouped train/val/test splits for export datasets")
    parser.add_argument(
        "--exports-root",
        type=str,
        default=str(DEFAULT_EXPORTS_ROOT),
        help="Root directory containing dataset export folders",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for grouped splitting")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio")
    parser.add_argument(
        "--label-key",
        type=str,
        default="cell_type",
        help="Metadata label key used for grouped anchor splitting",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.jsonl",
        help="JSONL glob to split inside each dataset directory",
    )
    parser.add_argument(
        "--group-keys",
        type=str,
        default=",".join(DEFAULT_GROUP_KEYS),
        help="Comma-separated metadata keys used to define grouped holdouts",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default="grouped",
        help="Tag inserted into output filenames and manifests",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing grouped split files and manifests if they already exist",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")
    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")
    if not args.output_tag:
        raise ValueError("--output-tag must not be empty")


def parse_label_from_answer(answer_text: str | None) -> str | None:
    if not answer_text:
        return None
    for raw_line in str(answer_text).splitlines():
        line = raw_line.strip()
        if line.startswith("FINAL:") or line.startswith("LABEL:"):
            return line.split(":", 1)[1].strip()
    text = str(answer_text).strip()
    return text or None


def clean_value(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"nan", "none", "null"}:
        return None
    return text


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


def parse_group_keys(raw_group_keys: str) -> list[str]:
    return [key.strip() for key in raw_group_keys.split(",") if key.strip()]


def infer_group_from_record_sample_id(sample_id: str | None) -> str | None:
    sample_text = clean_value(sample_id)
    if not sample_text:
        return None
    suffix = sample_text.split(":", 1)[-1]
    if "_" not in suffix:
        return suffix
    return suffix.rsplit("_", 1)[0]


def extract_group_values(record: dict, group_keys: list[str]) -> dict[str, str]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    values: dict[str, str] = {}
    for key in group_keys:
        value = clean_value(metadata.get(key))
        if value:
            values[key] = value
    return values


def resolve_group_from_values(
    group_values: dict[str, str],
    sample_id: str | None,
    strategy: GroupStrategy,
) -> GroupResolution:
    resolved_pairs = [(key, group_values[key]) for key in strategy.active_keys if key in group_values]
    if resolved_pairs:
        if len(resolved_pairs) == 1:
            key, value = resolved_pairs[0]
            return GroupResolution(group_id=f"{key}:{value}", group_key=key, group_value=value)
        group_id = "|".join(f"{key}={value}" for key, value in resolved_pairs)
        return GroupResolution(
            group_id=group_id,
            group_key="+".join(key for key, _ in resolved_pairs),
            group_value=" | ".join(value for _, value in resolved_pairs),
        )

    inferred_value = infer_group_from_record_sample_id(sample_id)
    if inferred_value:
        return GroupResolution(
            group_id=f"sample_prefix:{inferred_value}",
            group_key="sample_prefix",
            group_value=inferred_value,
        )

    sample_value = clean_value(sample_id)
    if not sample_value:
        raise ValueError("Record does not define a usable sample_id for grouped splitting")
    return GroupResolution(group_id=f"sample_id:{sample_value}", group_key="sample_id", group_value=sample_value)


def build_anchor_entries(path: Path, label_key: str, group_keys: list[str]) -> list[dict]:
    entries = []
    seen_sample_ids = set()
    for record in load_jsonl_records(path):
        sample_id = clean_value(record.get("sample_id"))
        if not sample_id or sample_id in seen_sample_ids:
            continue
        seen_sample_ids.add(sample_id)
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        label = metadata.get(label_key) or metadata.get("cell_type") or record.get("label")
        if not label:
            label = parse_label_from_answer(record.get("answer") or record.get("answer_text"))
        label = clean_value(label)
        if not label:
            continue
        entries.append(
            {
                "sample_id": sample_id,
                "label": label,
                "group_values": extract_group_values(record, group_keys),
            }
        )
    return entries


def target_counts(total_count: int, train_ratio: float, val_ratio: float) -> dict[str, int]:
    train_count = max(1, int(total_count * train_ratio))
    val_count = 0 if total_count < 10 else max(1, int(total_count * val_ratio))
    if train_count + val_count > total_count:
        val_count = max(0, total_count - train_count)
    test_count = total_count - train_count - val_count
    if total_count >= 3 and test_count == 0 and train_count > 1:
        train_count -= 1
        test_count += 1
    if total_count >= 10 and val_count == 0 and train_count > 1:
        train_count -= 1
        val_count += 1
    return {"train": train_count, "val": val_count, "test": test_count}


def order_group_ids(groups: dict[str, dict], label_totals: dict[str, int], seed: int) -> list[str]:
    rng = random.Random(seed)
    group_ids = list(groups)
    rng.shuffle(group_ids)

    def rarity_score(group_id: str) -> float:
        label_counts = groups[group_id]["label_counts"]
        score = 0.0
        for label, count in label_counts.items():
            score += count / max(1, label_totals.get(label, 1))
        return score

    return sorted(
        group_ids,
        key=lambda group_id: (
            -max(groups[group_id]["label_counts"].values()),
            -rarity_score(group_id),
            -len(groups[group_id]["sample_ids"]),
            group_id,
        ),
    )


def build_split_targets(label_totals: dict[str, int], train_ratio: float, val_ratio: float) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    label_targets = {
        label: target_counts(total_count, train_ratio=train_ratio, val_ratio=val_ratio)
        for label, total_count in label_totals.items()
    }
    total_targets = target_counts(sum(label_totals.values()), train_ratio=train_ratio, val_ratio=val_ratio)
    return label_targets, total_targets


def choose_best_split(
    group_payload: dict,
    split_label_counts: dict[str, Counter],
    split_totals: dict[str, int],
    label_targets: dict[str, dict[str, int]],
    total_targets: dict[str, int],
    total_samples: int,
) -> str:
    best_split = SPLIT_NAMES[0]
    best_cost = math.inf
    group_size = len(group_payload["sample_ids"])

    for split_name in SPLIT_NAMES:
        target_total = max(1, total_targets[split_name])
        current_total = split_totals[split_name]
        updated_total = current_total + group_size

        # Prefer assigning into the most underfilled split relative to its target,
        # and heavily penalize overshooting small validation/test quotas.
        total_fill_ratio = current_total / target_total
        total_overshoot_ratio = max(0, updated_total - total_targets[split_name]) / target_total
        cost = total_fill_ratio + (4.0 * total_overshoot_ratio * total_overshoot_ratio)

        if current_total == 0 and total_targets[split_name] > 0:
            cost -= 0.15

        for label, group_count in group_payload["label_counts"].items():
            target = label_targets[label][split_name]
            current_label_count = split_label_counts[split_name][label]

            if target <= 0:
                cost += 0.15 * (group_count / max(1, group_size))
                continue

            updated = current_label_count + group_count
            label_fill_ratio = current_label_count / target
            label_overshoot_ratio = max(0, updated - target) / target
            cost += (group_count / max(1, group_size)) * (
                0.35 * label_fill_ratio + (1.5 * label_overshoot_ratio * label_overshoot_ratio)
            )

        # Break near-ties toward the split with the larger remaining quota.
        remaining_quota = total_targets[split_name] - current_total
        cost -= max(0, remaining_quota) / max(1, total_samples) * 0.05

        if cost < best_cost:
            best_cost = cost
            best_split = split_name
    return best_split


def grouped_stratified_split_sample_ids(
    entries: list[dict],
    strategy: GroupStrategy,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[dict[str, str], dict[str, str], dict[str, dict]]:
    if not entries:
        return {}, {}, {}

    groups: dict[str, dict] = {}
    label_totals = Counter()
    for entry in entries:
        group_resolution = resolve_group_from_values(entry["group_values"], entry["sample_id"], strategy)
        group_payload = groups.setdefault(
            group_resolution.group_id,
            {
                "group_key": group_resolution.group_key,
                "group_value": group_resolution.group_value,
                "sample_ids": [],
                "label_counts": Counter(),
            },
        )
        group_payload["sample_ids"].append(entry["sample_id"])
        group_payload["label_counts"][entry["label"]] += 1
        label_totals[entry["label"]] += 1

    label_targets, total_targets = build_split_targets(dict(label_totals), train_ratio=train_ratio, val_ratio=val_ratio)
    split_label_counts = {split_name: Counter() for split_name in SPLIT_NAMES}
    split_totals = {split_name: 0 for split_name in SPLIT_NAMES}
    group_to_split: dict[str, str] = {}

    ordered_group_ids = order_group_ids(groups, dict(label_totals), seed=seed)
    total_samples = sum(len(group_payload["sample_ids"]) for group_payload in groups.values())

    for group_id in ordered_group_ids:
        split_name = choose_best_split(
            groups[group_id],
            split_label_counts=split_label_counts,
            split_totals=split_totals,
            label_targets=label_targets,
            total_targets=total_targets,
            total_samples=total_samples,
        )
        group_to_split[group_id] = split_name
        split_totals[split_name] += len(groups[group_id]["sample_ids"])
        split_label_counts[split_name].update(groups[group_id]["label_counts"])

    sample_to_split: dict[str, str] = {}
    sample_to_group: dict[str, str] = {}
    for group_id, group_payload in groups.items():
        split_name = group_to_split[group_id]
        for sample_id in group_payload["sample_ids"]:
            sample_to_split[sample_id] = split_name
            sample_to_group[sample_id] = group_id
    return sample_to_split, group_to_split, sample_to_group


def summarize_strategy_counts(entries: list[dict], strategy: GroupStrategy) -> tuple[int, Counter]:
    group_ids = Counter()
    for entry in entries:
        group_resolution = resolve_group_from_values(entry["group_values"], entry["sample_id"], strategy)
        group_ids[group_resolution.group_id] += 1
    return len(group_ids), group_ids


def score_split_counts(anchor_split_counts: Counter, total_count: int, train_ratio: float, val_ratio: float) -> float:
    target_ratio = {
        "train": train_ratio,
        "val": val_ratio,
        "test": max(0.0, 1.0 - train_ratio - val_ratio),
    }
    score = 0.0
    for split_name in SPLIT_NAMES:
        observed = anchor_split_counts.get(split_name, 0) / max(1, total_count)
        score += (observed - target_ratio[split_name]) ** 2
        if anchor_split_counts.get(split_name, 0) == 0:
            score += 10.0
    return score


def classify_split_quality(max_ratio_error: float) -> str:
    if max_ratio_error <= 0.03:
        return "high"
    if max_ratio_error <= 0.08:
        return "moderate"
    return "coarse"


def choose_group_strategy(
    entries: list[dict],
    group_keys: list[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[GroupStrategy, dict[str, object]]:
    available_key_counts: dict[str, int] = {}
    for key in group_keys:
        distinct_values = {entry["group_values"][key] for entry in entries if key in entry["group_values"]}
        if distinct_values:
            available_key_counts[key] = len(distinct_values)

    candidate_strategies: list[GroupStrategy] = []
    for key in ("donor_id", "specimen_id", "donor_tissue_assay", "donor_assay", "donor_tissue", "library_id", "10X_run"):
        if key in available_key_counts:
            candidate_strategies.append(GroupStrategy(strategy_name=key, active_keys=(key,)))

    composite_keys = tuple(
        key
        for key in (
            "donor_id",
            "specimen_id",
            "donor_tissue_assay",
            "donor_assay",
            "donor_tissue",
            "library_id",
            "10X_run",
            "sample_number",
            "replicate",
            "assay",
            "batch",
            "_scvi_batch",
            "method",
        )
        if key in available_key_counts
    )
    if composite_keys:
        candidate_strategies.append(GroupStrategy(strategy_name="composite_metadata", active_keys=composite_keys))

    candidate_strategies.append(GroupStrategy(strategy_name="sample_prefix", active_keys=tuple()))

    total_count = len(entries)
    candidate_reports: list[dict[str, object]] = []
    viable_candidates: list[tuple[GroupStrategy, dict[str, object]]] = []
    nonempty_candidates: list[tuple[GroupStrategy, dict[str, object]]] = []
    all_candidates: list[tuple[GroupStrategy, dict[str, object]]] = []

    def candidate_rank(item: tuple[GroupStrategy, dict[str, object]]) -> tuple[float, float, int]:
        _, report = item
        return (
            float(report["score"]),
            float(report["max_ratio_error"]),
            -int(report["unique_groups"]),
        )

    for strategy in candidate_strategies:
        sample_to_split, group_to_split, _ = grouped_stratified_split_sample_ids(
            entries,
            strategy=strategy,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
        anchor_split_counts = Counter(sample_to_split.values())
        unique_groups, _ = summarize_strategy_counts(entries, strategy)
        max_ratio_error = max(
            abs(anchor_split_counts.get("train", 0) / max(1, total_count) - train_ratio),
            abs(anchor_split_counts.get("val", 0) / max(1, total_count) - val_ratio),
            abs(anchor_split_counts.get("test", 0) / max(1, total_count) - (1.0 - train_ratio - val_ratio)),
        )
        report = {
            "strategy_name": strategy.strategy_name,
            "active_keys": list(strategy.active_keys),
            "unique_groups": unique_groups,
            "anchor_split_counts": dict(anchor_split_counts),
            "group_split_counts": dict(Counter(group_to_split.values())),
            "max_ratio_error": max_ratio_error,
            "score": score_split_counts(anchor_split_counts, total_count, train_ratio, val_ratio),
        }
        candidate_reports.append(report)
        all_candidates.append((strategy, report))

        has_nonempty_splits = all(anchor_split_counts.get(split_name, 0) > 0 for split_name in SPLIT_NAMES)
        if has_nonempty_splits:
            nonempty_candidates.append((strategy, report))

        if unique_groups >= 8 and has_nonempty_splits and max_ratio_error <= 0.18:
            viable_candidates.append((strategy, report))

    if viable_candidates:
        chosen_strategy, chosen_report = min(viable_candidates, key=candidate_rank)
        selection_mode = "best_viable_candidate"
    elif nonempty_candidates:
        chosen_strategy, chosen_report = min(nonempty_candidates, key=candidate_rank)
        selection_mode = "best_nonempty_fallback"
    else:
        chosen_strategy, chosen_report = min(all_candidates, key=candidate_rank)
        selection_mode = "best_available_fallback"

    return chosen_strategy, {
        "available_key_counts": available_key_counts,
        "selected_strategy": chosen_report,
        "selection_mode": selection_mode,
        "evaluated_candidates": candidate_reports,
    }


def stable_fallback_split(group_id: str, train_ratio: float, val_ratio: float) -> str:
    digest = hashlib.sha256(group_id.encode("utf-8")).hexdigest()
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


def output_split_path(source_path: Path, output_tag: str, split_name: str) -> Path:
    return source_path.with_name(f"{source_path.stem}_{output_tag}_{split_name}.jsonl")


def split_dataset_dir(dataset_dir: Path, args: argparse.Namespace) -> dict:
    anchor_path = dataset_dir / ANCHOR_FILENAME
    if not anchor_path.exists():
        raise FileNotFoundError(f"Anchor file not found: {anchor_path}")

    group_keys = parse_group_keys(args.group_keys)
    anchor_entries = build_anchor_entries(anchor_path, args.label_key, group_keys)
    if not anchor_entries:
        raise RuntimeError(f"No valid anchor entries found in {anchor_path}")

    strategy, strategy_summary = choose_group_strategy(
        anchor_entries,
        group_keys=group_keys,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    sample_to_split, group_to_split, sample_to_group = grouped_stratified_split_sample_ids(
        anchor_entries,
        strategy=strategy,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    resolved_group_keys = Counter()
    for entry in anchor_entries:
        resolved = resolve_group_from_values(entry["group_values"], entry["sample_id"], strategy)
        resolved_group_keys[resolved.group_key] += 1
    group_split_counts = Counter(group_to_split.values())
    anchor_split_counts = Counter(sample_to_split.values())
    source_files = sorted(
        path
        for path in dataset_dir.glob(args.glob)
        if path.is_file() and not path.stem.endswith(("_train", "_val", "_test"))
    )
    dataset_summary = {
        "dataset_dir": str(dataset_dir),
        "anchor_path": str(anchor_path),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "group_keys": group_keys,
        "selected_group_strategy": strategy.strategy_name,
        "selected_group_strategy_keys": list(strategy.active_keys),
        "group_strategy_summary": strategy_summary,
        "selection_mode": strategy_summary.get("selection_mode", "unknown"),
        "max_ratio_error": strategy_summary.get("selected_strategy", {}).get("max_ratio_error", 1.0),
        "split_quality": classify_split_quality(
            float(strategy_summary.get("selected_strategy", {}).get("max_ratio_error", 1.0))
        ),
        "output_tag": args.output_tag,
        "anchor_unique_sample_ids": len(sample_to_split),
        "anchor_unique_groups": len(group_to_split),
        "anchor_group_source_counts": dict(resolved_group_keys),
        "anchor_group_split_counts": dict(group_split_counts),
        "anchor_split_counts": dict(anchor_split_counts),
        "files": {},
    }

    for source_path in source_files:
        records = load_jsonl_records(source_path)
        split_records = {split_name: [] for split_name in SPLIT_NAMES}
        missing_sample_id_count = 0
        reused_group_assignment_count = 0
        hashed_group_assignment_count = 0
        for record in records:
            sample_id = clean_value(record.get("sample_id"))
            if not sample_id:
                missing_sample_id_count += 1
                continue

            split_name = sample_to_split.get(sample_id)
            if split_name is None:
                group_resolution = resolve_group_from_values(
                    extract_group_values(record, group_keys),
                    sample_id,
                    strategy,
                )
                split_name = group_to_split.get(group_resolution.group_id)
                if split_name is not None:
                    reused_group_assignment_count += 1
                else:
                    split_name = stable_fallback_split(group_resolution.group_id, args.train_ratio, args.val_ratio)
                    hashed_group_assignment_count += 1
            split_records[split_name].append(record)

        file_summary = {
            "source": str(source_path),
            "total_records": len(records),
            "written_records": sum(len(split_records[split_name]) for split_name in SPLIT_NAMES),
            "missing_sample_id_records": missing_sample_id_count,
            "reused_group_assignments": reused_group_assignment_count,
            "hashed_group_assignments": hashed_group_assignment_count,
            "splits": {},
        }
        for split_name in SPLIT_NAMES:
            split_path = output_split_path(source_path, args.output_tag, split_name)
            write_jsonl(split_path, split_records[split_name], overwrite=args.overwrite)
            file_summary["splits"][split_name] = {
                "path": str(split_path),
                "count": len(split_records[split_name]),
            }
        dataset_summary["files"][source_path.name] = file_summary

    manifest_path = dataset_dir / f"{args.output_tag}_split_manifest.json"
    summary_path = dataset_dir / f"{args.output_tag}_split_summary.json"
    if (manifest_path.exists() or summary_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing grouped split manifest in {dataset_dir} without --overwrite"
        )
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "sample_id_to_split": sample_to_split,
                "sample_id_to_group": sample_to_group,
                "group_id_to_split": group_to_split,
                "config": {
                    "seed": args.seed,
                    "train_ratio": args.train_ratio,
                    "val_ratio": args.val_ratio,
                    "label_key": args.label_key,
                    "glob": args.glob,
                    "group_keys": group_keys,
                    "selected_group_strategy": strategy.strategy_name,
                    "selected_group_strategy_keys": list(strategy.active_keys),
                    "output_tag": args.output_tag,
                },
            },
            handle,
            indent=2,
        )
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(dataset_summary, handle, indent=2)
    dataset_summary.update({"manifest_path": str(manifest_path), "summary_path": str(summary_path)})
    return dataset_summary


def main() -> None:
    args = parse_args()
    validate_args(args)

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
            f"[GroupedSplit] {dataset_dir.name}: "
            f"strategy={summary['selected_group_strategy']} "
            f"groups={summary['anchor_unique_groups']} "
            f"cells={summary['anchor_unique_sample_ids']} "
            f"train={summary['anchor_split_counts'].get('train', 0)} "
            f"val={summary['anchor_split_counts'].get('val', 0)} "
            f"test={summary['anchor_split_counts'].get('test', 0)} "
            f"quality={summary['split_quality']} "
            f"mode={summary['selection_mode']}",
            flush=True,
        )

    print(
        json.dumps(
            {
                "datasets": len(all_summaries),
                "output_tag": args.output_tag,
                "exports_root": str(exports_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()