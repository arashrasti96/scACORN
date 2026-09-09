"""Common data loading and splitting for comparison methods.

Reads JSONL exports produced by the dataset_pipeline and produces
(genes_list, labels) tuples with the same stratified split logic used
in the contrastive training scripts.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

# Default exports directory (relative to this file)
_DEFAULT_EXPORTS_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "dataset_pipeline"
    / "data"
    / "exports"
)


# ---------------------------------------------------------------------------
# Record loading
# ---------------------------------------------------------------------------

def _parse_label_from_answer(answer_text: str | None) -> str | None:
    """Fall-back: extract cell-type label from the structured answer string."""
    if not answer_text:
        return None
    for line in answer_text.splitlines():
        line = line.strip()
        if line.upper().startswith("FINAL:"):
            return line.split(":", 1)[1].strip()
        if line.upper().startswith("LABEL:"):
            return line.split(":", 1)[1].strip()
    return None


def load_records(
    jsonl_path: str | Path,
    top_genes: int = 200,
    label_key: str = "cell_type",
) -> list[tuple[list[str], str]]:
    """Load ``(genes, label)`` pairs from a JSONL export file.

    Parameters
    ----------
    jsonl_path:
        Path to the ``cell_annotation_rationale.jsonl`` file.
    top_genes:
        Max number of genes to keep per cell.
    label_key:
        Metadata key used for labels (falls back to ``cell_type``).
    """
    records: list[tuple[list[str], str]] = []
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)

            genes = rec.get("genes")
            if isinstance(genes, list):
                genes = [str(g) for g in genes][:top_genes]
            else:
                genes = _extract_genes_from_question(
                    rec.get("question") or rec.get("question_text") or ""
                )[:top_genes]

            meta = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
            label = (
                meta.get(label_key)
                or meta.get("cell_type")
                or rec.get("label")
            )
            if not label:
                label = _parse_label_from_answer(rec.get("answer") or rec.get("answer_text"))

            if genes and label:
                records.append((genes, str(label)))
    return records


def _extract_genes_from_question(question_text: str) -> list[str]:
    """Extract ordered gene list from the dataset question field."""
    cs_marker = "Cell sentence:\n"
    cs_start = question_text.find(cs_marker)
    if cs_start == -1:
        cs_marker = "Cell sentence:"
        cs_start = question_text.find(cs_marker)
    if cs_start == -1:
        return []
    after = question_text[cs_start + len(cs_marker) :]
    for stopper in ["\nQuestion:", "\nThe cell type"]:
        idx = after.find(stopper)
        if idx != -1:
            after = after[:idx]
            break
    return after.strip().split()


# ---------------------------------------------------------------------------
# Stratified splitting
# ---------------------------------------------------------------------------

def stratified_split(
    records: list[tuple[list[str], str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[tuple[list[str], str]]]:
    """Split records into train / val / test while preserving label proportions.

    Mirrors the logic in ``train_stage1_domain_lora._stratified_split_records``.
    """
    by_label: dict[str, list[tuple[list[str], str]]] = {}
    for genes, label in records:
        by_label.setdefault(label, []).append((genes, label))

    rng = random.Random(seed)
    splits: dict[str, list[tuple[list[str], str]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for label_recs in by_label.values():
        rng.shuffle(label_recs)
        n = len(label_recs)
        n_train = max(1, int(n * train_ratio))
        n_val = 0 if n < 10 else max(1, int(n * val_ratio))
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        n_test = n - n_train - n_val
        if n >= 3 and n_test == 0 and n_train > 1:
            n_train -= 1
            n_test += 1
        if n >= 10 and n_val == 0 and n_train > 1:
            n_train -= 1
            n_val += 1

        splits["train"].extend(label_recs[:n_train])
        splits["val"].extend(label_recs[n_train : n_train + n_val])
        splits["test"].extend(label_recs[n_train + n_val : n_train + n_val + n_test])
    return splits


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def discover_datasets(
    exports_root: str | Path | None = None,
) -> list[tuple[str, Path]]:
    """Find all dataset export directories containing JSONL files.

    Returns a sorted list of ``(dataset_name, jsonl_path)`` tuples.
    """
    root = Path(exports_root) if exports_root else _DEFAULT_EXPORTS_ROOT
    datasets: list[tuple[str, Path]] = []
    for subdir in sorted(root.iterdir()):
        jsonl = subdir / "cell_annotation_rationale.jsonl"
        if jsonl.is_file():
            datasets.append((subdir.name, jsonl))
    return datasets


# ---------------------------------------------------------------------------
# Convenience: load & split in one call
# ---------------------------------------------------------------------------

def load_and_split(
    jsonl_path: str | Path,
    top_genes: int = 200,
    label_key: str = "cell_type",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, tuple[list[list[str]], list[str]]]:
    """Load + split, returning ``{split: (genes_lists, labels)}`` per split."""
    records = load_records(jsonl_path, top_genes=top_genes, label_key=label_key)
    split_records = stratified_split(records, train_ratio, val_ratio, seed)
    result: dict[str, tuple[list[list[str]], list[str]]] = {}
    for split_name, recs in split_records.items():
        genes_lists = [g for g, _ in recs]
        labels = [l for _, l in recs]
        result[split_name] = (genes_lists, labels)
    return result
