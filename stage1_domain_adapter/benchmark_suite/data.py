from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DATASET_FILE_STEM = "cell_annotation_rationale"
STANDARD_FAMILY = "standard"
GROUPED_FAMILY = "grouped"
SPLIT_FAMILY_ALIASES = {
    "standard": STANDARD_FAMILY,
    "normal": STANDARD_FAMILY,
    "cell": STANDARD_FAMILY,
    "cell_level": STANDARD_FAMILY,
    "grouped": GROUPED_FAMILY,
    "group": GROUPED_FAMILY,
    "donor": GROUPED_FAMILY,
}


@dataclass(slots=True)
class SplitData:
    split_name: str
    split_family: str
    jsonl_path: Path
    genes_lists: list[list[str]]
    labels: list[str]
    sample_ids: list[str]
    metadata: list[dict]

    @property
    def n_samples(self) -> int:
        return len(self.labels)

    @property
    def n_labels(self) -> int:
        return len(set(self.labels))


def normalize_split_family(split_family: str) -> str:
    normalized = SPLIT_FAMILY_ALIASES.get(split_family.lower())
    if normalized is None:
        supported = ", ".join(sorted(SPLIT_FAMILY_ALIASES))
        raise ValueError(f"Unsupported split family '{split_family}'. Supported aliases: {supported}")
    return normalized


def discover_dataset_dirs(exports_root: str | Path) -> list[Path]:
    root = Path(exports_root)
    dataset_dirs: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        anchor = child / f"{DATASET_FILE_STEM}.jsonl"
        if anchor.is_file():
            dataset_dirs.append(child)
    return dataset_dirs


def filter_dataset_dirs(dataset_dirs: Iterable[Path], patterns: list[str] | None) -> list[Path]:
    if not patterns:
        return list(dataset_dirs)
    lowered = [pattern.lower() for pattern in patterns]
    filtered: list[Path] = []
    for dataset_dir in dataset_dirs:
        name = dataset_dir.name.lower()
        if any(pattern in name for pattern in lowered):
            filtered.append(dataset_dir)
    return filtered


def resolve_split_path(dataset_dir: str | Path, split_name: str, split_family: str) -> Path:
    dataset_path = Path(dataset_dir)
    family = normalize_split_family(split_family)
    family_tag = "" if family == STANDARD_FAMILY else "_grouped"
    split_path = dataset_path / f"{DATASET_FILE_STEM}{family_tag}_{split_name}.jsonl"
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    return split_path


def load_dataset_splits(
    dataset_dir: str | Path,
    split_family: str,
    splits: Iterable[str],
    top_genes: int = 200,
    label_key: str = "cell_type",
    max_cells_per_split: int | None = None,
) -> dict[str, SplitData]:
    dataset_path = Path(dataset_dir)
    records: dict[str, SplitData] = {}
    for split_name in splits:
        split_path = resolve_split_path(dataset_path, split_name, split_family)
        split_data = load_split_jsonl(
            split_path,
            split_name=split_name,
            split_family=normalize_split_family(split_family),
            top_genes=top_genes,
            label_key=label_key,
            max_cells=max_cells_per_split,
        )
        records[split_name] = split_data
    return records


def load_grouped_split_summary(dataset_dir: str | Path) -> dict | None:
    path = Path(dataset_dir) / "grouped_split_summary.json"
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_split_jsonl(
    jsonl_path: str | Path,
    split_name: str,
    split_family: str,
    top_genes: int = 200,
    label_key: str = "cell_type",
    max_cells: int | None = None,
) -> SplitData:
    path = Path(jsonl_path)
    genes_lists: list[list[str]] = []
    labels: list[str] = []
    sample_ids: list[str] = []
    metadata_rows: list[dict] = []

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            genes = record.get("genes")
            if not isinstance(genes, list):
                question_text = record.get("question") or record.get("question_text") or ""
                genes = extract_genes_from_question(question_text)
            genes = [str(gene) for gene in genes[:top_genes]]

            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            label = metadata.get(label_key) or metadata.get("cell_type") or record.get("label")
            if not label:
                label = parse_label_from_answer(record.get("answer") or record.get("answer_text"))

            if not genes or not label:
                continue

            sample_id = str(
                record.get("sample_id")
                or record.get("id")
                or metadata.get("sample_id")
                or f"{path.stem}:{len(labels)}"
            )
            genes_lists.append(genes)
            labels.append(str(label))
            sample_ids.append(sample_id)
            metadata_rows.append(metadata)

            if max_cells is not None and len(labels) >= max_cells:
                break

    return SplitData(
        split_name=split_name,
        split_family=split_family,
        jsonl_path=path,
        genes_lists=genes_lists,
        labels=labels,
        sample_ids=sample_ids,
        metadata=metadata_rows,
    )


def parse_label_from_answer(answer_text: str | None) -> str | None:
    if not answer_text:
        return None
    for raw_line in str(answer_text).splitlines():
        line = raw_line.strip()
        if line.upper().startswith("FINAL:"):
            return line.split(":", 1)[1].strip()
        if line.upper().startswith("LABEL:"):
            return line.split(":", 1)[1].strip()
    return None


def extract_genes_from_question(question_text: str) -> list[str]:
    cell_sentence_markers = ["Cell sentence:\n", "Cell sentence:"]
    for marker in cell_sentence_markers:
        start = question_text.find(marker)
        if start == -1:
            continue
        after = question_text[start + len(marker) :]
        for stopper in ("\nQuestion:", "\nThe cell type"):
            stop = after.find(stopper)
            if stop != -1:
                after = after[:stop]
                break
        return after.strip().split()
    return []
