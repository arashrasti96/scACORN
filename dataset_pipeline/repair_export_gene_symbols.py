from __future__ import annotations

import argparse
import re
from pathlib import Path

from .io_utils import read_jsonl, write_jsonl
from .normalize import _map_ensembl_ids_to_symbols, _normalize_ensembl_gene_id


ENSEMBL_GENE_ID_RE = re.compile(r"^ENSG[0-9]+(?:\.[0-9]+)?$")


def repair_gene_list(genes: list[str], mapping: dict[str, str]) -> list[str]:
    repaired = []
    seen = set()
    for gene in genes:
        text = str(gene).strip()
        gene_id = _normalize_ensembl_gene_id(text)
        if gene_id and gene_id in mapping:
            text = mapping[gene_id]
        if not text or ENSEMBL_GENE_ID_RE.match(text) or text in seen:
            continue
        repaired.append(text)
        seen.add(text)
    return repaired


def repair_jsonl(path: Path) -> tuple[int, int]:
    rows = read_jsonl(path)
    candidate_values = []
    for row in rows:
        candidate_values.extend(str(gene) for gene in row.get("genes", []))
        candidate_values.extend(str(gene) for gene in row.get("evidence_genes", []))

    mapping = _map_ensembl_ids_to_symbols(candidate_values)
    changed_rows = 0
    unresolved_rows = 0

    for row in rows:
        original_genes = list(row.get("genes", []))
        repaired_genes = repair_gene_list(original_genes, mapping)
        if repaired_genes != original_genes:
            row["genes"] = repaired_genes
            changed_rows += 1
        if any(ENSEMBL_GENE_ID_RE.match(str(gene).strip()) for gene in row.get("genes", [])):
            unresolved_rows += 1

        if "evidence_genes" in row:
            row["evidence_genes"] = repair_gene_list(list(row.get("evidence_genes", [])), mapping)

    write_jsonl(path, rows)
    return changed_rows, unresolved_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair ENSG gene identifiers in exported JSONL files")
    parser.add_argument("paths", nargs="+", help="JSONL file paths or dataset export directories")
    return parser.parse_args()


def iter_jsonl_paths(raw_paths: list[str]) -> list[Path]:
    resolved = []
    for raw in raw_paths:
        path = Path(raw)
        if path.is_dir():
            resolved.extend(sorted(path.glob("*.jsonl")))
        else:
            resolved.append(path)
    return resolved


def main() -> None:
    args = parse_args()
    for path in iter_jsonl_paths(args.paths):
        changed_rows, unresolved_rows = repair_jsonl(path)
        print({"path": str(path), "changed_rows": changed_rows, "rows_still_with_ensg": unresolved_rows})


if __name__ == "__main__":
    main()