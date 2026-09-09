from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .builder import (
    _build_identity_alias_vocabulary,
    _collect_evidence_genes_missing_from_input,
    _collect_allowed_grounding_genes,
    _collect_allowed_grounding_identity_aliases,
    _extract_gene_mentions,
    _extract_identity_mentions,
    read_jsonl,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
DEFAULT_BENCHMARK_ROOT = PIPELINE_DIR / "data" / "gene_only_benchmarks" / "gpt4o_mini_benchmark_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit an existing gene-only benchmark for answer gene and identity mentions that are not present in hidden grounding."
    )
    parser.add_argument(
        "--hidden-path",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT / "benchmark_hidden.jsonl",
        help="Path to benchmark_hidden.jsonl.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional JSON path for the audit summary. Defaults next to the hidden benchmark.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help="Maximum number of violating rows to include in the printed sample report.",
    )
    return parser


def _audit_row(row: dict[str, Any], *, known_identity_aliases: set[str]) -> dict[str, Any] | None:
    hidden_grounding = row.get("hidden_grounding") or {}
    visible_input = row.get("model_input") or {}
    allowed_genes = _collect_allowed_grounding_genes(hidden_grounding)
    allowed_identity_aliases = _collect_allowed_grounding_identity_aliases(hidden_grounding)
    answer = str(row.get("answer") or "")
    mentioned_genes = _extract_gene_mentions(answer)
    mentioned_identity_aliases = _extract_identity_mentions(answer, candidate_aliases=known_identity_aliases)
    missing_evidence_genes = _collect_evidence_genes_missing_from_input(
        visible_input=visible_input,
        hidden_grounding=hidden_grounding,
    )
    disallowed = sorted(gene for gene in mentioned_genes if gene not in allowed_genes)
    disallowed_identity_aliases = sorted(
        alias for alias in mentioned_identity_aliases if alias not in allowed_identity_aliases
    )
    if not disallowed and not disallowed_identity_aliases and not missing_evidence_genes:
        return None
    return {
        "id": row.get("id"),
        "question_type": row.get("question_type"),
        "level": row.get("level"),
        "disallowed_answer_genes": disallowed,
        "mentioned_answer_genes": sorted(mentioned_genes),
        "allowed_grounding_genes": sorted(allowed_genes),
        "disallowed_answer_identity_labels": disallowed_identity_aliases,
        "mentioned_answer_identity_labels": sorted(mentioned_identity_aliases),
        "allowed_grounding_identity_labels": sorted(allowed_identity_aliases),
        "hidden_evidence_genes_missing_from_input": missing_evidence_genes,
    }


def run_audit(hidden_path: Path, *, max_examples: int) -> dict[str, Any]:
    rows = read_jsonl(hidden_path)
    if not rows:
        raise RuntimeError(f"No rows found in hidden benchmark file: {hidden_path}")

    identity_aliases = _build_identity_alias_vocabulary(
        [
            type("AuditRecord", (), {"hidden_grounding": lambda self, row=row: row.get("hidden_grounding") or {}})()
            for row in rows
        ]
    )
    violations: list[dict[str, Any]] = []
    type_counter: Counter[str] = Counter()
    gene_counter: Counter[str] = Counter()
    identity_counter: Counter[str] = Counter()
    missing_evidence_counter: Counter[str] = Counter()

    for row in rows:
        violation = _audit_row(row, known_identity_aliases=identity_aliases)
        if violation is None:
            continue
        violations.append(violation)
        type_counter[str(violation["question_type"])] += 1
        gene_counter.update(violation["disallowed_answer_genes"])
        identity_counter.update(violation["disallowed_answer_identity_labels"])
        for genes in violation["hidden_evidence_genes_missing_from_input"].values():
            missing_evidence_counter.update(genes)

    summary = {
        "hidden_path": str(hidden_path),
        "total_rows": len(rows),
        "violating_rows": len(violations),
        "passing_rows": len(rows) - len(violations),
        "question_type_violation_counts": dict(type_counter),
        "top_disallowed_genes": gene_counter.most_common(25),
        "top_disallowed_identity_labels": identity_counter.most_common(25),
        "top_hidden_evidence_genes_missing_from_input": missing_evidence_counter.most_common(25),
        "sample_violations": violations[:max_examples],
    }
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = run_audit(args.hidden_path, max_examples=args.max_examples)
    summary_output = args.summary_output or args.hidden_path.with_suffix(".grounding_audit.json")
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved audit summary to {summary_output}")


if __name__ == "__main__":
    main()