#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

from langchain_openai import ChatOpenAI


SCRIPT_DIR = Path(__file__).resolve().parent
CONTRASTIVE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = CONTRASTIVE_ROOT.parent.parent.parent.parent

sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != SCRIPT_DIR]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CONTRASTIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTRASTIVE_ROOT))

from dataset_pipeline.marker_catalog import candidate_labels_for_gold  # noqa: E402
from dataset_pipeline.marker_catalog import canonicalize_label  # noqa: E402
from dataset_pipeline.marker_catalog import get_marker_profile  # noqa: E402
from stage3_ace_orchestrator.config import load_repo_dotenv  # noqa: E402


load_repo_dotenv(Path(__file__))

PROMPT_VERSION = "grounded-qna-v3"
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "stage3_gpt_qna" / "gpt5_mini_grounded_qna.jsonl"
STYLE_OPTIONS = (
    "plain annotation request",
    "uncertain user asking for help",
    "comparison-focused request",
    "confidence-focused request",
    "brief natural request",
)
TASK_FAMILY_ORDER = (
    "single_list_user_questions",
    "multi_list_user_questions",
)
SINGLE_LIST_MODES = (
    "identity_call",
    "plausible_alternatives",
    "confidence_check",
    "tissue_context",
    "contamination_or_quality",
)
MULTI_LIST_MODES = (
    "annotate_each",
    "compare_and_group",
    "odd_one_out",
    "same_or_different",
    "shared_lineage_or_state",
)
PROFILE_IDS = ("A", "B", "C", "D")
GENE_TOKEN_EXCEPTIONS = {
    "RNA",
    "DNA",
    "JSON",
    "CELL",
    "TYPE",
    "TOP",
    "MODEL",
    "GPT",
}


@dataclass(frozen=True)
class GroundedSourceRecord:
    source_dataset: str
    split: str
    source_sample_id: str
    cell_type: str
    tissue: str | None
    broad_cell_class: str | None
    genes: tuple[str, ...]
    evidence_genes: tuple[str, ...]
    available_negative_markers: tuple[str, ...]
    rank_key: int


@dataclass
class DatasetFacts:
    tissue: str | None
    cell_type_counts: Counter[str]
    broad_class_counts: Counter[str]

    @property
    def cell_type_set(self) -> set[str]:
        return set(self.cell_type_counts)

    @property
    def broad_class_set(self) -> set[str]:
        return set(self.broad_class_counts)


@dataclass(frozen=True)
class MultiListGroundedRecord:
    profiles: tuple[GroundedSourceRecord, ...]
    profile_ids: tuple[str, ...]
    bundle_id: str
    rank_key: int


@dataclass(frozen=True)
class GenerationTarget:
    task_family: str
    variant_index: int
    source: GroundedSourceRecord | None = None
    bundle: MultiListGroundedRecord | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build grounded free-form QnA rows from grouped exports with GPT-5 mini")
    parser.add_argument(
        "--exports-root",
        default=str(SCRIPT_DIR / "data" / "exports"),
        help="Root directory containing grouped stage-2 exports",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional JSON summary path. Defaults next to --output.",
    )
    parser.add_argument("--k", type=int, required=True, help="Number of QnA rows to generate")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Grouped export splits to sample from",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI chat model")
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature")
    parser.add_argument("--seed", type=int, default=17, help="Sampling and prompt seed")
    parser.add_argument(
        "--max-sources-per-group",
        type=int,
        default=3,
        help="Maximum source rows retained per dataset/cell-type/split group before generation",
    )
    parser.add_argument(
        "--top-genes",
        type=int,
        default=200,
        help="Number of ranked genes stored per output row and exposed to the generator",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per row when JSON parsing or grounding verification fails",
    )
    return parser.parse_args()


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _stable_hash_value(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _parse_csv_markers(text: str | None) -> list[str]:
    if not text:
        return []
    stripped = text.strip()
    if not stripped or stripped.lower() == "none":
        return []
    markers = []
    for part in stripped.split(","):
        marker = part.strip().upper()
        if marker and marker not in markers:
            markers.append(marker)
    return markers


def _parse_answer_fields(answer: str | None) -> dict[str, str]:
    if not answer:
        return {}
    parsed: dict[str, str] = {}
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip().upper()] = value.strip()
    return parsed


def _normalize_gene(gene: str) -> str:
    return gene.strip().upper()


def _stable_unique(values: list[str]) -> list[str]:
    ordered: list[str] = []
    for value in values:
        if value and value not in ordered:
            ordered.append(value)
    return ordered


def _stable_order(values: list[str], seed_text: str) -> list[str]:
    return sorted(values, key=lambda value: (_stable_hash_value(f"{seed_text}:{value}"), value))


def _dataset_display_tissue(dataset_name: str, facts: DatasetFacts | None) -> str:
    if facts and facts.tissue:
        return facts.tissue
    stripped = dataset_name.removesuffix("_cell_annotation")
    parts = [part for part in stripped.split("_") if part and part.lower() not in {"tabula", "sapiens"}]
    if not parts:
        return dataset_name
    return " ".join(part.capitalize() for part in parts)


def _extract_grounding(
    row: dict[str, Any],
    *,
    top_genes: int,
    seed: int,
    source_dataset: str,
    split: str,
) -> GroundedSourceRecord | None:
    metadata = row.get("metadata") or {}
    cell_type = str(metadata.get("cell_type") or "").strip()
    if not cell_type:
        return None

    genes = [str(gene).strip().upper() for gene in (row.get("genes") or []) if str(gene).strip()]
    if not genes:
        return None

    visible_genes = tuple(genes[:top_genes])
    evidence_candidates = [
        _normalize_gene(gene)
        for gene in (row.get("evidence_genes") or metadata.get("evidence_genes") or [])
        if str(gene).strip()
    ]
    negative_candidates = [
        _normalize_gene(gene)
        for gene in (metadata.get("negative_markers") or [])
        if str(gene).strip()
    ]

    if not evidence_candidates or not negative_candidates:
        return None

    observed_genes = set(visible_genes)
    observed_evidence = [marker for marker in evidence_candidates if marker in observed_genes]
    available_negative = [marker for marker in negative_candidates if marker not in observed_genes]

    if len(observed_evidence) < 1:
        return None

    source_sample_id = str(row.get("sample_id") or row.get("id") or "").strip()
    if not source_sample_id:
        return None

    tissue = metadata.get("tissue") or metadata.get("tissue_general")
    broad_cell_class = str(metadata.get("broad_cell_class") or "").strip() or None
    rank_key = _stable_hash_value(f"{seed}:{source_dataset}:{split}:{cell_type}:{source_sample_id}")
    return GroundedSourceRecord(
        source_dataset=source_dataset,
        split=split,
        source_sample_id=source_sample_id,
        cell_type=cell_type,
        tissue=str(tissue) if tissue else None,
        broad_cell_class=broad_cell_class,
        genes=visible_genes,
        evidence_genes=tuple(observed_evidence[:6]),
        available_negative_markers=tuple(available_negative[:4]),
        rank_key=rank_key,
    )


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                yield json.loads(line)


def _keep_top_records(pool: dict[tuple[str, str, str], list[GroundedSourceRecord]], record: GroundedSourceRecord, max_size: int) -> None:
    key = (record.split, record.source_dataset, record.cell_type)
    bucket = pool.setdefault(key, [])
    bucket.append(record)
    bucket.sort(key=lambda item: item.rank_key)
    if len(bucket) > max_size:
        del bucket[max_size:]


def _collect_source_pool(
    exports_root: Path,
    splits: list[str],
    *,
    top_genes: int,
    seed: int,
    max_sources_per_group: int,
) -> tuple[list[GroundedSourceRecord], dict[str, DatasetFacts]]:
    pool: dict[tuple[str, str, str], list[GroundedSourceRecord]] = {}
    dataset_facts: dict[str, DatasetFacts] = {}
    for dataset_dir in sorted(path for path in exports_root.iterdir() if path.is_dir()):
        facts = dataset_facts.setdefault(
            dataset_dir.name,
            DatasetFacts(tissue=None, cell_type_counts=Counter(), broad_class_counts=Counter()),
        )
        for split in splits:
            export_path = dataset_dir / f"cell_annotation_rationale_grouped_{split}.jsonl"
            if not export_path.exists():
                continue
            for row in _iter_jsonl(export_path):
                metadata = row.get("metadata") or {}
                tissue = metadata.get("tissue") or metadata.get("tissue_general")
                if tissue and not facts.tissue:
                    facts.tissue = str(tissue)
                cell_type = str(metadata.get("cell_type") or "").strip()
                if cell_type:
                    facts.cell_type_counts[cell_type] += 1
                broad_cell_class = str(metadata.get("broad_cell_class") or "").strip()
                if broad_cell_class:
                    facts.broad_class_counts[broad_cell_class] += 1

                record = _extract_grounding(
                    row,
                    top_genes=top_genes,
                    seed=seed,
                    source_dataset=dataset_dir.name,
                    split=split,
                )
                if record is None:
                    continue
                _keep_top_records(pool, record, max_sources_per_group)

    ordered_keys = sorted(pool, key=lambda item: (item[0], item[2], item[1]))
    flattened: list[GroundedSourceRecord] = []
    while ordered_keys:
        next_keys = []
        for key in ordered_keys:
            bucket = pool[key]
            if not bucket:
                continue
            flattened.append(bucket.pop(0))
            if bucket:
                next_keys.append(key)
        ordered_keys = next_keys
    return flattened, dataset_facts


def _style_for(source: GroundedSourceRecord, variant_index: int, seed: int) -> str:
    style_index = _stable_hash_value(f"{seed}:{source.source_sample_id}:{variant_index}") % len(STYLE_OPTIONS)
    return STYLE_OPTIONS[style_index]


def _sanitize_question_text(question: str, cell_type: str) -> str:
    return re.sub(r"\s+", " ", str(question).strip())


def _question_with_gene_list(question: str, genes: tuple[str, ...], cell_type: str) -> str:
    question_text = _sanitize_question_text(question, cell_type)
    gene_list = ", ".join(genes)
    return f"Here is the ranked gene list: {gene_list}. {question_text}"


def _question_with_profile_gene_lists(question: str, bundle: MultiListGroundedRecord) -> str:
    question_text = _sanitize_question_text(question, "")
    prefixes = [
        f"Here is the ranked gene list for cell {profile_id}: {', '.join(source.genes)}."
        for profile_id, source in zip(bundle.profile_ids, bundle.profiles)
    ]
    return " ".join(prefixes + [question_text])


def _question_mentions_marker(question: str, markers: set[str]) -> bool:
    for marker in markers:
        if re.search(rf"(?<![A-Z0-9]){re.escape(marker)}(?![A-Z0-9])", question, flags=re.IGNORECASE):
            return True
    return False


def _normalize_free_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _source_label_metadata(source: GroundedSourceRecord) -> dict[str, str]:
    metadata = {"cell_type": source.cell_type}
    if source.broad_cell_class:
        metadata["broad_cell_class"] = source.broad_cell_class
    if source.tissue:
        metadata["tissue"] = source.tissue
    return metadata


def _label_variants_for_source(source: GroundedSourceRecord) -> list[str]:
    metadata = _source_label_metadata(source)
    variants = [source.cell_type]
    canonical_label = canonicalize_label(source.cell_type, metadata)
    if canonical_label:
        variants.append(canonical_label)
    profile = get_marker_profile(source.cell_type, metadata)
    if profile is not None:
        variants.extend(profile.aliases)
        variants.append(profile.canonical_label)
    if source.cell_type.endswith(" epithelial cell"):
        variants.append(source.cell_type.removesuffix(" cell"))
        variants.append(source.cell_type.replace("epithelial cell", "epithelium"))
    return _stable_unique([str(value).strip() for value in variants if str(value).strip()])


def _answer_label_text(answer: str) -> str:
    answer_fields = _parse_answer_fields(answer)
    for key in ("LABEL", "FINAL_LABEL", "CELL_TYPE", "FINAL_ANSWER"):
        value = answer_fields.get(key)
        if value:
            return value
    return answer


def _answer_mentions_expected_label(answer: str, source: GroundedSourceRecord) -> bool:
    normalized_answer = f" {_normalize_free_text(_answer_label_text(answer))} "
    for variant in _label_variants_for_source(source):
        normalized_variant = _normalize_free_text(variant)
        if normalized_variant and f" {normalized_variant} " in normalized_answer:
            return True
    return False


def _label_matches_source(label: str, source: GroundedSourceRecord) -> bool:
    normalized_label = _normalize_free_text(label)
    if not normalized_label:
        return False
    return any(_normalize_free_text(variant) == normalized_label for variant in _label_variants_for_source(source))


def _text_mentions_any_label(text: str, label_variants: list[str]) -> bool:
    normalized_text = f" {_normalize_free_text(text)} "
    for variant in label_variants:
        normalized_variant = _normalize_free_text(variant)
        if normalized_variant and f" {normalized_variant} " in normalized_text:
            return True
    return False


def _stable_source_order(records: list[GroundedSourceRecord], seed_text: str) -> list[GroundedSourceRecord]:
    return sorted(
        records,
        key=lambda record: (
            _stable_hash_value(f"{seed_text}:{record.source_dataset}:{record.source_sample_id}"),
            record.rank_key,
            record.source_dataset,
            record.source_sample_id,
        ),
    )


def _select_diverse_sources(source_pool: list[GroundedSourceRecord], *, k: int, seed: int) -> list[GroundedSourceRecord]:
    limit = min(k, len(source_pool))
    if limit <= 0:
        return []

    buckets: dict[str, list[GroundedSourceRecord]] = {}
    for source in source_pool:
        buckets.setdefault(source.cell_type, []).append(source)

    for cell_type, records in list(buckets.items()):
        buckets[cell_type] = _stable_source_order(records, f"diverse_bucket:{seed}:{cell_type}")

    cell_type_order = _stable_order(list(buckets), f"diverse_cell_types:{seed}")
    selected: list[GroundedSourceRecord] = []
    while cell_type_order and len(selected) < limit:
        next_order: list[str] = []
        for cell_type in cell_type_order:
            bucket = buckets[cell_type]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            if bucket:
                next_order.append(cell_type)
            if len(selected) >= limit:
                break
        cell_type_order = next_order
    return selected


def _bundle_profile_count(selection_seed: str) -> int:
    digest = hashlib.sha1(f"bundle|{selection_seed}".encode("utf-8")).hexdigest()
    return 2 + (int(digest[:2], 16) % 3)


def _single_list_mode_for(source: GroundedSourceRecord, variant_index: int, seed: int) -> str:
    mode_index = _stable_hash_value(f"single_mode:{seed}:{source.source_sample_id}:{variant_index}") % len(SINGLE_LIST_MODES)
    return SINGLE_LIST_MODES[mode_index]


def _multi_list_mode_for(bundle: MultiListGroundedRecord, variant_index: int, seed: int) -> str:
    mode_index = _stable_hash_value(f"multi_mode:{seed}:{bundle.bundle_id}:{variant_index}") % len(MULTI_LIST_MODES)
    return MULTI_LIST_MODES[mode_index]


def _build_multi_list_bundle(
    anchor: GroundedSourceRecord,
    source_pool: list[GroundedSourceRecord],
    *,
    variant_index: int,
    seed: int,
) -> MultiListGroundedRecord | None:
    desired_count = _bundle_profile_count(f"{seed}:{anchor.source_sample_id}:{variant_index}")
    candidates = [
        source
        for source in source_pool
        if source.source_sample_id != anchor.source_sample_id and source.split == anchor.split
    ]
    ordered = sorted(
        candidates,
        key=lambda source: (
            source.cell_type == anchor.cell_type,
            source.tissue == anchor.tissue,
            _stable_hash_value(f"bundle_candidate:{seed}:{anchor.source_sample_id}:{variant_index}:{source.source_sample_id}"),
            source.rank_key,
        ),
    )
    selected = [anchor]
    used_sample_ids = {anchor.source_sample_id}
    used_cell_types = {anchor.cell_type}

    for candidate in ordered:
        if candidate.source_sample_id in used_sample_ids or candidate.cell_type in used_cell_types:
            continue
        selected.append(candidate)
        used_sample_ids.add(candidate.source_sample_id)
        used_cell_types.add(candidate.cell_type)
        if len(selected) >= desired_count:
            break

    if len(selected) < desired_count:
        for candidate in ordered:
            if candidate.source_sample_id in used_sample_ids:
                continue
            selected.append(candidate)
            used_sample_ids.add(candidate.source_sample_id)
            if len(selected) >= desired_count:
                break

    if len(selected) < 2:
        return None

    profile_ids = PROFILE_IDS[: len(selected)]
    bundle_id = "|".join(source.source_sample_id for source in selected)
    rank_key = _stable_hash_value(f"bundle_rank:{seed}:{bundle_id}:{variant_index}")
    return MultiListGroundedRecord(
        profiles=tuple(selected),
        profile_ids=tuple(profile_ids),
        bundle_id=bundle_id,
        rank_key=rank_key,
    )


def _build_single_task_spec(source: GroundedSourceRecord, question_mode: str) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "task_family": "single_list_user_questions",
        "task_mode": question_mode,
        "question_requirements": [],
        "answer_requirements": [
            "State the most likely cell identity explicitly.",
            "Ground any marker claims only in supported_evidence_genes and available_negative_markers.",
        ],
        "task_context": {},
        "expected_final_label": source.cell_type,
        "expected_abstain": False,
        "requires_tissue": False,
    }
    if question_mode == "identity_call":
        spec["task_goal"] = "Ask for the most likely identity of one cell from a ranked gene list."
        return spec
    if question_mode == "plausible_alternatives":
        spec["task_goal"] = "Ask for the most likely identity and one or two plausible alternatives for one ranked gene list."
        return spec
    if question_mode == "confidence_check":
        spec["task_goal"] = "Ask what the most likely identity is and how confident someone should be from the ranked gene list alone."
        return spec
    if question_mode == "tissue_context":
        spec["task_goal"] = "Ask what tissue or context the gene list is most consistent with, together with the likely cell identity."
        spec["requires_tissue"] = bool(source.tissue)
        if source.tissue:
            spec["answer_requirements"].append("Name the likely tissue or tissue context explicitly.")
            spec["task_context"] = {"expected_tissue": source.tissue}
        return spec
    if question_mode == "contamination_or_quality":
        spec["task_goal"] = "Ask whether the gene list looks mixed, contaminated, or low-quality, while still asking for the most likely identity."
        spec["answer_requirements"].append("Comment briefly on ambiguity, possible contamination, or quality if relevant.")
        return spec
    raise ValueError(f"Unsupported single-list question mode: {question_mode}")


def _build_multi_task_spec(bundle: MultiListGroundedRecord, question_mode: str) -> dict[str, Any]:
    profile_context = [
        {
            "profile_id": profile_id,
            "tissue": source.tissue,
        }
        for profile_id, source in zip(bundle.profile_ids, bundle.profiles)
    ]
    spec: dict[str, Any] = {
        "task_family": "multi_list_user_questions",
        "task_mode": question_mode,
        "question_requirements": [
            "Refer to the ranked lists only as cells identified by their profile IDs.",
            "Do not reveal the gold labels in the question body.",
        ],
        "answer_requirements": [
            "Annotate each listed cell separately in the answer.",
            "Prefix each per-cell annotation with its profile ID, for example Cell A:.",
            "Use the exact metadata cell_type string for each cell when you state the final label.",
            "Ground marker claims for each cell only in that cell's supported_evidence_genes and available_negative_markers.",
            "Finish with a brief comparison summary across the cells.",
        ],
        "task_context": {"profile_ids": list(bundle.profile_ids), "profiles": profile_context},
        "expected_abstain": False,
        "expected_final_labels": {
            profile_id: source.cell_type for profile_id, source in zip(bundle.profile_ids, bundle.profiles)
        },
    }
    if question_mode == "annotate_each":
        spec["task_goal"] = "Ask for a separate annotation for each ranked gene list."
        return spec
    if question_mode == "compare_and_group":
        spec["task_goal"] = "Ask to annotate each ranked gene list and say which cells look most similar or different."
        return spec
    if question_mode == "odd_one_out":
        spec["task_goal"] = "Ask to annotate each ranked gene list and identify whether one cell looks like the odd one out."
        return spec
    if question_mode == "same_or_different":
        spec["task_goal"] = "Ask whether the ranked gene lists look like the same cell type or different cell types, while still asking for per-cell annotations."
        return spec
    if question_mode == "shared_lineage_or_state":
        spec["task_goal"] = "Ask whether the ranked gene lists look like related states of one lineage or clearly distinct cell identities, while still asking for per-cell annotations."
        return spec
    raise ValueError(f"Unsupported multi-list question mode: {question_mode}")


def _build_alternative_labels(source: GroundedSourceRecord, dataset_facts: dict[str, DatasetFacts], limit: int = 3) -> list[str]:
    metadata = {
        "cell_type": source.cell_type,
        "broad_cell_class": source.broad_cell_class,
        "tissue": source.tissue,
    }
    candidates = [
        str(label).strip()
        for label in candidate_labels_for_gold(source.cell_type, metadata)
        if str(label).strip() and str(label).strip() != source.cell_type
    ]
    source_facts = dataset_facts.get(source.source_dataset)
    if source_facts:
        candidates.extend(
            label
            for label, _count in source_facts.cell_type_counts.most_common()
            if label and label != source.cell_type
        )
    ordered = _stable_order(_stable_unique(candidates), f"alternative_labels:{source.source_sample_id}")
    return ordered[:limit]


def _build_differential_panel(source: GroundedSourceRecord, dataset_facts: dict[str, DatasetFacts], panel_size: int = 4) -> list[str]:
    alternatives = _build_alternative_labels(source, dataset_facts, limit=max(3, panel_size))
    panel = [source.cell_type] + alternatives[: max(0, panel_size - 1)]
    return _stable_order(_stable_unique(panel), f"differential_panel:{source.source_sample_id}")


def _dataset_overlap_score(source_facts: DatasetFacts, candidate_facts: DatasetFacts) -> tuple[int, int]:
    shared_labels = len(source_facts.cell_type_set & candidate_facts.cell_type_set)
    shared_broad = len(source_facts.broad_class_set & candidate_facts.broad_class_set)
    return shared_labels, shared_broad


def _select_cross_experts(
    source: GroundedSourceRecord,
    dataset_facts: dict[str, DatasetFacts],
    max_experts: int = 4,
) -> tuple[list[str], list[str]]:
    source_facts = dataset_facts.get(source.source_dataset)
    if source_facts is None:
        return [source.source_dataset], [source.source_dataset]

    label_matches: list[str] = []
    broad_matches: list[str] = []
    fallback: list[str] = []
    for dataset_name, facts in dataset_facts.items():
        if dataset_name == source.source_dataset:
            continue
        if source.cell_type in facts.cell_type_set:
            label_matches.append(dataset_name)
            continue
        if source.broad_cell_class and source.broad_cell_class in facts.broad_class_set:
            broad_matches.append(dataset_name)
            continue
        fallback.append(dataset_name)

    fallback = [
        dataset_name
        for dataset_name, _shared_labels, _shared_broad in sorted(
            (
                (candidate, *_dataset_overlap_score(source_facts, dataset_facts[candidate]))
                for candidate in fallback
            ),
            key=lambda item: (-item[1], -item[2], item[0]),
        )
    ]
    ordered_related = _stable_order(
        _stable_unique(label_matches + broad_matches + fallback),
        f"cross_expert:{source.source_sample_id}",
    )
    candidate_experts = [source.source_dataset] + ordered_related[: max_experts - 1]
    oracle_experts: list[str] = []
    for dataset_name in candidate_experts:
        if dataset_name == source.source_dataset:
            oracle_experts.append(dataset_name)
            continue
        facts = dataset_facts[dataset_name]
        if source.cell_type in facts.cell_type_set:
            oracle_experts.append(dataset_name)
            continue
        if source.broad_cell_class and source.broad_cell_class in facts.broad_class_set:
            oracle_experts.append(dataset_name)
    return candidate_experts, _stable_unique(oracle_experts)


def _select_ood_experts(
    source: GroundedSourceRecord,
    dataset_facts: dict[str, DatasetFacts],
    count: int = 3,
) -> list[str]:
    if not source.cell_type and not source.broad_cell_class:
        return []

    candidates = []
    for dataset_name, facts in dataset_facts.items():
        if dataset_name == source.source_dataset:
            continue
        if source.cell_type in facts.cell_type_set:
            continue
        if source.broad_cell_class and source.broad_cell_class in facts.broad_class_set:
            continue
        candidates.append(dataset_name)
    return _stable_order(candidates, f"ood:{source.source_sample_id}")[:count]


def _build_single_prompt(
    source: GroundedSourceRecord,
    *,
    question_style: str,
    variant_index: int,
    task_spec: dict[str, Any],
) -> tuple[str, str]:
    system_prompt = (
        "You generate realistic single-cell analysis benchmark QnA examples. "
        "Return exactly one JSON object and no surrounding text."
    )
    payload = {
        "task": "Create one free-form user question and one free-form assistant answer for a single ranked gene list.",
        "constraints": [
            "The user question body must sound like a normal scientist or analyst asking for help after pasting one ranked gene list.",
            "Do not reveal the gold label or tissue in the question body.",
            "Do not mention specific marker genes in the question body.",
            "Do not restate the gene list inside the question body; the generator will prepend it.",
            "Use the exact metadata cell_type string when naming the final label in the answer and the structured label field.",
            "The answer must follow answer_requirements and stay grounded in the provided markers and ranked genes.",
            "The answer may only name genes from supported_evidence_genes or available_negative_markers.",
            "Do not mention dataset names.",
            "Prefer 2 to 4 marker mentions total across the answer.",
            "If you use negative markers, frame them as missing support for an alternative interpretation.",
            "Keep the answer concise but complete.",
        ],
        "task_family": task_spec["task_family"],
        "task_mode": task_spec["task_mode"],
        "task_goal": task_spec["task_goal"],
        "question_requirements": task_spec.get("question_requirements", []),
        "answer_requirements": task_spec.get("answer_requirements", []),
        "question_style": question_style,
        "variant_index": variant_index,
        "task_context": task_spec.get("task_context", {}),
        "grounding": {
            "cell_type": source.cell_type,
            "tissue": source.tissue,
            "supported_evidence_genes": list(source.evidence_genes),
            "available_negative_markers": list(source.available_negative_markers),
            "ranked_genes": list(source.genes),
        },
        "required_json_schema": {
            "question": "string",
            "answer": "string",
            "label": "string",
            "evidence_genes_used": ["marker1"],
            "negative_markers_used": ["marker2"],
            "style": "string",
        },
    }
    return system_prompt, _json_dumps(payload)


def _build_multi_prompt(
    bundle: MultiListGroundedRecord,
    *,
    question_style: str,
    variant_index: int,
    task_spec: dict[str, Any],
) -> tuple[str, str]:
    system_prompt = (
        "You generate realistic single-cell analysis benchmark QnA examples. "
        "Return exactly one JSON object and no surrounding text."
    )
    grounding_profiles = [
        {
            "profile_id": profile_id,
            "cell_type": source.cell_type,
            "tissue": source.tissue,
            "supported_evidence_genes": list(source.evidence_genes),
            "available_negative_markers": list(source.available_negative_markers),
            "ranked_genes": list(source.genes),
        }
        for profile_id, source in zip(bundle.profile_ids, bundle.profiles)
    ]
    payload = {
        "task": "Create one free-form user question and one free-form assistant answer for multiple ranked gene lists.",
        "constraints": [
            "The user question body must sound like a normal scientist or analyst asking for help after pasting multiple ranked gene lists.",
            "Refer to the lists only as cells identified by their profile IDs.",
            "Do not reveal any gold labels or tissues in the question body.",
            "Do not mention specific marker genes in the question body.",
            "Do not restate the ranked gene lists inside the question body; the generator will prepend them.",
            "For each profile, use the exact metadata cell_type string in both the prose answer and the structured label field, not an approximation or '-like' label.",
            "The answer must annotate each cell separately and follow answer_requirements.",
            "The answer may only name genes from the supported_evidence_genes or available_negative_markers for the matching profile.",
            "Do not mention dataset names.",
            "Keep the answer concise but complete.",
        ],
        "task_family": task_spec["task_family"],
        "task_mode": task_spec["task_mode"],
        "task_goal": task_spec["task_goal"],
        "question_requirements": task_spec.get("question_requirements", []),
        "answer_requirements": task_spec.get("answer_requirements", []),
        "question_style": question_style,
        "variant_index": variant_index,
        "task_context": task_spec.get("task_context", {}),
        "grounding": {"profiles": grounding_profiles},
        "required_json_schema": {
            "question": "string",
            "answer": "string",
            "profile_annotations": [
                {
                    "profile_id": "A",
                    "label": "string",
                    "evidence_genes_used": ["marker1"],
                    "negative_markers_used": ["marker2"],
                }
            ],
            "comparison_summary": "string",
            "style": "string",
        },
    }
    return system_prompt, _json_dumps(payload)


def _extract_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if not candidate.startswith("{"):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Model did not return JSON")
        candidate = candidate[start : end + 1]
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Model output is not a JSON object")
    return parsed


def _coerce_marker_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    markers = []
    for value in values:
        marker = str(value).strip().upper()
        if marker and marker not in markers:
            markers.append(marker)
    return markers


def _coerce_profile_annotations(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []
    annotations: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        annotations.append(
            {
                "profile_id": str(value.get("profile_id") or "").strip(),
                "label": str(value.get("label") or "").strip(),
                "evidence_genes_used": _coerce_marker_list(value.get("evidence_genes_used")),
                "negative_markers_used": _coerce_marker_list(value.get("negative_markers_used")),
            }
        )
    return annotations


def _extract_gene_mentions(text: str) -> set[str]:
    def flush_token(raw_token: list[str], mentions: set[str]) -> None:
        candidate_raw = "".join(raw_token).strip("/")
        if len(candidate_raw) < 3:
            return
        if "/" in candidate_raw:
            return
        uppercase_count = sum(1 for ch in candidate_raw if ch.isupper())
        has_digit = any(ch.isdigit() for ch in candidate_raw)
        if uppercase_count < 2 and not has_digit:
            return
        candidate = candidate_raw.upper()
        if candidate not in GENE_TOKEN_EXCEPTIONS:
            mentions.add(candidate)

    mentions = set()
    token = []
    for char in text:
        if char.isalnum() or char in {"-", "/"}:
            token.append(char)
            continue
        if token:
            flush_token(token, mentions)
            token = []
    if token:
        flush_token(token, mentions)
    return mentions


def _verify_single_generation(
    source: GroundedSourceRecord,
    payload: dict[str, Any],
    *,
    task_spec: dict[str, Any],
) -> tuple[bool, list[str], dict[str, Any]]:
    issues: list[str] = []
    question_body = str(payload.get("question_body") or payload.get("question") or "").strip()
    question = _question_with_gene_list(question_body, source.genes, source.cell_type)
    answer = str(payload.get("answer") or "").strip()
    structured_label = str(payload.get("label") or "").strip()
    evidence_genes_used = _coerce_marker_list(payload.get("evidence_genes_used") or payload.get("positive_markers_used"))
    negative_markers_used = _coerce_marker_list(payload.get("negative_markers_used"))

    allowed_evidence = set(source.evidence_genes)
    allowed_negative = set(source.available_negative_markers)
    disallowed_question_markers = allowed_evidence | allowed_negative
    accepted_label_variants = _label_variants_for_source(source)

    if not question:
        issues.append("missing_question")
    if not answer:
        issues.append("missing_answer")
    if _text_mentions_any_label(question_body, accepted_label_variants):
        issues.append("question_leaks_cell_type")
    if _question_mentions_marker(question_body, disallowed_question_markers):
        issues.append("question_mentions_grounding_marker")
    if structured_label and not _label_matches_source(structured_label, source):
        issues.append("structured_label_mismatch")
    if not _answer_mentions_expected_label(answer, source):
        issues.append("answer_missing_cell_type")
    if task_spec.get("requires_tissue") and source.tissue and source.tissue.lower() not in answer.lower():
        issues.append("answer_missing_tissue")
    if not evidence_genes_used and not negative_markers_used:
        issues.append("no_verified_markers_used")
    if not set(evidence_genes_used).issubset(allowed_evidence):
        issues.append("unsupported_evidence_gene_used")
    if not set(negative_markers_used).issubset(allowed_negative):
        issues.append("unsupported_negative_marker_used")

    for marker in evidence_genes_used + negative_markers_used:
        if marker.lower() not in answer.lower():
            issues.append(f"marker_not_present_in_text:{marker}")

    verification = {
        "verified": not issues,
        "issues": issues,
        "task_family": task_spec["task_family"],
        "task_mode": task_spec["task_mode"],
        "expected_final_label": task_spec["expected_final_label"],
        "accepted_label_variants": accepted_label_variants,
        "label_grounding_source": "source_metadata.cell_type",
        "allowed_evidence_genes": sorted(allowed_evidence),
        "allowed_negative_markers": sorted(allowed_negative),
        "evidence_genes_used": evidence_genes_used,
        "negative_markers_used": negative_markers_used,
    }
    return not issues, issues, verification


def _verify_multi_generation(
    bundle: MultiListGroundedRecord,
    payload: dict[str, Any],
    *,
    task_spec: dict[str, Any],
) -> tuple[bool, list[str], dict[str, Any]]:
    issues: list[str] = []
    question_body = str(payload.get("question_body") or payload.get("question") or "").strip()
    question = _question_with_profile_gene_lists(question_body, bundle)
    answer = str(payload.get("answer") or "").strip()
    annotations = _coerce_profile_annotations(payload.get("profile_annotations"))
    annotation_map: dict[str, dict[str, Any]] = {}
    all_label_variants = _stable_unique(
        [variant for source in bundle.profiles for variant in _label_variants_for_source(source)]
    )
    allowed_question_markers = {
        marker
        for source in bundle.profiles
        for marker in list(source.evidence_genes) + list(source.available_negative_markers)
    }

    if not question:
        issues.append("missing_question")
    if not answer:
        issues.append("missing_answer")
    if _text_mentions_any_label(question_body, all_label_variants):
        issues.append("question_leaks_cell_type")
    if _question_mentions_marker(question_body, allowed_question_markers):
        issues.append("question_mentions_grounding_marker")
    if not annotations:
        issues.append("missing_profile_annotations")

    for annotation in annotations:
        profile_id = annotation["profile_id"]
        if profile_id in annotation_map:
            issues.append(f"duplicate_profile_annotation:{profile_id}")
            continue
        annotation_map[profile_id] = annotation

    per_profile_verification: dict[str, dict[str, Any]] = {}
    for profile_id, source in zip(bundle.profile_ids, bundle.profiles):
        accepted_label_variants = _label_variants_for_source(source)
        annotation = annotation_map.get(profile_id)
        if annotation is None:
            issues.append(f"missing_profile_annotation:{profile_id}")
            per_profile_verification[profile_id] = {
                "expected_final_label": source.cell_type,
                "accepted_label_variants": accepted_label_variants,
                "allowed_evidence_genes": sorted(source.evidence_genes),
                "allowed_negative_markers": sorted(source.available_negative_markers),
                "issues": [f"missing_profile_annotation:{profile_id}"],
            }
            continue

        evidence_genes_used = annotation["evidence_genes_used"]
        negative_markers_used = annotation["negative_markers_used"]
        profile_issues: list[str] = []
        if not _label_matches_source(annotation["label"], source):
            profile_issues.append("structured_label_mismatch")
        if not _answer_mentions_expected_label(answer, source):
            profile_issues.append("answer_missing_cell_type")
        if not re.search(rf"cell\s+{re.escape(profile_id)}\s*:", answer, flags=re.IGNORECASE):
            profile_issues.append("answer_missing_profile_prefix")
        if not set(evidence_genes_used).issubset(set(source.evidence_genes)):
            profile_issues.append("unsupported_evidence_gene_used")
        if not set(negative_markers_used).issubset(set(source.available_negative_markers)):
            profile_issues.append("unsupported_negative_marker_used")
        if not evidence_genes_used and not negative_markers_used:
            profile_issues.append("no_verified_markers_used")
        for marker in evidence_genes_used + negative_markers_used:
            if marker.lower() not in answer.lower():
                profile_issues.append(f"marker_not_present_in_text:{marker}")

        if profile_issues:
            issues.extend(f"{profile_id}:{issue}" for issue in profile_issues)
        per_profile_verification[profile_id] = {
            "expected_final_label": source.cell_type,
            "accepted_label_variants": accepted_label_variants,
            "allowed_evidence_genes": sorted(source.evidence_genes),
            "allowed_negative_markers": sorted(source.available_negative_markers),
            "evidence_genes_used": evidence_genes_used,
            "negative_markers_used": negative_markers_used,
            "issues": profile_issues,
        }

    for profile_id in sorted(set(annotation_map) - set(bundle.profile_ids)):
        issues.append(f"unexpected_profile_annotation:{profile_id}")

    verification = {
        "verified": not issues,
        "issues": issues,
        "task_family": task_spec["task_family"],
        "task_mode": task_spec["task_mode"],
        "profile_ids": list(bundle.profile_ids),
        "profiles": per_profile_verification,
        "label_grounding_source": "source_metadata.cell_type",
    }
    return not issues, issues, verification


def _temperature_request_kwargs(*, model_name: str, temperature: float) -> dict[str, Any]:
    normalized_name = model_name.strip().lower()
    if normalized_name.startswith("gpt-5"):
        return {}
    return {"temperature": temperature}


def _build_llm(model: str, temperature: float, seed: int) -> ChatOpenAI:
    llm_kwargs: dict[str, Any] = {
        "model": model,
        "model_kwargs": {"seed": seed},
    }
    llm_kwargs.update(_temperature_request_kwargs(model_name=model, temperature=temperature))
    return ChatOpenAI(**llm_kwargs)


def _generate_single_one(
    llm: ChatOpenAI,
    source: GroundedSourceRecord,
    *,
    seed: int,
    variant_index: int,
    max_retries: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    question_mode = _single_list_mode_for(source, variant_index, seed)
    task_spec = _build_single_task_spec(source, question_mode)
    for attempt_index in range(max_retries):
        question_style = _style_for(source, variant_index + attempt_index, seed)
        system_prompt, user_prompt = _build_single_prompt(
            source,
            question_style=question_style,
            variant_index=variant_index,
            task_spec=task_spec,
        )
        message = llm.invoke([
            ("system", system_prompt),
            ("human", user_prompt),
        ])
        raw_text = str(message.content)
        attempt_record = {
            "attempt": attempt_index + 1,
            "style": question_style,
            "raw_response": raw_text,
        }
        try:
            payload = _extract_json_object(raw_text)
        except Exception as exc:  # noqa: BLE001
            attempt_record["error"] = f"json_parse_failed:{exc}"
            attempts.append(attempt_record)
            continue

        question_body = str(payload.get("question") or "").strip()
        payload["question_body"] = question_body
        payload["question"] = _question_with_gene_list(question_body, source.genes, source.cell_type)

        ok, issues, verification = _verify_single_generation(source, payload, task_spec=task_spec)
        attempt_record["verification"] = verification
        attempts.append(attempt_record)
        if not ok:
            attempt_record["error"] = ";".join(issues)
            continue

        row_id = hashlib.sha256(
            f"{source.source_sample_id}:{variant_index}:{task_spec['task_family']}:{task_spec['task_mode']}:{question_style}".encode("utf-8")
        ).hexdigest()[:16]
        result = {
            "id": f"gpt_qna_{row_id}",
            "question": str(payload.get("question") or "").strip(),
            "answer": str(payload.get("answer") or "").strip(),
            "genes": list(source.genes),
            "split": source.split,
            "source_dataset": source.source_dataset,
            "source_sample_id": source.source_sample_id,
            "task_family": task_spec["task_family"],
            "ground_truth": {
                "cell_type": source.cell_type,
                "expected_final_label": task_spec["expected_final_label"],
                "expected_abstain": bool(task_spec.get("expected_abstain")),
                "supported_evidence_genes": list(source.evidence_genes),
                "available_negative_markers": list(source.available_negative_markers),
                "tissue": source.tissue,
                "task_mode": task_spec["task_mode"],
                "task_context": task_spec.get("task_context", {}),
            },
            "generator": {
                "model": llm.model_name,
                "prompt_version": PROMPT_VERSION,
                "task_family": task_spec["task_family"],
                "task_mode": task_spec["task_mode"],
                "style": str(payload.get("style") or question_style).strip() or question_style,
                "variant_index": variant_index,
            },
            "verification": verification,
        }
        return result, {"attempts": attempts}
    return None, {"attempts": attempts}


def _generate_multi_one(
    llm: ChatOpenAI,
    bundle: MultiListGroundedRecord,
    *,
    seed: int,
    variant_index: int,
    max_retries: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    question_mode = _multi_list_mode_for(bundle, variant_index, seed)
    task_spec = _build_multi_task_spec(bundle, question_mode)
    anchor = bundle.profiles[0]
    for attempt_index in range(max_retries):
        question_style = _style_for(anchor, variant_index + attempt_index, seed)
        system_prompt, user_prompt = _build_multi_prompt(
            bundle,
            question_style=question_style,
            variant_index=variant_index,
            task_spec=task_spec,
        )
        message = llm.invoke([
            ("system", system_prompt),
            ("human", user_prompt),
        ])
        raw_text = str(message.content)
        attempt_record = {
            "attempt": attempt_index + 1,
            "style": question_style,
            "raw_response": raw_text,
        }
        try:
            payload = _extract_json_object(raw_text)
        except Exception as exc:  # noqa: BLE001
            attempt_record["error"] = f"json_parse_failed:{exc}"
            attempts.append(attempt_record)
            continue

        question_body = str(payload.get("question") or "").strip()
        payload["question_body"] = question_body
        payload["question"] = _question_with_profile_gene_lists(question_body, bundle)

        ok, issues, verification = _verify_multi_generation(bundle, payload, task_spec=task_spec)
        attempt_record["verification"] = verification
        attempts.append(attempt_record)
        if not ok:
            attempt_record["error"] = ";".join(issues)
            continue

        row_id = hashlib.sha256(
            f"{bundle.bundle_id}:{variant_index}:{task_spec['task_family']}:{task_spec['task_mode']}:{question_style}".encode("utf-8")
        ).hexdigest()[:16]
        result = {
            "id": f"gpt_qna_{row_id}",
            "question": str(payload.get("question") or "").strip(),
            "answer": str(payload.get("answer") or "").strip(),
            "split": bundle.profiles[0].split if len({profile.split for profile in bundle.profiles}) == 1 else "mixed",
            "source_dataset": bundle.profiles[0].source_dataset,
            "source_sample_id": bundle.bundle_id,
            "task_family": task_spec["task_family"],
            "profiles": [
                {
                    "profile_id": profile_id,
                    "genes": list(source.genes),
                    "split": source.split,
                    "source_dataset": source.source_dataset,
                    "source_sample_id": source.source_sample_id,
                }
                for profile_id, source in zip(bundle.profile_ids, bundle.profiles)
            ],
            "ground_truth": {
                "profiles": [
                    {
                        "profile_id": profile_id,
                        "cell_type": source.cell_type,
                        "expected_final_label": source.cell_type,
                        "supported_evidence_genes": list(source.evidence_genes),
                        "available_negative_markers": list(source.available_negative_markers),
                        "tissue": source.tissue,
                    }
                    for profile_id, source in zip(bundle.profile_ids, bundle.profiles)
                ],
                "task_mode": task_spec["task_mode"],
                "task_context": task_spec.get("task_context", {}),
            },
            "generator": {
                "model": llm.model_name,
                "prompt_version": PROMPT_VERSION,
                "task_family": task_spec["task_family"],
                "task_mode": task_spec["task_mode"],
                "style": str(payload.get("style") or question_style).strip() or question_style,
                "variant_index": variant_index,
            },
            "verification": verification,
        }
        return result, {"attempts": attempts}
    return None, {"attempts": attempts}


def _select_generation_targets(
    source_pool: list[GroundedSourceRecord],
    *,
    k: int,
    seed: int,
) -> list[GenerationTarget]:
    if not source_pool:
        return []
    selected = _select_diverse_sources(source_pool, k=min(len(source_pool), max(k * 2, 24)), seed=seed)
    targets: list[GenerationTarget] = []
    for index in range(k):
        source = selected[index % len(selected)]
        variant_index = index // len(selected)
        task_family = TASK_FAMILY_ORDER[index % len(TASK_FAMILY_ORDER)]
        if task_family == "multi_list_user_questions":
            bundle = _build_multi_list_bundle(source, source_pool, variant_index=variant_index, seed=seed)
            if bundle is not None:
                targets.append(
                    GenerationTarget(
                        task_family=task_family,
                        variant_index=variant_index,
                        bundle=bundle,
                    )
                )
                continue
        targets.append(
            GenerationTarget(
                task_family="single_list_user_questions",
                variant_index=variant_index,
                source=source,
            )
        )
    return targets


def main() -> None:
    args = parse_args()
    exports_root = Path(args.exports_root).resolve()
    output_path = Path(args.output).resolve()
    summary_path = Path(args.summary_output).resolve() if args.summary_output else output_path.with_suffix(".summary.json")

    if not exports_root.exists():
        raise FileNotFoundError(f"Exports root does not exist: {exports_root}")

    source_pool, dataset_facts = _collect_source_pool(
        exports_root,
        args.splits,
        top_genes=args.top_genes,
        seed=args.seed,
        max_sources_per_group=args.max_sources_per_group,
    )
    if not source_pool:
        raise RuntimeError("No eligible grouped exports with visible evidence genes and metadata negative markers were found")

    targets = _select_generation_targets(source_pool, k=args.k, seed=args.seed)
    llm = _build_llm(args.model, args.temperature, args.seed)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, target in enumerate(targets, start=1):
        if target.task_family == "multi_list_user_questions" and target.bundle is not None:
            row, debug_payload = _generate_multi_one(
                llm,
                target.bundle,
                seed=args.seed,
                variant_index=target.variant_index,
                max_retries=args.max_retries,
            )
            target_id = target.bundle.bundle_id
            target_dataset = target.bundle.profiles[0].source_dataset
            target_split = target.bundle.profiles[0].split if len({profile.split for profile in target.bundle.profiles}) == 1 else "mixed"
            failure_record = {
                "source_sample_id": target.bundle.bundle_id,
                "source_dataset": target_dataset,
                "split": target_split,
                "variant_index": target.variant_index,
                "task_family": target.task_family,
                "profiles": [
                    {
                        "profile_id": profile_id,
                        "source_sample_id": source.source_sample_id,
                        "source_dataset": source.source_dataset,
                    }
                    for profile_id, source in zip(target.bundle.profile_ids, target.bundle.profiles)
                ],
                "debug": debug_payload,
            }
        else:
            if target.source is None:
                raise RuntimeError("Single-list target missing source")
            row, debug_payload = _generate_single_one(
                llm,
                target.source,
                seed=args.seed,
                variant_index=target.variant_index,
                max_retries=args.max_retries,
            )
            target_id = target.source.source_sample_id
            failure_record = {
                "source_sample_id": target.source.source_sample_id,
                "source_dataset": target.source.source_dataset,
                "split": target.source.split,
                "variant_index": target.variant_index,
                "task_family": target.task_family,
                "debug": debug_payload,
            }
        if row is None:
            failures.append(failure_record)
            print(f"[{index}/{len(targets)}] failed {target_id}", flush=True)
            continue
        rows.append(row)
        print(f"[{index}/{len(targets)}] wrote {row['id']} from {target_id}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "output": str(output_path),
        "requested_k": args.k,
        "written_rows": len(rows),
        "failed_rows": len(failures),
        "source_pool": len(source_pool),
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "splits": args.splits,
        "failures": failures,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(_json_dumps(summary), encoding="utf-8")
    print(_json_dumps({key: summary[key] for key in ["output", "written_rows", "failed_rows", "source_pool", "model"]}))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()