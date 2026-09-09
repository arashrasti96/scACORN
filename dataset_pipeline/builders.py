from collections import Counter, defaultdict

from .config import PipelineConfig
from .formatting import ANSWER_SCHEMA_NAME, format_answer
from .marker_catalog import (
    candidate_labels_for_gold,
    canonicalize_label,
    get_marker_profile,
    supported_markers,
    top_missing_markers,
)
from .schemas import Claim, NormalizedCellRecord, TaskExample


TASK_TO_STAGE2 = {
    "cell_annotation_rationale": "cell_type",
    "tissue_identification_rationale": "tissue_inference",
    "cell_state_rationale": "state_classification",
    "differential_diagnosis": "differential_diagnosis",
}


TASK_QUESTIONS = {
    "cell_annotation_rationale": "What is the most likely cell annotation for this expression profile?",
    "tissue_identification_rationale": "What tissue or compartment is most consistent with this expression profile?",
    "cell_state_rationale": "What biological state is most consistent with this expression profile?",
    "differential_diagnosis": "Which candidate label best matches this expression profile?",
}


CELL_ANNOTATION_CONTEXT_KEYS = [
    "tissue",
    "tissue_general",
    "assay",
    "disease",
    "sex",
    "development_stage",
    "suspension_type",
]

TISSUE_IDENTIFICATION_CONTEXT_KEYS = [
    "cell_type",
    "assay",
    "disease",
    "sex",
    "development_stage",
    "suspension_type",
]

CELL_STATE_CONTEXT_KEYS = [
    "cell_type",
    "tissue",
    "tissue_general",
    "assay",
    "disease",
    "sex",
    "development_stage",
    "suspension_type",
]

DIFFERENTIAL_CONTEXT_KEYS = [
    "tissue",
    "tissue_general",
    "assay",
    "disease",
    "sex",
    "development_stage",
    "suspension_type",
]


def _pick_context_claims(record: NormalizedCellRecord, config: PipelineConfig, context_keys: list[str]) -> list[Claim]:
    claims = []
    for key in context_keys:
        if key not in config.defaults.context_allowlist:
            continue
        value = record.metadata.get(key)
        if not value:
            continue
        claims.append(
            Claim(
                value=f"{key}={value}",
                source_type="source_metadata",
                source_key=key,
                verified=True,
            )
        )
    return claims


def _pick_evidence_claims(record: NormalizedCellRecord, count: int) -> list[Claim]:
    claims = []
    for gene in record.genes[:count]:
        claims.append(
            Claim(
                value=gene,
                source_type="observed_gene",
                source_key="genes",
                verified=True,
            )
        )
    return claims


def _pick_marker_grounded_evidence_claims(record: NormalizedCellRecord, label: str, count: int) -> tuple[list[Claim], str | None]:
    canonical_label = canonicalize_label(label, record.metadata)
    supported = supported_markers(record.genes, label, record.metadata)
    claims = []
    for gene, score in supported[:count]:
        claims.append(
            Claim(
                value=gene,
                source_type="marker_catalog",
                source_key="evidence_genes",
                verified=True,
                details="reason=evidence_gene_for_gold",
                claim_kind="evidence",
                source_tier="tier1_curated",
                matched_label=canonical_label or "",
                support_direction="supports_gold",
                score=score,
            )
        )
    return claims, canonical_label


def _negative_markers_for_answer(
    label: str | None,
    metadata: dict,
    canonical_label: str | None = None,
) -> list[str]:
    lookup_label = canonical_label or label
    if not lookup_label:
        return []

    profile = get_marker_profile(lookup_label, {**metadata, "cell_type": canonical_label or label or lookup_label})
    if profile is None:
        return []
    return list(profile.negative_markers)


def _build_marker_grounded_exclusion_claim(
    record: NormalizedCellRecord,
    gold_label: str,
    gold_evidence_claims: list[Claim],
    candidate_label: str,
) -> Claim | None:
    gold_canonical = canonicalize_label(gold_label, record.metadata)
    candidate_canonical = canonicalize_label(candidate_label, {**record.metadata, "cell_type": candidate_label})
    if not gold_canonical or not candidate_canonical or gold_canonical == candidate_canonical:
        return None

    candidate_profile = get_marker_profile(candidate_label, {**record.metadata, "cell_type": candidate_label})
    if candidate_profile is None:
        return None

    candidate_supported = supported_markers(record.genes, candidate_label, {**record.metadata, "cell_type": candidate_label})
    missing_candidate_markers = top_missing_markers(record.genes, candidate_label, {**record.metadata, "cell_type": candidate_label}, count=2)
    if candidate_supported:
        return None

    if not missing_candidate_markers or not gold_evidence_claims:
        return None

    gold_markers = ",".join(claim.value for claim in gold_evidence_claims[:2])
    missing_markers = ",".join(missing_candidate_markers)
    score = round(min(0.99, 0.5 + 0.1 * len(gold_evidence_claims)), 4)
    return Claim(
        value=candidate_label,
        source_type="marker_catalog",
        source_key="candidate_marker_profile",
        verified=True,
        details=(
            "reason=missing_candidate_markers;"
            f"candidate_markers={missing_markers};"
            f"gold_markers={gold_markers}"
        ),
        claim_kind="exclusion",
        source_tier="tier1_curated",
        matched_label=candidate_canonical,
        support_direction="against_candidate",
        score=score,
    )


def _build_cell_annotation(record: NormalizedCellRecord, config: PipelineConfig) -> TaskExample | None:
    label = record.metadata.get("cell_type") or record.metadata.get("cell_type_ontology_term_id")
    if not label:
        return None

    evidence_claims, canonical_label = _pick_marker_grounded_evidence_claims(record, label, config.defaults.evidence_gene_count)
    if len(evidence_claims) < 2:
        return None
    context_claims = _pick_context_claims(record, config, CELL_ANNOTATION_CONTEXT_KEYS)
    negative_markers = _negative_markers_for_answer(label, record.metadata, canonical_label)
    answer_text = format_answer(
        label,
        evidence_claims,
        [],
        context_claims,
        confidence="high",
        negative_markers=negative_markers,
    )

    return TaskExample(
        sample_id=record.sample_id,
        task_type="cell_annotation_rationale",
        source_name=record.source_name,
        source_dataset_id=record.source_dataset_id,
        organism=record.organism,
        genes=record.genes,
        question_text=TASK_QUESTIONS["cell_annotation_rationale"],
        answer_text=answer_text,
        answer_schema_name=ANSWER_SCHEMA_NAME,
        gold_fields={"label": label},
        evidence_claims=evidence_claims,
        context_claims=context_claims,
        metadata=dict(record.metadata),
        provenance={
            "builder": "cell_annotation_rationale",
            "verified_sources_only": True,
            "grounding_mode": "marker_catalog_v1",
            "canonical_label_for_grounding": canonical_label or "",
        },
    )


def _build_tissue_identification(record: NormalizedCellRecord, config: PipelineConfig) -> TaskExample | None:
    label = record.metadata.get("tissue_general") or record.metadata.get("tissue")
    if not label:
        return None

    evidence_claims = _pick_evidence_claims(record, config.defaults.evidence_gene_count)
    context_claims = _pick_context_claims(record, config, TISSUE_IDENTIFICATION_CONTEXT_KEYS)
    answer_text = format_answer(label, evidence_claims, [], context_claims, confidence="medium")

    return TaskExample(
        sample_id=record.sample_id,
        task_type="tissue_identification_rationale",
        source_name=record.source_name,
        source_dataset_id=record.source_dataset_id,
        organism=record.organism,
        genes=record.genes,
        question_text=TASK_QUESTIONS["tissue_identification_rationale"],
        answer_text=answer_text,
        answer_schema_name=ANSWER_SCHEMA_NAME,
        gold_fields={"label": label},
        evidence_claims=evidence_claims,
        context_claims=context_claims,
        metadata=dict(record.metadata),
        provenance={"builder": "tissue_identification_rationale", "verified_sources_only": True},
    )


def _state_label(record: NormalizedCellRecord, config: PipelineConfig) -> tuple[str | None, str | None]:
    for key in config.defaults.state_label_keys:
        value = record.metadata.get(key)
        if value:
            return key, value
    return None, None


def _build_cell_state(record: NormalizedCellRecord, config: PipelineConfig) -> TaskExample | None:
    state_key, label = _state_label(record, config)
    if not label:
        return None

    evidence_claims = _pick_evidence_claims(record, config.defaults.evidence_gene_count)
    context_keys = [key for key in CELL_STATE_CONTEXT_KEYS if key != state_key]
    context_claims = _pick_context_claims(record, config, context_keys)
    answer_text = format_answer(label, evidence_claims, [], context_claims, confidence="medium")

    return TaskExample(
        sample_id=record.sample_id,
        task_type="cell_state_rationale",
        source_name=record.source_name,
        source_dataset_id=record.source_dataset_id,
        organism=record.organism,
        genes=record.genes,
        question_text=TASK_QUESTIONS["cell_state_rationale"],
        answer_text=answer_text,
        answer_schema_name=ANSWER_SCHEMA_NAME,
        gold_fields={"label": label, "source_key": state_key},
        evidence_claims=evidence_claims,
        context_claims=context_claims,
        metadata=dict(record.metadata),
        provenance={"builder": "cell_state_rationale", "verified_sources_only": True},
    )


def _candidate_pool(records: list[NormalizedCellRecord]) -> dict[str, list[str]]:
    by_tissue = defaultdict(list)
    for record in records:
        cell_type = record.metadata.get("cell_type")
        tissue = record.metadata.get("tissue_general") or record.metadata.get("tissue") or "unknown"
        if cell_type:
            by_tissue[tissue].append(cell_type)

    pools = {}
    for tissue, labels in by_tissue.items():
        pools[tissue] = [label for label, _ in Counter(labels).most_common()]
    return pools


def _build_differential(record: NormalizedCellRecord, config: PipelineConfig, pools: dict[str, list[str]]) -> TaskExample | None:
    gold = record.metadata.get("cell_type")
    if not gold:
        return None

    gold_evidence_claims, canonical_label = _pick_marker_grounded_evidence_claims(record, gold, config.defaults.evidence_gene_count)
    if len(gold_evidence_claims) < 2:
        return None

    tissue = record.metadata.get("tissue_general") or record.metadata.get("tissue") or "unknown"
    candidate_priority = []
    for label in candidate_labels_for_gold(gold, record.metadata):
        if label != gold:
            candidate_priority.append(label)
    for label in pools.get(tissue, []):
        if label == gold or label in candidate_priority:
            continue
        candidate_priority.append(label)

    exclusion_claims = []
    candidate_labels = [gold]
    for candidate_label in candidate_priority:
        exclusion_claim = _build_marker_grounded_exclusion_claim(record, gold, gold_evidence_claims, candidate_label)
        if exclusion_claim is None:
            continue
        exclusion_claims.append(exclusion_claim)
        candidate_labels.append(candidate_label)
        if len(candidate_labels) >= max(2, config.defaults.differential_candidate_count):
            break

    if len(candidate_labels) < 2:
        return None
    context_claims = _pick_context_claims(record, config, DIFFERENTIAL_CONTEXT_KEYS)
    negative_markers = _negative_markers_for_answer(gold, record.metadata, canonical_label)
    answer_text = format_answer(
        gold,
        gold_evidence_claims,
        exclusion_claims,
        context_claims,
        confidence="high",
        negative_markers=negative_markers,
    )

    return TaskExample(
        sample_id=record.sample_id,
        task_type="differential_diagnosis",
        source_name=record.source_name,
        source_dataset_id=record.source_dataset_id,
        organism=record.organism,
        genes=record.genes,
        question_text=TASK_QUESTIONS["differential_diagnosis"],
        answer_text=answer_text,
        answer_schema_name=ANSWER_SCHEMA_NAME,
        gold_fields={"label": gold},
        evidence_claims=gold_evidence_claims,
        exclusion_claims=exclusion_claims,
        context_claims=context_claims,
        metadata=dict(record.metadata),
        provenance={
            "builder": "differential_diagnosis",
            "verified_sources_only": True,
            "grounding_mode": "marker_catalog_v1",
            "canonical_label_for_grounding": canonical_label or "",
        },
        candidate_labels=candidate_labels,
    )


def build_examples(records: list[NormalizedCellRecord], config: PipelineConfig, task_types: list[str]) -> list[TaskExample]:
    pools = _candidate_pool(records)
    examples = []
    for record in records:
        if "cell_annotation_rationale" in task_types:
            example = _build_cell_annotation(record, config)
            if example is not None:
                examples.append(example)
        if "tissue_identification_rationale" in task_types:
            example = _build_tissue_identification(record, config)
            if example is not None:
                examples.append(example)
        if "cell_state_rationale" in task_types:
            example = _build_cell_state(record, config)
            if example is not None:
                examples.append(example)
        if "differential_diagnosis" in task_types:
            example = _build_differential(record, config, pools)
            if example is not None:
                examples.append(example)
    return examples


def to_stage2_jsonl_row(example: TaskExample) -> dict:
    stage2_task_type = TASK_TO_STAGE2[example.task_type]
    label_for_markers = (
        example.provenance.get("canonical_label_for_grounding")
        or example.gold_fields.get("label")
        or example.metadata.get("cell_type")
    )
    profile = get_marker_profile(str(label_for_markers) if label_for_markers else None, example.metadata)
    row = {
        "sample_id": example.sample_id,
        "task_type": stage2_task_type,
        "genes": example.genes,
        "question_text": example.question_text,
        "answer": example.answer_text,
        "metadata": {
            **example.metadata,
            "source_name": example.source_name,
            "answer_schema_name": example.answer_schema_name,
            "provenance": example.provenance,
        },
    }
    if example.candidate_labels:
        row["candidate_labels"] = example.candidate_labels
    if example.evidence_claims:
        row["evidence_genes"] = [claim.value for claim in example.evidence_claims]
    if profile is not None and profile.negative_markers:
        row["metadata"]["negative_markers"] = list(profile.negative_markers)
    if example.context_claims:
        row["context"] = "; ".join(claim.value for claim in example.context_claims)
    return row