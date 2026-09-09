from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_METRICS = [
    "nuanced_overall",
    "label_semantic_score",
    "expert_routing_quality",
    "abstention_calibration",
    "evidence_alignment",
    "explanation_similarity",
    "marker_mention_coverage",
    "support_grounding_score",
    "negative_evidence_quality",
]

NUANCED_WEIGHTS = {
    "label_semantic_score": 0.25,
    "expert_routing_quality": 0.15,
    "abstention_calibration": 0.15,
    "evidence_alignment": 0.15,
    "explanation_similarity": 0.15,
    "marker_mention_coverage": 0.10,
    "support_grounding_score": 0.10,
}

CHECKPOINT_PATTERN = re.compile(r"after\s+(\d+)\s+training\s+samples", re.IGNORECASE)
CELL_GENE_BLOCK_PATTERN = re.compile(r"cell\s+[A-Z]:\s*(.*?)(?=(?:cell\s+[A-Z]:)|$)", re.IGNORECASE | re.DOTALL)
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+\-/]*")
ABSTAIN_PHRASES = (
    "abstain",
    "insufficient evidence",
    "cannot determine",
    "unable to determine",
    "not enough evidence",
)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "be",
    "because",
    "best",
    "by",
    "cell",
    "cells",
    "confidence",
    "consistent",
    "driven",
    "evidence",
    "for",
    "from",
    "given",
    "high",
    "identity",
    "in",
    "indicates",
    "is",
    "it",
    "label",
    "likely",
    "low",
    "marker",
    "markers",
    "moderate",
    "most",
    "not",
    "of",
    "or",
    "points",
    "presence",
    "present",
    "profile",
    "rationale",
    "reason",
    "supported",
    "supporting",
    "supports",
    "that",
    "the",
    "this",
    "to",
    "unknown",
    "with",
}
CONFIDENCE_ORDER = {"low": 0.0, "moderate": 0.5, "medium": 0.5, "high": 1.0}

LABEL_CANONICAL_REPLACEMENTS = (
    (r"\bt\s*reg\b", "regulatory t cell"),
    (r"\btreg\b", "regulatory t cell"),
    (r"\bnk\s*t\s*cell\b", "natural killer t cell"),
    (r"\bnkt\s*cell\b", "natural killer t cell"),
    (r"\bnk\s*cell\b", "natural killer cell"),
    (r"\bnatural killer t cell\b", "mature nk t cell"),
    (r"\bcd4\+\b", "cd4 positive"),
    (r"\bcd8\+\b", "cd8 positive"),
    (r"\bcd4-positive\b", "cd4 positive"),
    (r"\bcd8-positive\b", "cd8 positive"),
    (r"\balpha beta\b", "alpha-beta"),
    (r"\btype b pancreatic cell\b", "pancreatic beta cell"),
    (r"\bpancreatic b cell\b", "pancreatic beta cell"),
    (r"\bpancreatic d cell\b", "pancreatic delta cell"),
    (r"\balveolar type 1 cell\b", "pulmonary alveolar type 1 cell"),
    (r"\bblood vessel smooth muscle cell\b", "vascular associated smooth muscle cell"),
    (r"\bvascular endothelial cell\b", "endothelial cell of blood vessel"),
    (r"\bretinal blood vessel endothelial cell\b", "retinal endothelial cell of blood vessel"),
    (r"\bmueller cell\b", "muller glial cell"),
)

LABEL_ONTOLOGY_PATTERNS = {
    "endothelial cell": ("endothelial", "vascular", "blood vessel", "capillary", "venule"),
    "capillary endothelial cell": ("capillary endothelial",),
    "epithelial cell": ("epithelial", "epithelium", "urothelial", "enterocyte", "goblet", "club cell", "basal cell", "ductal", "acinar", "ionocyte", "secretory", "serous", "conjunctival", "corneal", "surface epithelial"),
    "enterocyte": ("enterocyte", "best4"),
    "t cell": ("t cell", "alpha-beta t", "cd4 positive", "cd8 positive"),
    "cd4-positive t cell": ("cd4 positive",),
    "cd8-positive t cell": ("cd8 positive",),
    "regulatory t cell": ("regulatory t cell",),
    "natural killer cell": ("natural killer cell",),
    "natural killer t cell": ("nk t cell", "mature nk t cell", "natural killer t cell"),
    "myeloid cell": ("myeloid", "monocyte", "macrophage", "phagocyte", "dendritic", "granulocyte", "neutrophil"),
    "mononuclear phagocyte": ("mononuclear phagocyte", "monocyte", "macrophage"),
    "monocyte": ("monocyte", "classical monocyte", "intermediate monocyte", "non-classical monocyte"),
    "dendritic cell": ("dendritic cell",),
    "plasmacytoid dendritic cell": ("plasmacytoid dendritic",),
    "stromal cell": ("stromal", "fibroblast", "keratocyte", "pericyte", "smooth muscle", "theca", "stellate", "schwann", "mesenchymal"),
    "fibroblast": ("fibroblast", "keratocyte"),
    "pericyte": ("pericyte",),
    "smooth muscle cell": ("smooth muscle",),
    "endocrine cell": ("pancreatic beta", "pancreatic delta", "pancreatic alpha", "enteroendocrine", "endocrine"),
    "blood cell": ("erythrocyte", "platelet"),
    "neural cell": ("neuron", "photoreceptor", "glial", "muller"),
    "photoreceptor cell": ("photoreceptor",),
    "cardiac muscle cell": ("cardiac muscle", "ventricular cardiac muscle"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assess Stage3 train/test performance over time using answer-derived, nuanced metrics "
            "instead of the built-in exact-match evaluator. The input can be a single JSONL file, "
            "a single run directory, or a parent directory containing multiple run directories."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("test_results.jsonl"),
        help="Path to a Stage3 results file, run directory, or parent runs directory.",
    )
    parser.add_argument(
        "--phase",
        choices=["train", "test"],
        default="test",
        help="Which phase rows to evaluate from the input JSONL.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Derived metrics to display. Use 'all' to show every derived metric.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to save the computed summary as JSON.",
    )
    return parser.parse_args()


def checkpoint_step(label: str) -> int:
    if label.strip().lower() == "initial":
        return 0
    match = CHECKPOINT_PATTERN.search(label)
    if match:
        return int(match.group(1))
    return 10**9


def sort_checkpoint_labels(labels: Iterable[str]) -> list[str]:
    return sorted(labels, key=lambda label: (checkpoint_step(label), label))


def results_filename(phase_filter: str) -> str:
    return f"{phase_filter}_results.jsonl"


def read_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def build_source_descriptor(path: Path, phase_filter: str) -> dict[str, Any]:
    result_path = path if path.is_file() else path / results_filename(phase_filter)
    run_dir = result_path.parent
    manifest = read_json_file(run_dir / "run_manifest.json")
    return {
        "path": result_path,
        "run_dir": run_dir,
        "source_name": run_dir.name,
        "created_at_utc": manifest.get("created_at_utc") or "",
    }


def collect_input_sources(path: Path, phase_filter: str) -> list[dict[str, Any]]:
    if path.is_file():
        return [build_source_descriptor(path, phase_filter)]

    if not path.exists():
        raise ValueError(f"Input path does not exist: {path}")

    phase_file = path / results_filename(phase_filter)
    if phase_file.exists():
        return [build_source_descriptor(path, phase_filter)]

    sources = [
        build_source_descriptor(child, phase_filter)
        for child in path.iterdir()
        if child.is_dir() and (child / results_filename(phase_filter)).exists()
    ]
    sources.sort(
        key=lambda source: (
            source["created_at_utc"] == "",
            source["created_at_utc"],
            source["source_name"],
        )
    )
    if not sources:
        raise ValueError(f"No run directories with {results_filename(phase_filter)} found under {path}")
    for index, source in enumerate(sources):
        source["source_order"] = index
    return sources


def checkpoint_sort_key(label_meta: dict[str, dict[str, Any]], label: str) -> tuple[Any, ...]:
    meta = label_meta[label]
    return (
        meta.get("source_order", 0),
        meta.get("step", checkpoint_step(meta.get("raw_label", label))),
        meta.get("display_label", label),
    )


def parse_json_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def to_iterable(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [value]


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def normalize_value(value: Any) -> str:
    return normalize_text(value)


def tokenize(text: Any, *, drop_stopwords: bool = True) -> list[str]:
    tokens = [token.lower() for token in TOKEN_PATTERN.findall(str(text or ""))]
    if not drop_stopwords:
        return tokens
    return [token for token in tokens if token not in STOPWORDS]


def token_f1(left: Any, right: Any) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 2 * precision * recall / (precision + recall)


def set_f1(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if overlap == 0:
        return 0.0
    precision = overlap / len(left)
    recall = overlap / len(right)
    return 2 * precision * recall / (precision + recall)


def normalized_set(values: Any) -> set[str]:
    return {normalize_value(item) for item in to_iterable(values) if normalize_value(item)}


def is_unknown(value: Any) -> bool:
    normalized = normalize_value(value)
    return normalized in {"", "unknown", "n/a", "none"}


def parse_confidence(value: Any) -> float | None:
    normalized = normalize_value(value)
    for key, score in CONFIDENCE_ORDER.items():
        if key in normalized:
            return score
    return None


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = normalize_value(value)
    return normalized in {"1", "true", "yes", "y"}


def extract_ground_truth(record: dict[str, Any]) -> dict[str, Any]:
    sample = record.get("sample") or {}
    return parse_json_dict(sample.get("ground_truth"))


def extract_answer_fields(record: dict[str, Any]) -> dict[str, Any]:
    generator_output = record.get("generator_output") or {}
    answer_fields = generator_output.get("answer_fields")
    if isinstance(answer_fields, dict):
        return dict(answer_fields)
    raw_payload = generator_output.get("raw")
    if isinstance(raw_payload, dict) and isinstance(raw_payload.get("answer_fields"), dict):
        return dict(raw_payload["answer_fields"])
    return {}


def extract_freeform_answer(record: dict[str, Any]) -> str:
    generator_output = record.get("generator_output") or {}
    return str(
        generator_output.get("final_freeform_answer")
        or generator_output.get("final_answer")
        or ""
    )


def extract_input_genes(record: dict[str, Any]) -> set[str]:
    sample = record.get("sample") or {}
    metadata = sample.get("metadata") or {}
    request = metadata.get("stage3_request") or {}
    genes = request.get("genes")
    if isinstance(genes, list) and genes:
        return {normalize_value(gene) for gene in genes if normalize_value(gene)}

    question_text = request.get("question_text") or ""
    question_text = str(question_text)

    parsed_genes: set[str] = set()
    for match in CELL_GENE_BLOCK_PATTERN.finditer(question_text):
        for raw_gene in match.group(1).split(","):
            gene = normalize_value(raw_gene.strip().strip(".;"))
            if gene and " " not in gene:
                parsed_genes.add(gene)
    if parsed_genes:
        return parsed_genes

    match = re.search(r"gene list:\s*(.*)", question_text, re.IGNORECASE)
    if not match:
        return set()
    return {normalize_value(gene) for gene in match.group(1).split(",") if normalize_value(gene)}


def extract_stage3_request(record: dict[str, Any]) -> dict[str, Any]:
    sample = record.get("sample") or {}
    metadata = sample.get("metadata") or {}
    request = metadata.get("stage3_request") or {}
    return request if isinstance(request, dict) else {}


def extract_called_experts(record: dict[str, Any]) -> list[str]:
    tool_calls = record.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        return []
    experts: list[str] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        expert_name = normalize_value(tool_call.get("expert_name"))
        if expert_name:
            experts.append(expert_name)
    return experts


def expert_routing_scores(record: dict[str, Any]) -> tuple[float, float, float]:
    request = extract_stage3_request(record)
    called_experts = set(extract_called_experts(record))
    oracle_experts = {normalize_value(item) for item in to_iterable(request.get("oracle_experts")) if normalize_value(item)}
    candidate_experts = {normalize_value(item) for item in to_iterable(request.get("candidate_experts")) if normalize_value(item)}

    target_experts = oracle_experts or candidate_experts
    if not target_experts and not called_experts:
        return 1.0, 1.0, 1.0

    if target_experts:
        routing_recall = len(called_experts & target_experts) / len(target_experts)
    else:
        routing_recall = 1.0 if not called_experts else 0.0

    if called_experts:
        allowed_experts = target_experts or candidate_experts
        routing_precision = len(called_experts & allowed_experts) / len(called_experts) if allowed_experts else 0.0
    else:
        routing_precision = 0.0 if target_experts else 1.0

    if target_experts:
        exact_expert_hit = 1.0 if called_experts & target_experts else 0.0
    else:
        exact_expert_hit = 1.0 if not called_experts else 0.0

    routing_quality = 0.5 * exact_expert_hit + 0.3 * routing_recall + 0.2 * routing_precision
    return routing_quality, routing_recall, routing_precision


def infer_abstention(answer_fields: dict[str, Any], freeform_answer: str) -> bool:
    final_label = answer_fields.get("final_label")
    predicted_tissue = answer_fields.get("predicted_tissue")
    if parse_bool(answer_fields.get("abstain")):
        return True
    if is_unknown(final_label) and is_unknown(predicted_tissue):
        return True
    normalized_answer = normalize_text(freeform_answer)
    return any(phrase in normalized_answer for phrase in ABSTAIN_PHRASES) or "unknown" in normalized_answer


def semantic_field_score(expected: Any, observed: Any) -> float:
    if is_unknown(expected) and is_unknown(observed):
        return 1.0
    if is_unknown(expected) or is_unknown(observed):
        return 0.0
    if normalize_value(expected) == normalize_value(observed):
        return 1.0
    return token_f1(expected, observed)


def canonicalize_label_text(value: Any) -> str:
    text = normalize_text(value)
    text = re.sub(r"^most likely identity:\s*", "", text)
    text = re.sub(r"^most likely cell identity:\s*", "", text)
    text = re.sub(r"^predicted label:\s*", "", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = text.replace("+", " positive ")
    text = text.replace("/", " ")
    text = re.sub(r"[,]+", " ", text)
    for pattern, replacement in LABEL_CANONICAL_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\bcell type\b", "cell", text)
    text = re.sub(r"\s+", " ", text).strip(" ;:,.\n\t")
    return text


def extract_label_candidates(value: Any) -> list[str]:
    if is_unknown(value):
        return []

    raw_text = str(value or "")
    parts = re.split(r";|\n", raw_text)
    candidates: list[str] = []
    seen: set[str] = set()
    for part in parts:
        cleaned = re.sub(r"^\s*(?:cell|profile)\s+[a-z0-9]+\s*:\s*", "", part, flags=re.IGNORECASE)
        cleaned = re.sub(r"^\s*[a-z0-9]+\s*:\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.split(" -- ", 1)[0]
        cleaned = cleaned.split(": ", 1)[-1] if re.match(r"^\s*(?:cell|profile)\s+[a-z0-9]+\s*:", part, flags=re.IGNORECASE) else cleaned
        canonical = canonicalize_label_text(cleaned)
        if canonical and canonical not in seen:
            seen.add(canonical)
            candidates.append(canonical)

    if candidates:
        return candidates

    canonical = canonicalize_label_text(raw_text)
    return [canonical] if canonical else []


def label_ontology_terms(label: str) -> set[str]:
    terms: set[str] = set()
    for ontology_term, patterns in LABEL_ONTOLOGY_PATTERNS.items():
        if ontology_term in label or any(pattern in label for pattern in patterns):
            terms.add(ontology_term)
    return terms


def conflicting_label_subtypes(left: str, right: str) -> bool:
    conflict_pairs = (
        ("cd4 positive", "cd8 positive"),
        ("classical monocyte", "intermediate monocyte"),
        ("classical monocyte", "non-classical monocyte"),
        ("intermediate monocyte", "non-classical monocyte"),
        ("pancreatic beta cell", "pancreatic delta cell"),
        ("pancreatic beta cell", "pancreatic alpha cell"),
        ("pancreatic delta cell", "pancreatic alpha cell"),
    )
    return any((first in left and second in right) or (second in left and first in right) for first, second in conflict_pairs)


def ontology_label_pair_score(expected_label: str, observed_label: str) -> float:
    if expected_label == observed_label:
        return 1.0

    lexical_score = token_f1(expected_label, observed_label)
    if expected_label in observed_label or observed_label in expected_label:
        lexical_score = max(lexical_score, 0.85)

    expected_terms = label_ontology_terms(expected_label)
    observed_terms = label_ontology_terms(observed_label)
    shared_terms = expected_terms & observed_terms
    score = lexical_score

    specific_terms = {
        "capillary endothelial cell",
        "enterocyte",
        "regulatory t cell",
        "natural killer t cell",
        "mononuclear phagocyte",
        "monocyte",
        "plasmacytoid dendritic cell",
        "pericyte",
        "fibroblast",
        "smooth muscle cell",
        "photoreceptor cell",
        "cardiac muscle cell",
    }
    broad_terms = {
        "endothelial cell",
        "epithelial cell",
        "t cell",
        "myeloid cell",
        "stromal cell",
        "endocrine cell",
        "blood cell",
        "neural cell",
    }

    if shared_terms & specific_terms:
        score = max(score, 0.85)
    elif shared_terms & broad_terms:
        score = max(score, 0.70)

    if conflicting_label_subtypes(expected_label, observed_label):
        score = min(score, 0.45)

    return min(score, 1.0)


def ontology_field_score(expected: Any, observed: Any) -> float:
    if is_unknown(expected) and is_unknown(observed):
        return 1.0
    if is_unknown(expected) or is_unknown(observed):
        return 0.0

    expected_labels = extract_label_candidates(expected)
    observed_labels = extract_label_candidates(observed)
    if not expected_labels or not observed_labels:
        return semantic_field_score(expected, observed)

    if len(expected_labels) == 1 or len(observed_labels) == 1:
        return max(
            ontology_label_pair_score(expected_label, observed_label)
            for expected_label in expected_labels
            for observed_label in observed_labels
        )

    expected_cover = sum(
        max(ontology_label_pair_score(expected_label, observed_label) for observed_label in observed_labels)
        for expected_label in expected_labels
    ) / len(expected_labels)
    observed_cover = sum(
        max(ontology_label_pair_score(expected_label, observed_label) for expected_label in expected_labels)
        for observed_label in observed_labels
    ) / len(observed_labels)
    return 0.5 * (expected_cover + observed_cover)


def mention_recall(text: str, markers: Iterable[str]) -> float:
    markers = [marker for marker in markers if normalize_value(marker)]
    if not markers:
        return 1.0
    normalized_text = normalize_text(text)
    mentions = sum(1 for marker in markers if normalize_value(marker) in normalized_text)
    return mentions / len(markers)


def support_grounding_score(answer_fields: dict[str, Any], input_genes: set[str], freeform_answer: str) -> float:
    cited_support = normalized_set(answer_fields.get("supporting_genes"))
    if not cited_support:
        return 0.25 if freeform_answer.strip() else 0.0
    if not input_genes:
        return 0.0
    return len(cited_support & input_genes) / len(cited_support)


def grounding_score(answer_fields: dict[str, Any], input_genes: set[str], freeform_answer: str) -> float:
    return support_grounding_score(answer_fields, input_genes, freeform_answer)


def negative_evidence_quality_score(expected: dict[str, Any], observed: dict[str, Any], freeform_answer: str) -> float:
    expected_negative = normalized_set(expected.get("negative_markers"))
    observed_negative = normalized_set(observed.get("negative_markers"))
    structured_alignment = set_f1(observed_negative, expected_negative)
    freeform_coverage = mention_recall(freeform_answer, to_iterable(expected.get("negative_markers")))
    return 0.5 * structured_alignment + 0.5 * freeform_coverage


def evidence_alignment_score(expected: dict[str, Any], observed: dict[str, Any]) -> tuple[float, float, float]:
    expected_support = normalized_set(expected.get("supporting_genes"))
    observed_support = normalized_set(observed.get("supporting_genes"))
    support_score = set_f1(observed_support, expected_support)

    expected_negative = normalized_set(expected.get("negative_markers"))
    observed_negative = normalized_set(observed.get("negative_markers"))
    negative_score = set_f1(observed_negative, expected_negative)

    return 0.9 * support_score + 0.1 * negative_score, support_score, negative_score


def compute_record_metrics(record: dict[str, Any]) -> dict[str, float]:
    ground_truth = extract_ground_truth(record)
    answer_fields = extract_answer_fields(record)
    freeform_answer = extract_freeform_answer(record)
    input_genes = extract_input_genes(record)

    label_lexical_score = semantic_field_score(
        ground_truth.get("final_label"),
        answer_fields.get("final_label"),
    )
    label_semantic_score = ontology_field_score(
        ground_truth.get("final_label"),
        answer_fields.get("final_label"),
    )
    expert_routing_quality, expert_routing_recall, expert_routing_precision = expert_routing_scores(record)
    tissue_semantic_score = semantic_field_score(
        ground_truth.get("predicted_tissue"),
        answer_fields.get("predicted_tissue"),
    )

    expected_abstain = parse_bool(ground_truth.get("abstain")) or is_unknown(ground_truth.get("final_label"))
    observed_abstain = infer_abstention(answer_fields, freeform_answer)
    if expected_abstain == observed_abstain:
        abstention_calibration = 1.0
    elif observed_abstain and not expected_abstain:
        abstention_calibration = 0.35
    else:
        abstention_calibration = 0.0

    evidence_alignment, support_gene_alignment, negative_marker_alignment = evidence_alignment_score(
        ground_truth,
        answer_fields,
    )
    explanation_similarity = token_f1(
        freeform_answer,
        ground_truth.get("golden_answer") or "",
    )
    marker_mention_coverage = 0.9 * mention_recall(
        freeform_answer,
        to_iterable(ground_truth.get("supporting_genes")),
    ) + 0.1 * mention_recall(
        freeform_answer,
        to_iterable(ground_truth.get("negative_markers")),
    )

    confidence_expected = parse_confidence(ground_truth.get("confidence"))
    confidence_observed = parse_confidence(answer_fields.get("confidence"))
    if confidence_expected is None and confidence_observed is None:
        confidence_alignment = 1.0
    elif confidence_expected is None or confidence_observed is None:
        confidence_alignment = 0.5
    else:
        confidence_alignment = max(0.0, 1.0 - abs(confidence_expected - confidence_observed))

    support_grounded_evidence = support_grounding_score(answer_fields, input_genes, freeform_answer)
    negative_evidence_quality = negative_evidence_quality_score(
        ground_truth,
        answer_fields,
        freeform_answer,
    )

    component_scores = {
        "label_semantic_score": label_semantic_score,
        "expert_routing_quality": expert_routing_quality,
        "abstention_calibration": abstention_calibration,
        "evidence_alignment": evidence_alignment,
        "explanation_similarity": explanation_similarity,
        "marker_mention_coverage": marker_mention_coverage,
        "support_grounding_score": support_grounded_evidence,
    }
    weighted_total = sum(NUANCED_WEIGHTS[name] * value for name, value in component_scores.items())

    return {
        "nuanced_overall": weighted_total,
        **component_scores,
        "label_lexical_score": label_lexical_score,
        "grounding_score": support_grounded_evidence,
        "negative_evidence_quality": negative_evidence_quality,
        "tissue_semantic_score": tissue_semantic_score,
        "expert_routing_recall": expert_routing_recall,
        "expert_routing_precision": expert_routing_precision,
        "support_gene_alignment": support_gene_alignment,
        "negative_marker_alignment": negative_marker_alignment,
        "confidence_alignment": confidence_alignment,
    }


def load_results_file(
    path: Path,
    phase_filter: str,
    *,
    source_name: str,
    source_order: int,
    multi_source: bool,
    created_at_utc: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]], int, int, list[str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    label_meta: dict[str, dict[str, Any]] = {}
    sample_count = 0
    summary_rows = 0
    metric_names: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc

            phase = record.get("phase")
            if phase == "test_summary":
                summary_rows += 1
                continue
            if phase != phase_filter:
                continue

            sample = record.get("sample") or {}
            if not sample.get("sample_id"):
                continue

            if phase_filter == "test" and record.get("index") is None:
                continue

            derived_metrics = compute_record_metrics(record)
            record["derived_metrics"] = derived_metrics
            metric_names.update(derived_metrics.keys())

            label = record.get("label")
            if not label:
                if phase_filter == "train":
                    step = record.get("step")
                    epoch = record.get("epoch")
                    if isinstance(step, int):
                        label = f"after {step} training samples"
                    elif epoch is not None:
                        label = f"epoch {epoch}"
                if not label:
                    label = f"unlabeled_{line_number}"
            display_label = label if not multi_source else f"{source_name} :: {label}"
            grouped[display_label].append(record)
            label_meta.setdefault(
                display_label,
                {
                    "display_label": display_label,
                    "raw_label": label,
                    "source_name": source_name,
                    "source_order": source_order,
                    "created_at_utc": created_at_utc,
                    "step": checkpoint_step(label),
                },
            )
            sample_count += 1

    return grouped, label_meta, sample_count, summary_rows, sorted(metric_names)


def load_results(
    path: Path, phase_filter: str
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    int,
    int,
    list[str],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    sources = collect_input_sources(path, phase_filter)
    multi_source = len(sources) > 1
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    label_meta: dict[str, dict[str, Any]] = {}
    sample_count = 0
    summary_rows = 0
    metric_names: set[str] = set()
    used_sources: list[dict[str, Any]] = []
    skipped_sources: list[dict[str, Any]] = []

    for source in sources:
        source_order = int(source.get("source_order", 0))
        file_grouped, file_label_meta, file_sample_count, file_summary_rows, file_metrics = load_results_file(
            source["path"],
            phase_filter,
            source_name=str(source["source_name"]),
            source_order=source_order,
            multi_source=multi_source,
            created_at_utc=str(source.get("created_at_utc") or ""),
        )
        if not file_grouped:
            skipped_sources.append(
                {
                    "source_name": source["source_name"],
                    "path": str(source["path"]),
                    "reason": f"no usable {phase_filter} rows",
                }
            )
            summary_rows += file_summary_rows
            continue

        for label, records in file_grouped.items():
            grouped[label].extend(records)
        label_meta.update(file_label_meta)
        sample_count += file_sample_count
        summary_rows += file_summary_rows
        metric_names.update(file_metrics)
        used_sources.append(source)

    if not grouped:
        raise ValueError(f"No usable {phase_filter} sample rows found in {path}")

    return grouped, label_meta, sample_count, summary_rows, sorted(metric_names), used_sources, skipped_sources


def summarize_checkpoints(
    grouped: dict[str, list[dict[str, Any]]], label_meta: dict[str, dict[str, Any]], metric_names: list[str]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for label in sorted(grouped, key=lambda item: checkpoint_sort_key(label_meta, item)):
        records = grouped[label]
        meta = label_meta.get(label, {})
        sums = {metric: 0.0 for metric in metric_names}
        counts = {metric: 0 for metric in metric_names}

        for record in records:
            metrics = record.get("derived_metrics") or {}
            for metric in metric_names:
                value = metrics.get(metric)
                if isinstance(value, (int, float)):
                    sums[metric] += float(value)
                    counts[metric] += 1

        averages = {
            metric: (sums[metric] / counts[metric]) if counts[metric] else None
            for metric in metric_names
        }
        summaries.append(
            {
                "label": meta.get("display_label", label),
                "raw_label": meta.get("raw_label", label),
                "source_name": meta.get("source_name"),
                "created_at_utc": meta.get("created_at_utc"),
                "step": meta.get("step", checkpoint_step(label)),
                "source_order": meta.get("source_order", 0),
                "samples": len(records),
                "averages": averages,
            }
        )
    return summaries


def compute_deltas(summaries: list[dict[str, Any]], metric_names: list[str]) -> list[dict[str, Any]]:
    baseline = summaries[0]
    baseline_averages = baseline["averages"]
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        deltas = {}
        for metric in metric_names:
            current = summary["averages"][metric]
            baseline_value = baseline_averages[metric]
            if current is None or baseline_value is None:
                deltas[metric] = None
            else:
                deltas[metric] = current - baseline_value
        rows.append(
            {
                "label": summary["label"],
                "step": summary["step"],
                "samples": summary["samples"],
                "deltas": deltas,
            }
        )
    return rows


def sample_change_summary(
    grouped: dict[str, list[dict[str, Any]]], label_meta: dict[str, dict[str, Any]], metric_names: list[str]
) -> dict[str, dict[str, int]]:
    ordered_labels = sorted(grouped, key=lambda item: checkpoint_sort_key(label_meta, item))
    first_label = ordered_labels[0]
    last_label = ordered_labels[-1]

    trajectories: dict[str, dict[str, dict[str, float | None]]] = defaultdict(dict)
    for label in ordered_labels:
        for record in grouped[label]:
            sample_id = ((record.get("sample") or {}).get("sample_id"))
            metrics = record.get("derived_metrics") or {}
            trajectories[sample_id][label] = {
                metric: float(metrics[metric]) if isinstance(metrics.get(metric), (int, float)) else None
                for metric in metric_names
            }

    change_summary: dict[str, dict[str, int]] = {}
    for metric in metric_names:
        improved = 0
        same = 0
        regressed = 0
        missing = 0
        for sample_history in trajectories.values():
            initial = (sample_history.get(first_label) or {}).get(metric)
            final = (sample_history.get(last_label) or {}).get(metric)
            if initial is None or final is None:
                missing += 1
            elif final > initial:
                improved += 1
            elif final < initial:
                regressed += 1
            else:
                same += 1
        change_summary[metric] = {
            "improved": improved,
            "same": same,
            "regressed": regressed,
            "missing": missing,
        }

    return change_summary


def format_float(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def render_metric_table(
    rows: list[dict[str, Any]],
    metric_names: list[str],
    value_key: str,
    signed: bool = False,
) -> str:
    headers = ["checkpoint", "samples", *metric_names]
    rendered_rows: list[list[str]] = []
    for row in rows:
        metric_values = row[value_key]
        rendered_rows.append(
            [
                str(row["label"]),
                str(row["samples"]),
                *[format_float(metric_values.get(metric), signed=signed) for metric in metric_names],
            ]
        )

    widths = [len(header) for header in headers]
    for rendered in rendered_rows:
        for index, cell in enumerate(rendered):
            widths[index] = max(widths[index], len(cell))

    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    for rendered in rendered_rows:
        lines.append("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(rendered)))
    return "\n".join(lines)


def render_change_table(change_summary: dict[str, dict[str, int]], metric_names: list[str]) -> str:
    headers = ["metric", "improved", "same", "regressed", "missing"]
    rows = []
    for metric in metric_names:
        counts = change_summary[metric]
        rows.append(
            [
                metric,
                str(counts["improved"]),
                str(counts["same"]),
                str(counts["regressed"]),
                str(counts["missing"]),
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    for row in rows:
        lines.append("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))
    return "\n".join(lines)


def metric_definitions() -> dict[str, str]:
    return {
        "nuanced_overall": "Weighted blend of semantic label quality, expert routing quality, abstention behavior, evidence alignment, explanation similarity, marker mention coverage, and support grounding.",
        "label_semantic_score": "Ontology-aware compatibility score between predicted and gold final labels, with partial credit for canonical aliases, shared lineage, and parent-child matches.",
        "label_lexical_score": "Legacy token-level semantic similarity between predicted and gold final labels.",
        "expert_routing_quality": "Composite routing score derived from actual expert tool calls versus oracle/candidate experts stored in the request metadata.",
        "abstention_calibration": "Rewards abstaining when the gold answer abstains, and gives a small partial score for cautious abstention on non-abstain items.",
        "evidence_alignment": "Weighted F1 overlap between predicted and gold supporting/negative markers.",
        "explanation_similarity": "Token-F1 similarity between the freeform answer and the golden answer text.",
        "marker_mention_coverage": "How much of the gold support/exclusion marker set is explicitly mentioned in the freeform answer.",
        "support_grounding_score": "Fraction of cited supporting genes that are actually present in the input gene list.",
        "grounding_score": "Deprecated alias for support_grounding_score kept for backward compatibility.",
        "negative_evidence_quality": "Average of structured negative-marker alignment and freeform mention recall for the gold negative markers.",
        "tissue_semantic_score": "Token-level semantic similarity between predicted and gold tissues.",
        "expert_routing_recall": "Fraction of oracle experts that were actually called from the raw tool-call trace.",
        "expert_routing_precision": "Fraction of called experts that fall inside the oracle/candidate expert set for the sample.",
        "support_gene_alignment": "F1 overlap between predicted and gold supporting genes.",
        "negative_marker_alignment": "F1 overlap between predicted and gold negative markers.",
        "confidence_alignment": "Agreement between gold and predicted confidence levels on an ordinal scale.",
    }


def main() -> None:
    args = parse_args()
    grouped, label_meta, sample_count, summary_rows, all_metrics, used_sources, skipped_sources = load_results(
        args.input, args.phase
    )

    requested_metrics = all_metrics if args.metrics == ["all"] else args.metrics
    unknown_metrics = [metric for metric in requested_metrics if metric not in all_metrics]
    if unknown_metrics:
        raise ValueError(
            f"Unknown metric(s): {', '.join(unknown_metrics)}. Available derived metrics: {', '.join(all_metrics)}"
        )

    summaries = summarize_checkpoints(grouped, label_meta, requested_metrics)
    delta_rows = compute_deltas(summaries, requested_metrics)
    change_summary = sample_change_summary(grouped, label_meta, requested_metrics)

    print(f"Input file: {args.input}")
    print(f"Phase: {args.phase}")
    if used_sources:
        print(f"Source runs/files used: {len(used_sources)}")
    if skipped_sources:
        print(f"Skipped empty runs/files: {len(skipped_sources)}")
    print(f"Usable sample rows: {sample_count}")
    print(f"Checkpoints: {len(summaries)}")
    print("Metric source: derived from raw answers, answer_fields, gold annotations, and input genes.")
    if summary_rows and args.phase == "test":
        print(f"Ignored test_summary rows: {summary_rows}")

    print("\nCheckpoint averages")
    print(render_metric_table(summaries, requested_metrics, value_key="averages"))

    print("\nDelta vs initial")
    print(render_metric_table(delta_rows, requested_metrics, value_key="deltas", signed=True))

    print("\nSample-level change from initial to final checkpoint")
    print(render_change_table(change_summary, requested_metrics))

    if args.output_json:
        payload = {
            "input_file": str(args.input),
            "phase": args.phase,
            "sample_rows": sample_count,
            "used_sources": used_sources,
            "skipped_sources": skipped_sources,
            "metric_definitions": metric_definitions(),
            "checkpoints": summaries,
            "delta_vs_initial": delta_rows,
            "sample_change": change_summary,
        }
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved summary JSON to {args.output_json}")


if __name__ == "__main__":
    main()