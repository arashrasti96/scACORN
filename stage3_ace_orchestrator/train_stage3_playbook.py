#!/usr/bin/env python
"""Train a stage-3 ACE playbook over tool-routed orchestration samples.

This is the stage-3 analogue of the example ACE playbook trainer: the Generator
is a LangChain agent that can call local stage-2 expert adapters exposed as
tools, and the ACE loop uses environment feedback plus ground truth to grow a
playbook.

The expected training data can be either:

- existing stage-3 benchmark rows from `dataset_pipeline/data/stage3_exports`
- ad hoc JSON/JSONL/CSV rows that define a question, genes, and ground truth

For stage-3 benchmark rows, the trainer uses `candidate_experts` and
`oracle_experts` as routing-analysis hints in the environment feedback.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from functools import lru_cache
import json
import random
import re
import textwrap
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any, Dict, Iterable, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
CONTRASTIVE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = CONTRASTIVE_ROOT

# When this file is executed directly, Python prepends SCRIPT_DIR to sys.path.
# That causes the sibling requests.py module to mask the third-party requests package.
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != SCRIPT_DIR]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CONTRASTIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTRASTIVE_ROOT))

from ace import Curator, Generator, OfflineAdapter, Playbook, Reflector, Sample
from ace.adaptation import AdapterStepResult, EnvironmentResult, TaskEnvironment
from ace.delta import DeltaBatch

from stage3_ace_orchestrator.config import Stage3Paths, Stage3RuntimeConfig, load_repo_dotenv
from stage3_ace_orchestrator.requests import (
    NormalizedStage3Request,
    extract_ranked_gene_lists_from_question,
    extract_ranked_genes_from_question,
)

load_repo_dotenv(Path(__file__))

if TYPE_CHECKING:
    from stage3_ace_orchestrator.engine import Stage3ExpertRuntime


DEFAULT_PLAYBOOK_PATH = SCRIPT_DIR / "ACE_stage3_playbook.json"
RESUME_STATE_FILENAME = "resume_state.json"
ANSWER_SCHEMA_NAME = "stage3_orchestration_v1"
STAGE3_REFLECTOR_PROMPT = """\
You are reviewing a Stage3 biology tool-orchestration attempt.
The Generator can call local tissue/domain expert tools and must synthesize their public outputs into the requested final answer.

Focus only on executable lessons for the current Stage3 system:
- how to choose one or more expert tools from the question, candidate tissue contexts, ordered gene list, and expert descriptions
- whether the Generator chose the most relevant expert first, missed an obviously better expert, or called unnecessary weakly matched experts
- for multi-profile questions, whether the Generator separated profiles, routed each profile with the right genes, and avoided mixing evidence across profiles
- how to pass the ordered genes, question, task family/type, context, and candidate labels to the expert tool
- how to synthesize expert outputs into final_label, alternative_labels, predicted_tissue, supporting_genes, negative_markers, rationale, expert_models_used, confidence, and abstain
- how to handle conflicting experts or OOD/weak evidence conservatively

Ground-truth policy:
- Treat ground_truth as the golden target.
- If ground_truth includes a golden_answer field, use that golden_answer text as the primary answer-style target and use structured fields (label/markers/abstain) as grounding constraints.

Do not invent validators, preflight systems, provenance fetchers, mapping summaries, auto-fix modules, missing_head, extra_head, reviewer flags, ontology fields, or new output fields.

Question:
{question}

Generator reasoning:
{reasoning}

Generator final answer:
{prediction}

Ground truth:
{ground_truth}

Environment feedback:
{feedback}

Playbook excerpts consulted:
{playbook_excerpt}

Return exactly one JSON object:
{{
    "reasoning": "<brief diagnosis grounded in feedback and the actual final answer>",
    "error_identification": "<short comma-separated issues such as wrong_expert, no_expert_called, missing_profile_routing, profile_mixing, weak_synthesis, wrong_label, bad_supporting_genes, bad_negative_markers, missing_rationale, expert_models_used_mismatch>",
    "root_cause_analysis": "<why the generator failed or succeeded, phrased as a routing/synthesis process issue>",
    "correct_approach": "<3-5 concrete next-time steps using only the current Stage3 tool and answer schema>",
    "key_insight": "<one reusable rule, or [NO_NEW_BULLET] if no new routing/synthesis lesson>",
    "bullet_tags": [{{"id": "<bullet-id>", "tag": "helpful|harmful|neutral"}}]
}}
"""

STAGE3_CURATOR_PROMPT = """\
You are curating a compact Stage3 playbook for a biology expert-tool orchestrator.

The playbook must teach only these executable behaviors:
1. Expert routing: how to decide which local expert tool(s) to call using the question, candidate tissue contexts, ordered gene markers, expert descriptions, and prior tool outcomes, while preferring the most directly matched expert first and avoiding unnecessary weakly matched calls.
2. Tool input restructuring: how to pass the question, ordered genes, task family/type, grounded context, and candidate labels to the chosen expert tool(s), including per-profile routing for multi-cell questions.
3. Expert-output synthesis: how to combine expert final labels, supporting evidence, negative markers, confidence, and metadata into the final answer without mixing evidence across profiles.
4. Conflict/OOD handling: how to abstain or lower confidence when expert outputs conflict, no candidate tissue is supported, or evidence is weak.

Current answer schema is fixed. The Generator may only rely on these final_answer fields:
final_label, alternative_labels, predicted_tissue, supporting_genes, negative_markers, rationale, expert_models_used, confidence, abstain.

Hard prohibitions:
- Do not mention or add rules about preflight, validators, mapping_summary, invoked_experts, provenance_fetched, provenance_string, missing_head, extra_head, head_artifact_fraction, reviewer_flag, AUTO_INSERT, generator_corrections, ontology_parent, URL/DOI provenance, or blocking modules.
- Do not add tissue-specific marker atlases unless the reflection clearly shows that marker set was useful across this task type.
- Do not add rules that depend on oracle_experts; those are training-only feedback, not generator input.
- Do not create more than one ADD operation for a single sample.

Prefer UPDATE/TAG over ADD. Add a new bullet only if the reflection contains a genuinely reusable routing or synthesis lesson.

Training progress: {progress}
Playbook stats: {stats}

Recent reflection:
{reflection}

Current playbook:
{playbook}

Question context:
{question_context}

Allowed sections:
- expert_routing
- tool_input_restructuring
- expert_output_synthesis
- conflict_and_abstain
- evidence_grounding

Return exactly one JSON object:
{{
    "reasoning": "<why each operation is needed, or why no operation is needed>",
    "operations": [
        {{
            "type": "ADD|UPDATE|TAG|REMOVE",
            "section": "expert_routing|tool_input_restructuring|expert_output_synthesis|conflict_and_abstain|evidence_grounding",
            "content": "<one concise executable Stage3 rule, no invented fields>",
            "bullet_id": "<required for UPDATE/TAG/REMOVE>",
            "metadata": {{"helpful": 1, "harmful": 0, "neutral": 0}}
        }}
    ]
}}

If the reflection does not justify a new or changed executable rule, return an empty operations list.
"""

STAGE3_ALLOWED_PLAYBOOK_SECTIONS = {
        "expert_routing",
        "tool_input_restructuring",
        "expert_output_synthesis",
        "conflict_and_abstain",
        "evidence_grounding",
}
STAGE3_FORBIDDEN_PLAYBOOK_TERMS = {
        "preflight",
        "validator",
        "mapping_summary",
        "invoked_experts",
        "provenance_fetched",
        "provenance_string",
        "provenance",
        "missing_head",
        "extra_head",
        "head_artifact",
        "reviewer_flag",
        "auto_insert",
        "generator_corrections",
        "ontology_parent",
        "oracle_experts",
        "url/doi",
        "doi",
        "block with",
}
REQUIRED_OUTPUT_FIELDS = (
    "final_label",
    "alternative_labels",
    "predicted_tissue",
    "supporting_genes",
    "negative_markers",
    "rationale",
    "expert_models_used",
    "confidence",
    "abstain",
)
EXPERT_TISSUE_ALIAS_HINTS = {
    "bladder": "tabula_sapiens_bladder_cell_annotation",
    "bladder organ": "tabula_sapiens_bladder_cell_annotation",
    "cardiac ventricle": "tabula_sapiens_heart_cell_annotation",
    "ear": "tabula_sapiens_ear_cell_annotation",
    "exocrine pancreas": "tabula_sapiens_pancreas_cell_annotation",
    "eye": "tabula_sapiens_eye_cell_annotation",
    "heart": "tabula_sapiens_heart_cell_annotation",
    "lung": "tabula_sapiens_trachea_cell_annotation",
    "mucosa of stomach": "tabula_sapiens_stomach_cell_annotation",
    "ovary": "tabula_sapiens_ovary_cell_annotation",
    "pancreas": "tabula_sapiens_pancreas_cell_annotation",
    "parotid gland": "tabula_sapiens_salivary_gland_cell_annotation",
    "prostate": "tabula_sapiens_prostate_cell_annotation",
    "prostate gland": "tabula_sapiens_prostate_cell_annotation",
    "retinal neural layer": "tabula_sapiens_eye_cell_annotation",
    "right ovary": "tabula_sapiens_ovary_cell_annotation",
    "salivary gland": "tabula_sapiens_salivary_gland_cell_annotation",
    "small intestine": "tabula_sapiens_small_intestine_cell_annotation",
    "spleen": "tabula_sapiens_spleen_cell_annotation",
    "stomach": "tabula_sapiens_stomach_cell_annotation",
    "stomach mucosa": "tabula_sapiens_stomach_cell_annotation",
    "submandibular gland": "tabula_sapiens_salivary_gland_cell_annotation",
    "trachea": "tabula_sapiens_trachea_cell_annotation",
}


def _to_iterable(record: Optional[Iterable[str] | str]) -> List[str]:
    if record is None:
        return []
    if isinstance(record, str):
        values = [item.strip() for item in record.replace(";", ",").split(",")]
        return [value for value in values if value]
    return [str(value).strip() for value in record if str(value).strip()]


def _normalize_token(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _stable_unique(values: Iterable[str]) -> List[str]:
    unique_values: List[str] = []
    seen = set()
    for value in values:
        token = str(value).strip()
        if not token:
            continue
        normalized = _normalize_token(token)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_values.append(token)
    return unique_values


@lru_cache(maxsize=1)
def _available_expert_aliases() -> dict[str, str]:
    from stage3_ace_orchestrator.registry import load_expert_registry

    registry = load_expert_registry(Stage3Paths.discover())
    aliases: dict[str, str] = {}
    for name, descriptor in registry.items():
        aliases[_normalize_token(name)] = name
        aliases[_normalize_token(descriptor.dataset_name)] = name
        aliases[_normalize_token(descriptor.tissue_label)] = name

    for alias, expert_name in EXPERT_TISSUE_ALIAS_HINTS.items():
        if expert_name in registry:
            aliases[_normalize_token(alias)] = expert_name
    return aliases


def _resolve_candidate_experts(tissues: Iterable[str], fallback: Iterable[str] = ()) -> tuple[str, ...]:
    aliases = _available_expert_aliases()
    resolved: List[str] = []

    def _append_if_known(raw_value: str) -> None:
        normalized = _normalize_token(raw_value)
        if not normalized:
            return
        expert_name = aliases.get(normalized)
        if expert_name is None:
            for alias, candidate_name in aliases.items():
                if normalized in alias or alias in normalized:
                    expert_name = candidate_name
                    break
        if expert_name and expert_name not in resolved:
            resolved.append(expert_name)

    for tissue in tissues:
        _append_if_known(str(tissue))
    for value in fallback:
        _append_if_known(str(value))
    return tuple(resolved)


def _build_alternative_labels(gold_label: str | None, labels: Iterable[str]) -> list[str]:
    gold_normalized = _normalize_token(gold_label)
    alternatives = []
    for label in _stable_unique(labels):
        if _normalize_token(label) == gold_normalized:
            continue
        alternatives.append(label)
    return alternatives


def _profile_context_entries(task_context: Dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = task_context.get("profiles") or {}
    if isinstance(entries, dict):
        return {
            str(profile_id): dict(profile_context)
            for profile_id, profile_context in entries.items()
            if isinstance(profile_context, dict)
        }
    if isinstance(entries, list):
        normalized_entries: dict[str, dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            profile_id = str(entry.get("profile_id") or entry.get("id") or "").strip()
            if not profile_id:
                continue
            normalized_entries[profile_id] = dict(entry)
        return normalized_entries
    return {}


def _collect_profile_tissues(record: Dict[str, object], task_context: Dict[str, Any]) -> tuple[str, ...]:
    tissues: list[str] = []
    seen: set[str] = set()

    def _append(raw_value: Any) -> None:
        token = str(raw_value or "").strip()
        normalized = _normalize_token(token)
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        tissues.append(token)

    for profile_context in _profile_context_entries(task_context).values():
        for tissue in _to_iterable(profile_context.get("candidate_expert_tissues")):
            _append(tissue)
        for key in ("expected_tissue", "tissue", "source_dataset", "dataset_name"):
            _append(profile_context.get(key))

    for profile in record.get("profiles") or []:
        if not isinstance(profile, dict):
            continue
        for key in ("expected_tissue", "tissue", "source_dataset", "dataset_name"):
            _append(profile.get(key))

    return tuple(tissues)


def _collect_unknown_ground_truth_paths(value: Any, path: str) -> list[str]:
    if isinstance(value, dict):
        paths: list[str] = []
        for key, nested_value in value.items():
            paths.extend(_collect_unknown_ground_truth_paths(nested_value, f"{path}.{key}"))
        return paths
    if isinstance(value, list):
        paths: list[str] = []
        for index, nested_value in enumerate(value):
            paths.extend(_collect_unknown_ground_truth_paths(nested_value, f"{path}[{index}]"))
        return paths
    return [path] if _normalize_label(value) == "unknown" else []


def _profile_ground_truths(record: Dict[str, object], task_context: Dict[str, Any]) -> dict[str, dict[str, Any]]:
    verification = dict(record.get("verification") or {})
    raw_profiles = verification.get("profiles") or {}
    if not isinstance(raw_profiles, dict):
        return {}

    profile_contexts = _profile_context_entries(task_context)
    profile_ground_truths: dict[str, dict[str, Any]] = {}
    for raw_profile_id, raw_profile in raw_profiles.items():
        if not isinstance(raw_profile, dict):
            continue
        profile_id = str(raw_profile_id).strip()
        if not profile_id:
            continue
        profile_context = profile_contexts.get(profile_id, {})
        gold_label = str(
            raw_profile.get("expected_final_label")
            or raw_profile.get("final_label")
            or raw_profile.get("cell_type")
            or ""
        ).strip()
        alternative_labels = _build_alternative_labels(
            gold_label,
            [gold_label] + _to_iterable(raw_profile.get("accepted_label_variants")),
        )
        profile_ground_truths[profile_id] = {
            "final_label": gold_label,
            "alternative_labels": alternative_labels,
            "predicted_tissue": str(
                profile_context.get("expected_tissue")
                or profile_context.get("tissue")
                or raw_profile.get("expected_tissue")
                or raw_profile.get("tissue")
                or ""
            ).strip(),
            "supporting_genes": _stable_unique(
                _to_iterable(raw_profile.get("allowed_evidence_genes") or raw_profile.get("evidence_genes_used"))
            ),
            "negative_markers": _stable_unique(
                _to_iterable(raw_profile.get("allowed_negative_markers") or raw_profile.get("negative_markers_used"))
            ),
        }
    return profile_ground_truths


def _normalized_ground_truth(record: Dict[str, object]) -> tuple[dict[str, Any], tuple[str, ...]]:
    ground_truth = dict(record.get("ground_truth") or {})
    verification = dict(record.get("verification") or {})
    task_context = dict(ground_truth.get("task_context") or {})
    task_family = str(record.get("task_family") or "")
    profile_ground_truths = _profile_ground_truths(record, task_context)
    unknown_ground_truth_paths = _stable_unique(
        _collect_unknown_ground_truth_paths(ground_truth, "ground_truth")
        + _collect_unknown_ground_truth_paths(verification, "verification")
    )

    gold_label = str(
        ground_truth.get("expected_final_label")
        or ground_truth.get("final_label")
        or ground_truth.get("cell_type")
        or record.get("label")
        or "unknown"
    )
    predicted_tissue = str(task_context.get("expected_tissue") or ground_truth.get("tissue") or "unknown")
    supporting_genes = _stable_unique(
        _to_iterable(
            ground_truth.get("supported_evidence_genes")
            or verification.get("allowed_evidence_genes")
            or verification.get("evidence_genes_used")
        )
    )
    negative_markers = _stable_unique(
        _to_iterable(
            ground_truth.get("available_negative_markers")
            or verification.get("allowed_negative_markers")
            or verification.get("negative_markers_used")
        )
    )
    abstain = _to_bool(ground_truth.get("expected_abstain"))
    candidate_labels = _stable_unique(
        _to_iterable(task_context.get("candidate_labels"))
        or _to_iterable(verification.get("accepted_label_variants"))
    )
    golden_answer = str(record.get("answer") or "").strip()

    if task_family == "multi_list_user_questions" and profile_ground_truths:
        profile_labels = {
            profile_id: str(profile_ground_truth.get("final_label") or "").strip()
            for profile_id, profile_ground_truth in profile_ground_truths.items()
            if str(profile_ground_truth.get("final_label") or "").strip()
        }
        profile_label_summary = "; ".join(
            f"Cell {profile_id}: {label}" for profile_id, label in profile_labels.items()
        )
        profile_tissue_summary = "; ".join(
            f"Cell {profile_id}: {profile_ground_truth.get('predicted_tissue')}"
            for profile_id, profile_ground_truth in profile_ground_truths.items()
            if str(profile_ground_truth.get("predicted_tissue") or "").strip()
        )
        supporting_genes = _stable_unique(
            gene
            for profile_ground_truth in profile_ground_truths.values()
            for gene in _to_iterable(profile_ground_truth.get("supporting_genes"))
        )
        negative_markers = _stable_unique(
            marker
            for profile_ground_truth in profile_ground_truths.values()
            for marker in _to_iterable(profile_ground_truth.get("negative_markers"))
        )
        candidate_labels = _stable_unique(
            label
            for profile_ground_truth in profile_ground_truths.values()
            for label in [profile_ground_truth.get("final_label"), *_to_iterable(profile_ground_truth.get("alternative_labels"))]
            if str(label or "").strip()
        )
        normalized_ground_truth = {
            "final_label": profile_label_summary,
            "profile_labels": profile_labels,
            "profile_ground_truths": profile_ground_truths,
            "alternative_labels": [],
            "predicted_tissue": profile_tissue_summary,
            "supporting_genes": supporting_genes,
            "negative_markers": negative_markers,
            "golden_answer": golden_answer,
            "rationale": "",
            "expert_models_used": [],
            "confidence": "high" if verification.get("verified", True) else "medium",
            "abstain": False,
            "ground_truth_unknown_paths": unknown_ground_truth_paths,
        }
        return normalized_ground_truth, tuple(candidate_labels)

    if gold_label and not abstain:
        candidate_labels = _stable_unique([gold_label] + candidate_labels)

    normalized_ground_truth = {
        "final_label": "unknown" if abstain else gold_label,
        "alternative_labels": _build_alternative_labels(gold_label, candidate_labels),
        "predicted_tissue": "unknown" if abstain else predicted_tissue,
        "supporting_genes": supporting_genes,
        "negative_markers": negative_markers,
        "golden_answer": golden_answer,
        "rationale": "",
        "expert_models_used": [],
        "confidence": "high" if verification.get("verified", True) else "medium",
        "abstain": abstain,
        "ground_truth_unknown_paths": unknown_ground_truth_paths,
    }
    return normalized_ground_truth, tuple(candidate_labels)


def _request_from_grounded_qna_record(record: Dict[str, object], index: int) -> NormalizedStage3Request:
    raw_ground_truth = dict(record.get("ground_truth") or {})
    task_context = dict(raw_ground_truth.get("task_context") or {})
    question_text = str(record.get("question") or record.get("question_text") or "Answer the grounded biology query.")
    profile_gene_lists: dict[str, list[str]] = {}
    for profile in record.get("profiles") or []:
        if not isinstance(profile, dict):
            continue
        profile_id = str(profile.get("profile_id") or profile.get("id") or "").strip()
        genes = [str(gene).strip() for gene in (profile.get("genes") or []) if str(gene).strip()][:200]
        if profile_id and genes:
            profile_gene_lists[profile_id] = genes
    if not profile_gene_lists:
        profile_gene_lists = extract_ranked_gene_lists_from_question(question_text, limit=200)

    metadata = {
        "split": record.get("split"),
        "source_dataset": record.get("source_dataset"),
        "source_sample_id": record.get("source_sample_id"),
        "generator": dict(record.get("generator") or {}),
        "verification": dict(record.get("verification") or {}),
        "task_context": task_context,
        "profile_ids": list(profile_gene_lists),
        "profile_gene_lists": profile_gene_lists,
    }
    metadata = {key: value for key, value in metadata.items() if value not in (None, "", {}, [])}

    ground_truth, candidate_labels = _normalized_ground_truth(record)
    task_family = str(record.get("task_family") or "freeform_qa")
    source_dataset = str(record.get("source_dataset") or "")
    candidate_tissues = _to_iterable(task_context.get("candidate_expert_tissues"))
    oracle_tissues = _to_iterable(task_context.get("oracle_expert_tissues"))
    if ground_truth.get("ground_truth_unknown_paths"):
        metadata["ground_truth_unknown_paths"] = list(ground_truth["ground_truth_unknown_paths"])

    profile_tissues = _collect_profile_tissues(record, task_context)
    candidate_experts = _resolve_candidate_experts(
        list(candidate_tissues) + list(profile_tissues),
        fallback=[source_dataset],
    )
    oracle_experts = _resolve_candidate_experts(
        list(oracle_tissues) + list(profile_tissues),
        fallback=[source_dataset],
    )
    if not candidate_experts and source_dataset:
        candidate_experts = _resolve_candidate_experts([source_dataset])
    if not oracle_experts and source_dataset and task_family != "ood_abstain":
        oracle_experts = _resolve_candidate_experts([source_dataset])
    ground_truth["expert_models_used"] = list(oracle_experts or candidate_experts)

    primary_target = "abstain" if task_family == "ood_abstain" else "final_label"
    if task_family == "multi_list_user_questions" and ground_truth.get("profile_labels"):
        primary_target = "profile_labels"
    genes = extract_ranked_genes_from_question(question_text, limit=200)
    if not genes:
        genes = _to_iterable(record.get("genes") or record.get("ordered_genes"))[:200]
    if len(profile_gene_lists) > 1:
        genes = []
    elif len(profile_gene_lists) == 1 and not genes:
        genes = next(iter(profile_gene_lists.values()))[:200]

    return NormalizedStage3Request(
        sample_id=str(record.get("sample_id") or record.get("id") or f"train_{index + 1}"),
        task_family=task_family,
        task_type=NormalizedStage3Request.from_freeform_query(
            question_text=question_text,
            task_family=task_family,
        ).task_type,
        question_text=question_text,
        genes=tuple(genes),
        context=str(record.get("context") or ""),
        answer_schema_name=ANSWER_SCHEMA_NAME,
        required_output_fields=REQUIRED_OUTPUT_FIELDS,
        primary_target=primary_target,
        candidate_labels=candidate_labels,
        candidate_experts=candidate_experts,
        oracle_experts=oracle_experts if task_family != "ood_abstain" else (),
        ground_truth=ground_truth,
        metadata=metadata,
    )


def _parse_json_payload(text: str) -> dict[str, Any] | None:
    trimmed = (text or "").strip()
    if not trimmed:
        return None
    if trimmed.startswith("```"):
        lines = trimmed.splitlines()[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        trimmed = "\n".join(lines).strip()
    if trimmed and not trimmed.startswith("{"):
        start = trimmed.find("{")
        end = trimmed.rfind("}")
        if start != -1 and end != -1 and end > start:
            trimmed = trimmed[start : end + 1]
    try:
        parsed = json.loads(trimmed)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_answer_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _extract_first_mentioned(text: str, options: Iterable[str]) -> str | None:
    normalized_text = _normalize_answer_text(text)
    best_option = None
    best_index = None
    for option in options:
        normalized_option = _normalize_answer_text(str(option))
        if not normalized_option:
            continue
        index = normalized_text.find(normalized_option)
        if index == -1:
            continue
        if best_index is None or index < best_index:
            best_index = index
            best_option = str(option)
    return best_option


def _extract_markers_in_text(text: str, candidates: Iterable[str]) -> list[str]:
    normalized_text = _normalize_answer_text(text)
    matches: list[str] = []
    seen = set()
    for marker in candidates:
        token = str(marker).strip()
        normalized_marker = _normalize_answer_text(token)
        if not normalized_marker or normalized_marker in seen:
            continue
        if normalized_marker in normalized_text:
            matches.append(token)
            seen.add(normalized_marker)
    return matches


def _extract_profile_labels_from_text(text: str, request: NormalizedStage3Request) -> dict[str, str]:
    raw_profile_ground_truths = request.ground_truth.get("profile_ground_truths")
    if not isinstance(raw_profile_ground_truths, dict):
        return {}

    extracted_labels: dict[str, str] = {}
    for profile_id, raw_profile_ground_truth in raw_profile_ground_truths.items():
        if not isinstance(raw_profile_ground_truth, dict):
            continue
        candidates = _stable_unique(
            [raw_profile_ground_truth.get("final_label"), *_to_iterable(raw_profile_ground_truth.get("alternative_labels"))]
        )
        if not candidates:
            continue
        section_match = re.search(
            rf"(?:cell|profile)\s*{re.escape(str(profile_id))}\s*:\s*(.*?)(?=(?:cell|profile)\s*[A-Za-z0-9_-]+\s*:|$)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        section_text = section_match.group(1).strip() if section_match else ""
        matched_label = _extract_first_mentioned(section_text or text, candidates)
        if matched_label:
            extracted_labels[str(profile_id)] = matched_label
    return extracted_labels


def _extract_confidence_from_text(text: str) -> str:
    normalized_text = _normalize_answer_text(text)
    if "confidence" in normalized_text and "low" in normalized_text:
        return "low"
    if "confidence" in normalized_text and ("moderate" in normalized_text or "medium" in normalized_text):
        return "moderate"
    if "confidence" in normalized_text and "high" in normalized_text:
        return "high"
    if " low " in f" {normalized_text} ":
        return "low"
    if " moderate " in f" {normalized_text} " or " medium " in f" {normalized_text} ":
        return "moderate"
    if " high " in f" {normalized_text} ":
        return "high"
    return ""


def _extract_abstain_from_text(text: str) -> bool:
    normalized_text = _normalize_answer_text(text)
    if "abstain" in normalized_text:
        return True
    return any(
        phrase in normalized_text
        for phrase in (
            "insufficient evidence",
            "cannot determine",
            "unable to determine",
            "not enough evidence",
            "unknown",
        )
    )


def _extract_freeform_answer_fields(
    final_answer_text: str,
    *,
    request: NormalizedStage3Request,
    called_experts: list[str],
) -> dict[str, Any]:
    normalized_text = _normalize_answer_text(final_answer_text)
    candidate_labels = list(request.candidate_labels)
    final_label = _extract_first_mentioned(final_answer_text, candidate_labels)
    expected_label = str(request.ground_truth.get("final_label") or "").strip()
    if final_label is None and expected_label and _normalize_answer_text(expected_label) in normalized_text:
        final_label = expected_label

    abstain = _extract_abstain_from_text(final_answer_text)
    if abstain and not final_label:
        final_label = "unknown"

    predicted_tissue = ""
    expected_tissue = str(request.ground_truth.get("predicted_tissue") or "").strip()
    if expected_tissue and _normalize_answer_text(expected_tissue) in normalized_text:
        predicted_tissue = expected_tissue

    expected_supporting = _to_iterable(request.ground_truth.get("supporting_genes"))
    supporting_genes = _extract_markers_in_text(final_answer_text, expected_supporting)

    expected_negative = _to_iterable(request.ground_truth.get("negative_markers"))
    negative_markers = _extract_markers_in_text(final_answer_text, expected_negative)

    profile_labels = _extract_profile_labels_from_text(final_answer_text, request)

    alternative_labels = [
        label
        for label in candidate_labels
        if final_label is None or _normalize_label(label) != _normalize_label(final_label)
        if _normalize_answer_text(label) in normalized_text
    ]

    expert_models_used = [
        expert_name
        for expert_name in called_experts
        if _normalize_answer_text(expert_name) in normalized_text
    ]

    confidence = _extract_confidence_from_text(final_answer_text)
    if not confidence:
        confidence = "moderate"

    return {
        "final_label": final_label or ("unknown" if abstain else ""),
        "alternative_labels": alternative_labels,
        "predicted_tissue": predicted_tissue or ("unknown" if abstain else ""),
        "profile_labels": profile_labels,
        "supporting_genes": supporting_genes,
        "negative_markers": negative_markers,
        "rationale": final_answer_text,
        "expert_models_used": expert_models_used,
        "confidence": confidence,
        "abstain": abstain,
    }


def _observed_answer_fields(
    final_answer_text: str,
    *,
    request: NormalizedStage3Request,
    called_experts: list[str],
    generator_raw: Optional[Dict[str, Any]] = None,
) -> tuple[dict[str, Any], str]:
    if generator_raw:
        answer_fields = generator_raw.get("answer_fields")
        if isinstance(answer_fields, dict):
            return dict(answer_fields), "answer_fields"
    parsed_answer = _parse_json_payload(final_answer_text)
    if parsed_answer is not None:
        return parsed_answer, "json"
    return _extract_freeform_answer_fields(
        final_answer_text,
        request=request,
        called_experts=called_experts,
    ), "freeform"


def _normalize_label(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "|".join(_normalize_label(item) for item in value)
    return str(value).strip().lower()


def _primary_target_match(primary_target: str, expected_value: Any, observed_value: Any) -> bool:
    if primary_target == "abstain":
        return _to_bool(expected_value) == _to_bool(observed_value)

    if primary_target == "profile_labels":
        if not isinstance(expected_value, dict):
            return False
        expected_labels = {
            str(profile_id): _normalize_label(label)
            for profile_id, label in expected_value.items()
            if _normalize_label(label)
        }
        if not isinstance(observed_value, dict):
            return False
        observed_labels = {
            str(profile_id): _normalize_label(label)
            for profile_id, label in observed_value.items()
            if _normalize_label(label)
        }
        return bool(expected_labels) and all(observed_labels.get(profile_id) == label for profile_id, label in expected_labels.items())

    if primary_target == "supporting_genes":
        expected_genes = {_normalize_label(item) for item in _to_iterable(expected_value) if _normalize_label(item)}
        observed_genes = {_normalize_label(item) for item in _to_iterable(observed_value) if _normalize_label(item)}
        return bool(observed_genes) and observed_genes.issubset(expected_genes)

    return _normalize_label(expected_value) == _normalize_label(observed_value)


def _normalized_value_set(values: Any) -> set[str]:
    return {_normalize_label(item) for item in _to_iterable(values) if _normalize_label(item)}


def _mentions_any_marker(text: Any, markers: Iterable[str]) -> bool:
    normalized_text = _normalize_label(text)
    if not normalized_text:
        return False
    return any(_normalize_label(marker) in normalized_text for marker in markers if _normalize_label(marker))


def _request_from_record(record: Dict[str, object], index: int) -> NormalizedStage3Request:
    if any(key in record for key in ("generator", "verification", "question", "id")):
        return _request_from_grounded_qna_record(record, index)

    if any(key in record for key in ("task_family", "answer_schema_name", "required_output_fields", "candidate_experts")):
        return NormalizedStage3Request.from_stage3_row(record)

    sample_id = str(record.get("sample_id") or f"train_{index + 1}")
    question_text = str(record.get("question_text") or record.get("question") or "Answer the grounded biology query.")
    task_family = str(record.get("task_family") or "freeform_qa")
    task_type = str(record.get("task_type") or "") or None
    genes = tuple(_to_iterable(record.get("genes") or record.get("ordered_genes") or record.get("ordered_proteins")))
    context = str(record.get("context") or "")
    candidate_labels = tuple(_to_iterable(record.get("candidate_labels")))

    primary_target = str(record.get("primary_target") or "final_label")
    raw_ground_truth = record.get("ground_truth")
    if isinstance(raw_ground_truth, dict):
        ground_truth = dict(raw_ground_truth)
    else:
        resolved_ground_truth = raw_ground_truth or record.get("label") or record.get("annotation") or record.get("answer")
        ground_truth = {primary_target: resolved_ground_truth} if resolved_ground_truth is not None else {}

    return NormalizedStage3Request(
        sample_id=sample_id,
        task_family=task_family,
        task_type=task_type or NormalizedStage3Request.from_freeform_query(question_text=question_text, task_family=task_family).task_type,
        question_text=question_text,
        genes=genes,
        context=context,
        answer_schema_name=str(record.get("answer_schema_name")) if record.get("answer_schema_name") else None,
        required_output_fields=tuple(_to_iterable(record.get("required_output_fields"))),
        primary_target=primary_target,
        candidate_labels=candidate_labels,
        candidate_experts=tuple(_to_iterable(record.get("candidate_experts"))),
        oracle_experts=tuple(_to_iterable(record.get("oracle_experts"))),
        ground_truth=ground_truth,
        metadata=dict(record.get("metadata") or {}),
    )


class Stage3TrainingEnvironment(TaskEnvironment):
    """Evaluate tool-routed stage-3 answers against structured ground truth."""

    def __init__(self, runtime: Stage3ExpertRuntime) -> None:
        self.runtime = runtime

    def evaluate(self, sample: Sample, generator_output) -> EnvironmentResult:
        request_dict = dict(sample.metadata.get("stage3_request") or {})
        request = NormalizedStage3Request(**request_dict)
        final_answer_text = str(generator_output.final_answer or "")
        call_log = self.runtime.get_call_log()

        called_experts = []
        for call in call_log:
            expert_name = str(call.get("expert_name") or "")
            if expert_name and expert_name not in called_experts:
                called_experts.append(expert_name)

        parsed_answer, answer_format = _observed_answer_fields(
            final_answer_text,
            request=request,
            called_experts=called_experts,
            generator_raw=getattr(generator_output, "raw", None),
        )

        required_fields = set(request.required_output_fields)
        if required_fields:
            present_fields = {
                field
                for field in required_fields
                if parsed_answer.get(field) not in (None, "", [], {})
            }
            missing_fields = sorted(required_fields - present_fields)
            schema_score = len(required_fields & present_fields) / len(required_fields)
        else:
            missing_fields = []
            schema_score = 1.0

        primary_target = request.primary_target or "final_label"
        expected_value = request.ground_truth.get(primary_target)
        observed_value = parsed_answer.get(primary_target)
        ground_truth_unknown_paths = _to_iterable(
            request.metadata.get("ground_truth_unknown_paths") or request.ground_truth.get("ground_truth_unknown_paths")
        )
        if _normalize_label(expected_value) == "unknown" and ground_truth_unknown_paths:
            primary_match = True
        else:
            primary_match = _primary_target_match(primary_target, expected_value, observed_value)

        predicted_tissue_expected = request.ground_truth.get("predicted_tissue")
        predicted_tissue_observed = parsed_answer.get("predicted_tissue")
        if primary_target == "profile_labels":
            predicted_tissue_match = True
        else:
            predicted_tissue_match = _normalize_label(predicted_tissue_expected) == _normalize_label(predicted_tissue_observed)

        candidate_hit = 1.0 if set(called_experts) & set(request.candidate_experts) else 0.0
        oracle_hit = 1.0 if set(called_experts) & set(request.oracle_experts) else 0.0

        supporting_genes = parsed_answer.get("supporting_genes") or []
        supporting_gene_subset = 1.0
        if supporting_genes:
            input_genes = set(request.genes)
            if not input_genes:
                for profile_genes in (request.metadata.get("profile_gene_lists") or {}).values():
                    input_genes.update(str(gene) for gene in _to_iterable(profile_genes))
            subset_ok = all(str(gene) in input_genes for gene in supporting_genes)
            supporting_gene_subset = 1.0 if subset_ok else 0.0
        expected_supporting_genes = _to_iterable(request.ground_truth.get("supporting_genes"))
        observed_supporting_genes = _to_iterable(supporting_genes)
        expected_supporting_set = _normalized_value_set(expected_supporting_genes)
        observed_supporting_set = _normalized_value_set(observed_supporting_genes)
        supporting_gene_evidence_hit = 1.0
        if expected_supporting_set:
            supporting_gene_evidence_hit = 1.0 if observed_supporting_set & expected_supporting_set else 0.0

        negative_markers = parsed_answer.get("negative_markers") or []
        expected_negative_markers = _to_iterable(request.ground_truth.get("negative_markers"))
        observed_negative_markers = _to_iterable(negative_markers)
        expected_negative_set = _normalized_value_set(expected_negative_markers)
        observed_negative_set = _normalized_value_set(observed_negative_markers)
        negative_marker_subset = 1.0
        negative_marker_present = 1.0
        if expected_negative_set:
            negative_marker_present = 1.0 if observed_negative_set else 0.0
            negative_marker_subset = 1.0 if observed_negative_set and observed_negative_set.issubset(expected_negative_set) else 0.0

        rationale = str(parsed_answer.get("rationale") or final_answer_text or "")
        rationale_mentions_supporting = 1.0
        if expected_supporting_genes:
            rationale_mentions_supporting = 1.0 if _mentions_any_marker(rationale, expected_supporting_genes) else 0.0
        rationale_mentions_negative = 1.0
        if expected_negative_markers:
            rationale_mentions_negative = 1.0 if _mentions_any_marker(rationale, expected_negative_markers) else 0.0

        reported_experts = _to_iterable(parsed_answer.get("expert_models_used"))
        reported_expert_set = set(reported_experts)
        called_expert_set = set(called_experts)
        expert_models_reported_match = 1.0
        if called_experts:
            expert_models_reported_match = 1.0 if reported_expert_set == called_expert_set else 0.0
        elif reported_expert_set:
            expert_models_reported_match = 0.0

        expected_expert_hit = oracle_hit if request.oracle_experts else candidate_hit
        if not request.oracle_experts and not request.candidate_experts:
            expected_expert_hit = 1.0

        exact_match = 1.0 if (
            primary_match
            and not missing_fields
            and supporting_gene_subset == 1.0
            and supporting_gene_evidence_hit == 1.0
            and negative_marker_subset == 1.0
            and negative_marker_present == 1.0
            and rationale_mentions_supporting == 1.0
            and rationale_mentions_negative == 1.0
            and expert_models_reported_match == 1.0
            and expected_expert_hit == 1.0
        ) else 0.0

        feedback_parts = []
        if ground_truth_unknown_paths:
            feedback_parts.append(
                "Ground-truth quality issue: unexpected 'unknown' values found at "
                + ", ".join(ground_truth_unknown_paths)
                + "."
            )
        if not final_answer_text.strip():
            feedback_parts.append("Empty final_answer. Provide a grounded free-form answer with label and marker evidence.")
        if missing_fields:
            feedback_parts.append("Missing required fields: " + ", ".join(missing_fields) + ".")
        if not primary_match:
            feedback_parts.append(
                f"Primary target mismatch for '{primary_target}'. Expected '{expected_value}' but received '{observed_value}'."
            )
        if predicted_tissue_expected and _normalize_label(predicted_tissue_expected) != "unknown" and not predicted_tissue_match:
            feedback_parts.append(
                f"Predicted tissue mismatch. Expected '{predicted_tissue_expected}' but received '{predicted_tissue_observed}'."
            )
        if supporting_gene_subset == 0.0:
            feedback_parts.append("supporting_genes must be chosen from the input genes only.")
        if supporting_gene_evidence_hit == 0.0:
            feedback_parts.append(
                "supporting_genes did not cite expected evidence markers: " + ", ".join(expected_supporting_genes) + "."
            )
        if negative_marker_present == 0.0:
            feedback_parts.append(
                "final_answer must include negative_markers to explain which alternatives are not supported."
            )
        elif negative_marker_subset == 0.0:
            feedback_parts.append(
                "negative_markers should come from the provided/expert-supported exclusion markers: "
                + ", ".join(expected_negative_markers)
                + "."
            )
        if rationale_mentions_supporting == 0.0:
            feedback_parts.append("rationale must explicitly mention supporting evidence genes such as " + ", ".join(expected_supporting_genes) + ".")
        if rationale_mentions_negative == 0.0:
            feedback_parts.append("rationale must explicitly use negative markers such as " + ", ".join(expected_negative_markers) + " to exclude alternatives.")
        if expert_models_reported_match == 0.0:
            feedback_parts.append(
                "expert_models_used must list exactly the expert tools that were actually called: "
                + ", ".join(called_experts)
                + "."
            )
        if request.candidate_experts and candidate_hit == 0.0:
            feedback_parts.append(
                "The orchestrator did not use any candidate expert tools even though candidate experts were available."
            )
        if request.oracle_experts and oracle_hit == 0.0:
            feedback_parts.append(
                "The orchestrator missed the oracle expert set for this training sample."
            )
        if not feedback_parts:
            feedback_parts.append("Correct structured answer with appropriate expert evidence.")
        feedback_parts.append(f"Observed answer format: {answer_format}.")

        golden_answer = str(request.ground_truth.get("golden_answer") or "").strip()
        golden_overlap = 1.0
        if golden_answer and final_answer_text.strip():
            golden_tokens = {token for token in _normalize_answer_text(golden_answer).split(" ") if token}
            answer_tokens = {token for token in _normalize_answer_text(final_answer_text).split(" ") if token}
            if golden_tokens:
                golden_overlap = len(golden_tokens & answer_tokens) / max(1, len(golden_tokens))
        elif golden_answer:
            golden_overlap = 0.0

        routing_summary = (
            " Routing context: called_experts="
            + json.dumps(called_experts, ensure_ascii=False)
            + "; candidate_experts="
            + json.dumps(list(request.candidate_experts), ensure_ascii=False)
            + "; oracle_experts="
            + json.dumps(list(request.oracle_experts), ensure_ascii=False)
            + "."
        )
        tool_trace_summary = " Tool trace summary: " + _tool_trace_summary(call_log) + "."

        metrics = {
            "exact_match": exact_match,
            "primary_target_match": 1.0 if primary_match else 0.0,
            "predicted_tissue_match": 1.0 if predicted_tissue_match else 0.0,
            "schema_coverage": float(schema_score),
            "candidate_expert_hit": candidate_hit,
            "oracle_expert_hit": oracle_hit,
            "expected_expert_hit": expected_expert_hit,
            "supporting_gene_subset": supporting_gene_subset,
            "supporting_gene_evidence_hit": supporting_gene_evidence_hit,
            "negative_marker_present": negative_marker_present,
            "negative_marker_subset": negative_marker_subset,
            "rationale_mentions_supporting_genes": rationale_mentions_supporting,
            "rationale_mentions_negative_markers": rationale_mentions_negative,
            "expert_models_reported_match": expert_models_reported_match,
            "golden_answer_overlap": golden_overlap,
            "ground_truth_unknown_flag": 1.0 if ground_truth_unknown_paths else 0.0,
        }
        return EnvironmentResult(
            feedback=" ".join(feedback_parts) + routing_summary + tool_trace_summary,
            ground_truth=sample.ground_truth,
            metrics=metrics,
        )


def parse_samples(samples_path: Path, max_samples: int | None = None) -> List[Sample]:
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples file not found: {samples_path}")

    return _records_to_samples(_load_raw_records(samples_path), max_samples=max_samples)


def _load_raw_records(samples_path: Path) -> List[Dict[str, object]]:
    suffix = samples_path.suffix.lower()
    raw_records: List[Dict[str, object]] = []

    if suffix == ".jsonl":
        with samples_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("//") or line.startswith("#"):
                    continue
                raw_records.append(json.loads(line))
    elif suffix == ".json":
        data = json.loads(samples_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "samples" in data:
            raw_records = list(data["samples"])
        elif isinstance(data, list):
            raw_records = data
        else:
            raise ValueError("JSON samples must be a list or contain a 'samples' list.")
    elif suffix == ".csv":
        with samples_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            raw_records = list(reader)
    else:
        raise ValueError("Unsupported samples format. Use .json, .jsonl, or .csv files.")

    return raw_records


def _records_to_samples(raw_records: List[Dict[str, object]], max_samples: int | None = None) -> List[Sample]:
    samples: List[Sample] = []
    skipped_missing_ground_truth = 0
    total_records = len(raw_records)
    for idx, record in enumerate(raw_records):
        if max_samples is not None and len(samples) >= max_samples:
            break
        if not isinstance(record, dict):
            raise ValueError(f"Sample #{idx + 1} is not an object: {record!r}")

        request = _request_from_record(record, idx)
        if not request.ground_truth:
            skipped_missing_ground_truth += 1
            print(f"Skipping sample #{idx + 1}: missing ground truth.")
            continue

        metadata = {
            "sample_id": request.sample_id,
            "task_family": request.task_family,
            "task_type": request.task_type,
            "evaluator_context": request.ace_context,
            "stage3_request": asdict(request),
        }

        samples.append(
            Sample(
                question=request.ace_question,
                context=request.ace_context,
                ground_truth=json.dumps(request.ground_truth, ensure_ascii=False),
                metadata=metadata,
            )
        )

    print(
        f"Parsed {len(samples)} samples (skipped {skipped_missing_ground_truth} missing ground truth) "
        f"out of {total_records} records."
    )
    return samples


def _random_train_test_split(
    raw_records: List[Dict[str, object]],
    *,
    train_ratio: float,
    seed: int,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    if len(raw_records) < 2:
        return list(raw_records), []

    shuffled = list(raw_records)
    random.Random(seed).shuffle(shuffled)
    train_count = int(round(len(shuffled) * train_ratio))
    train_count = max(1, min(len(shuffled) - 1, train_count))
    return shuffled[:train_count], shuffled[train_count:]


def _write_jsonl(path: Path, records: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _contains_forbidden_stage3_term(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in STAGE3_FORBIDDEN_PLAYBOOK_TERMS)


def _filter_stage3_playbook_operations(delta: DeltaBatch) -> tuple[DeltaBatch, List[Dict[str, object]]]:
    kept_operations = []
    dropped_operations = []
    for operation in delta.operations:
        op_type = str(operation.type).upper()
        section = str(operation.section or "").strip()
        content = str(operation.content or "")
        if section and section not in STAGE3_ALLOWED_PLAYBOOK_SECTIONS:
            dropped_operations.append(operation.to_json())
            continue
        if op_type in {"ADD", "UPDATE"}:
            if not content.strip() or _contains_forbidden_stage3_term(f"{section} {content}"):
                dropped_operations.append(operation.to_json())
                continue
        kept_operations.append(operation)
    return DeltaBatch(reasoning=delta.reasoning, operations=kept_operations), dropped_operations


class Stage3Curator(Curator):
    """Curator constrained to executable Stage3 expert-routing rules."""

    def curate(self, **kwargs):
        output = super().curate(**kwargs)
        filtered_delta, dropped_operations = _filter_stage3_playbook_operations(output.delta)
        if dropped_operations:
            output.raw["stage3_dropped_operations"] = dropped_operations
        output.delta = filtered_delta
        return output


def build_adapter(
    runtime: Stage3ExpertRuntime,
    agent_model: str,
    playbook: Playbook,
    reflector_model: Optional[str] = None,
) -> OfflineAdapter:
    from ace.llm_providers import LiteLLMClient
    from stage3_ace_orchestrator.orchestrator import build_stage3_generator

    generator = build_stage3_generator(runtime, model=agent_model)

    aux_model = reflector_model or agent_model
    aux_client = LiteLLMClient(model=aux_model, max_tokens=16380)
    reflector = Reflector(aux_client, prompt_template=STAGE3_REFLECTOR_PROMPT)
    curator = Stage3Curator(aux_client, prompt_template=STAGE3_CURATOR_PROMPT)
    return OfflineAdapter(
        generator=generator,
        reflector=reflector,
        curator=curator,
        playbook=playbook,
    )


def prune_playbook(adapter: OfflineAdapter, limit: int = 200, helpful_keep: int = 70) -> None:
    bullets = adapter.playbook.bullets()
    if len(bullets) <= limit:
        return

    print(f"\nWARNING: Playbook too large ({len(bullets)} bullets), pruning to {limit}...")
    all_bullets = list(bullets)

    scored_bullets = [(bullet, bullet.helpful - bullet.harmful) for bullet in all_bullets]
    scored_bullets.sort(key=lambda item: item[1], reverse=True)
    top_helpful = [bullet for bullet, _ in scored_bullets[:helpful_keep]]

    recent_count = max(limit - helpful_keep, 0)
    all_bullets.sort(key=lambda bullet: bullet.created_at, reverse=True)
    recent_bullets = all_bullets[:recent_count]

    kept_bullets = []
    seen_ids = set()
    for bullet in top_helpful + recent_bullets:
        if bullet.id in seen_ids:
            continue
        kept_bullets.append(bullet)
        seen_ids.add(bullet.id)
        if len(kept_bullets) >= limit:
            break

    new_playbook = Playbook()
    for bullet in kept_bullets:
        new_playbook.add_bullet(
            section=bullet.section,
            content=bullet.content,
            bullet_id=bullet.id,
            metadata={
                "helpful": bullet.helpful,
                "harmful": bullet.harmful,
                "neutral": bullet.neutral,
            },
        )

    adapter.playbook = new_playbook
    print(f"Pruned to {len(adapter.playbook.bullets())} bullets (helpful + recent).")


def summarize_run(results, playbook: Playbook, playbook_path: Path) -> None:
    print("\nTraining results:")
    for step, result in enumerate(results, start=1):
        metrics_str = ", ".join(f"{k}={v:.2f}" for k, v in result.environment_result.metrics.items())
        print(
            f"  Step {step}: {result.sample.metadata.get('sample_id')} | "
            f"{result.environment_result.feedback} | {metrics_str}"
        )

    stats = playbook.stats()
    print("\nUpdated playbook snapshot:")
    print(f"  Bullets: {stats['bullets']}")
    print(f"  Sections: {stats['sections']}")
    print(f"  Helpful tags: {stats['tags']['helpful']}")
    print(f"  Saved to: {playbook_path}")


def _shuffled_epoch_samples(
    samples: List[Sample],
    *,
    epoch: int,
    shuffle_seed: int,
) -> List[tuple[int, Sample]]:
    indexed_samples = list(enumerate(samples, start=1))
    epoch_rng = random.Random(shuffle_seed + epoch - 1)
    epoch_rng.shuffle(indexed_samples)
    return indexed_samples


def _default_results_dir(playbook_path: Path) -> Path:
    return playbook_path.parent / f"{playbook_path.stem}_results"


def _save_playbook_checkpoint(playbook: Playbook, playbook_path: Path, *, training_samples_seen: int) -> None:
    playbook.save_to_file(str(playbook_path))
    print(f"Saved playbook checkpoint after {training_samples_seen} training samples to {playbook_path}")


def _resume_state_path(results_dir: Path) -> Path:
    return results_dir / RESUME_STATE_FILENAME


def _write_resume_state(
    *,
    results_dir: Path,
    playbook_path: Path,
    total_samples: int,
    epochs: int,
    train_shuffle_seed: int,
    test_every: int,
    run_initial_test: bool,
    trained_samples_seen: int,
    epoch: int,
    sample_index: Optional[int],
    status: str,
) -> None:
    total_training_steps = total_samples * epochs
    _write_json(
        _resume_state_path(results_dir),
        {
            "version": 1,
            "updated_at_utc": _utc_now_iso(),
            "playbook_path": str(playbook_path.resolve()),
            "results_dir": str(results_dir.resolve()),
            "total_samples": total_samples,
            "epochs": epochs,
            "total_training_steps": total_training_steps,
            "train_shuffle_seed": train_shuffle_seed,
            "test_every": test_every,
            "run_initial_test": run_initial_test,
            "trained_samples_seen": trained_samples_seen,
            "last_completed_epoch": epoch,
            "last_completed_sample_index": sample_index,
            "remaining_training_steps": max(total_training_steps - trained_samples_seen, 0),
            "status": status,
        },
    )


def _load_resume_state(
    *,
    results_dir: Path,
    playbook_path: Path,
    total_samples: int,
    epochs: int,
    train_shuffle_seed: int,
    test_every: int,
    run_initial_test: bool,
) -> Dict[str, object]:
    resume_state_path = _resume_state_path(results_dir)
    if not resume_state_path.exists():
        raise SystemExit(
            f"--resume was requested but no resume state was found at {resume_state_path}."
        )

    with resume_state_path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)

    expected = {
        "playbook_path": str(playbook_path.resolve()),
        "total_samples": total_samples,
        "epochs": epochs,
        "train_shuffle_seed": train_shuffle_seed,
        "test_every": test_every,
        "run_initial_test": run_initial_test,
    }
    mismatches = []
    for key, expected_value in expected.items():
        actual_value = state.get(key)
        if actual_value != expected_value:
            mismatches.append(f"{key}: expected {expected_value!r}, found {actual_value!r}")

    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise SystemExit(
            "Resume state does not match the current run configuration. "
            f"Refusing to resume: {mismatch_text}"
        )

    trained_samples_seen = int(state.get("trained_samples_seen") or 0)
    if trained_samples_seen > 0 and not playbook_path.exists():
        raise SystemExit(
            "Resume state indicates prior progress, but the saved playbook file is missing: "
            f"{playbook_path}"
        )
    return state


def _prepare_results_dir(results_dir: Path, *, resume: bool = False) -> tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    train_results_path = results_dir / "train_results.jsonl"
    test_results_path = results_dir / "test_results.jsonl"
    if resume:
        if not train_results_path.exists():
            train_results_path.write_text("", encoding="utf-8")
        if not test_results_path.exists():
            test_results_path.write_text("", encoding="utf-8")
        print(f"Resuming train results in {train_results_path}")
        print(f"Resuming test results in {test_results_path}")
        return train_results_path, test_results_path

    train_results_path.write_text("", encoding="utf-8")
    test_results_path.write_text("", encoding="utf-8")
    print(f"Saving train results to {train_results_path}")
    print(f"Saving test results to {test_results_path}")
    return train_results_path, test_results_path


def _epoch_resume_offset(*, total_samples: int, trained_samples_seen: int, epoch: int) -> int:
    return max(0, min(total_samples, trained_samples_seen - ((epoch - 1) * total_samples)))


def _persist_training_checkpoint(
    *,
    playbook: Playbook,
    playbook_path: Path,
    results_dir: Path,
    total_samples: int,
    epochs: int,
    train_shuffle_seed: int,
    test_every: int,
    run_initial_test: bool,
    trained_samples_seen: int,
    epoch: int,
    sample_index: Optional[int],
    status: str,
    announce: bool = False,
) -> None:
    playbook.save_to_file(str(playbook_path))
    _write_resume_state(
        results_dir=results_dir,
        playbook_path=playbook_path,
        total_samples=total_samples,
        epochs=epochs,
        train_shuffle_seed=train_shuffle_seed,
        test_every=test_every,
        run_initial_test=run_initial_test,
        trained_samples_seen=trained_samples_seen,
        epoch=epoch,
        sample_index=sample_index,
        status=status,
    )
    if announce:
        print(
            f"Saved resume checkpoint after {trained_samples_seen} training samples to "
            f"{_resume_state_path(results_dir)}"
        )


def _append_jsonl(path: Path, record: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _sample_result_record(sample: Sample) -> Dict[str, object]:
    metadata = sample.metadata or {}
    return {
        "sample_id": metadata.get("sample_id"),
        "task_family": metadata.get("task_family"),
        "task_type": metadata.get("task_type"),
        "question": sample.question,
        "context": sample.context,
        "ground_truth": sample.ground_truth,
        "metadata": metadata,
    }


def _sample_manifest_record(sample: Sample) -> Dict[str, object]:
    metadata = sample.metadata or {}
    stage3_request = metadata.get("stage3_request") or {}
    if not isinstance(stage3_request, dict):
        stage3_request = {}
    ranked_genes = stage3_request.get("genes") or []
    if not isinstance(ranked_genes, list):
        ranked_genes = []

    return {
        "sample_id": metadata.get("sample_id"),
        "task_family": metadata.get("task_family"),
        "task_type": metadata.get("task_type"),
        "question": sample.question,
        "ranked_gene_count": len(ranked_genes),
        "ranked_genes_head": ranked_genes[:20],
        "candidate_experts": _to_iterable(stage3_request.get("candidate_experts")),
        "oracle_experts": _to_iterable(stage3_request.get("oracle_experts")),
        "candidate_labels": _to_iterable(stage3_request.get("candidate_labels")),
        "ground_truth_unknown_paths": _to_iterable(stage3_request.get("ground_truth_unknown_paths")),
    }


def _write_split_manifest(
    path: Path,
    *,
    label: str,
    source_path: Optional[Path],
    samples: List[Sample],
) -> None:
    _write_json(
        path,
        {
            "label": label,
            "created_at_utc": _utc_now_iso(),
            "source_path": str(source_path.resolve()) if source_path is not None else None,
            "count": len(samples),
            "sample_ids": [sample.metadata.get("sample_id") for sample in samples],
            "samples": [_sample_manifest_record(sample) for sample in samples],
        },
    )


def _write_run_manifest(
    *,
    results_dir: Path,
    playbook_path: Path,
    train_source_path: Path,
    test_source_path: Optional[Path],
    samples: List[Sample],
    test_samples: List[Sample],
    epochs: int,
    model: str,
    agent_model: str,
    test_every: int,
    run_initial_test: bool,
    train_shuffle_seed: int,
    max_samples: Optional[int],
    split_metadata: Dict[str, object],
) -> None:
    _write_json(
        results_dir / "run_manifest.json",
        {
            "created_at_utc": _utc_now_iso(),
            "playbook_path": str(playbook_path.resolve()),
            "results_dir": str(results_dir.resolve()),
            "train_source_path": str(train_source_path.resolve()),
            "test_source_path": str(test_source_path.resolve()) if test_source_path is not None else None,
            "train_sample_count": len(samples),
            "test_sample_count": len(test_samples),
            "epochs": epochs,
            "reflector_model": model,
            "agent_model": agent_model,
            "test_every": test_every,
            "run_initial_test": run_initial_test,
            "train_shuffle_seed": train_shuffle_seed,
            "max_samples": max_samples,
            "results_files": {
                "train_results": "train_results.jsonl",
                "test_results": "test_results.jsonl",
                "train_split_manifest": "train_split_manifest.json",
                "test_split_manifest": "test_split_manifest.json",
            },
            "split": split_metadata,
        },
    )
    _write_split_manifest(
        results_dir / "train_split_manifest.json",
        label="train",
        source_path=train_source_path,
        samples=samples,
    )
    if test_source_path is not None or test_samples:
        _write_split_manifest(
            results_dir / "test_split_manifest.json",
            label="test",
            source_path=test_source_path,
            samples=test_samples,
        )


def _generator_output_record(generator_output) -> Dict[str, object]:
    raw = generator_output.raw or {}
    answer_fields = raw.get("answer_fields") if isinstance(raw, dict) else None
    return {
        "reasoning": generator_output.reasoning,
        "final_answer": generator_output.final_answer,
        "final_freeform_answer": generator_output.final_answer,
        "answer_fields": answer_fields if isinstance(answer_fields, dict) else None,
        "bullet_ids": list(generator_output.bullet_ids),
        "bullet_ids_used_from_playbook": list(generator_output.bullet_ids_used_from_playbook),
        "raw": raw,
    }


def _environment_result_record(environment_result: EnvironmentResult) -> Dict[str, object]:
    return {
        "feedback": environment_result.feedback,
        "ground_truth": environment_result.ground_truth,
        "metrics": environment_result.metrics,
    }


def _tool_trace_summary(call_log: List[Dict[str, Any]]) -> str:
    if not call_log:
        return "No expert tools were called."

    summarized_calls = []
    for call in call_log:
        parsed = call.get("parsed_output") or {}
        if not isinstance(parsed, dict):
            parsed = {}
        final_label = parsed.get("final") or parsed.get("final_label") or parsed.get("label") or "unknown"
        evidence = _to_iterable(parsed.get("supporting_genes") or parsed.get("supporting_evidence") or parsed.get("evidence"))
        negative_markers = _to_iterable(parsed.get("negative_markers"))
        confidence = parsed.get("confidence") or "unknown"
        summarized_calls.append(
            {
                "expert_name": call.get("expert_name"),
                "task_type": call.get("task_type"),
                "final_label": final_label,
                "supporting_genes": evidence,
                "negative_markers": negative_markers,
                "confidence": confidence,
            }
        )
    return json.dumps(summarized_calls, ensure_ascii=False)


def _sample_ranked_genes(sample: Sample) -> List[str]:
    stage3_request = (sample.metadata or {}).get("stage3_request") or {}
    if isinstance(stage3_request, dict):
        genes = stage3_request.get("genes") or []
        if isinstance(genes, (list, tuple)):
            return [str(gene).strip() for gene in genes if str(gene).strip()]
    return []


def _sample_profile_gene_lists(sample: Sample) -> Dict[str, List[str]]:
    stage3_request = (sample.metadata or {}).get("stage3_request") or {}
    if not isinstance(stage3_request, dict):
        return {}
    metadata = stage3_request.get("metadata") or {}
    if not isinstance(metadata, dict):
        return {}
    raw_profile_gene_lists = metadata.get("profile_gene_lists") or {}
    if not isinstance(raw_profile_gene_lists, dict):
        return {}

    profile_gene_lists: Dict[str, List[str]] = {}
    for raw_profile_id, raw_genes in raw_profile_gene_lists.items():
        genes = [str(gene).strip() for gene in raw_genes if str(gene).strip()]
        profile_id = str(raw_profile_id).strip()
        if profile_id and genes:
            profile_gene_lists[profile_id] = genes
    return profile_gene_lists


def _set_runtime_sample_genes(runtime: Stage3ExpertRuntime, sample: Sample) -> None:
    genes = _sample_ranked_genes(sample)
    runtime.set_active_ranked_genes(genes)
    runtime.set_active_profile_gene_lists(_sample_profile_gene_lists(sample))


def _save_train_result(
    path: Path,
    result: AdapterStepResult,
    runtime: Stage3ExpertRuntime,
    *,
    epoch: int,
    sample_index: int,
    training_samples_seen: int,
) -> None:
    curator_raw = result.curator_output.raw or {}
    record: Dict[str, object] = {
        "phase": "train",
        "epoch": epoch,
        "step": training_samples_seen,
        "sample_index": sample_index,
        "training_samples_seen": training_samples_seen,
        "label": f"after {training_samples_seen} training samples",
        "adapter_epoch": getattr(result, "epoch", None),
        "adapter_step": getattr(result, "step", None),
        "sample": _sample_result_record(result.sample),
        "generator_output": _generator_output_record(result.generator_output),
        "environment_result": _environment_result_record(result.environment_result),
        "reflection": result.reflection.raw,
        "curator": curator_raw,
        "playbook_snapshot": getattr(result, "playbook_snapshot", None),
        "bullet_metadata": getattr(result, "bullet_metadata", None),
        "tool_calls": runtime.get_call_log(),
    }
    _append_jsonl(path, record)


def _save_train_epoch_summary(
    path: Path,
    *,
    epoch: int,
    processed: int,
    exact_match_total: float,
    training_samples_seen: int,
) -> None:
    average_exact_match = exact_match_total / processed if processed else 0.0
    _append_jsonl(
        path,
        {
            "phase": "train_epoch_summary",
            "epoch": epoch,
            "processed": processed,
            "training_samples_seen": training_samples_seen,
            "average_metrics": {
                "exact_match": average_exact_match,
            },
        },
    )


def _save_train_run_summary(
    path: Path,
    *,
    epochs_completed: int,
    training_samples_seen: int,
    playbook: Playbook,
) -> None:
    _append_jsonl(
        path,
        {
            "phase": "train_run_summary",
            "epochs_completed": epochs_completed,
            "training_samples_seen": training_samples_seen,
            "playbook_stats": playbook.stats(),
        },
    )


def _save_test_result(
    path: Path,
    *,
    label: str,
    index: int,
    sample: Sample,
    generator_output,
    environment_result: EnvironmentResult,
    runtime: Stage3ExpertRuntime,
    playbook: Playbook,
) -> None:
    record: Dict[str, object] = {
        "phase": "test",
        "label": label,
        "index": index,
        "sample": _sample_result_record(sample),
        "generator_output": _generator_output_record(generator_output),
        "environment_result": _environment_result_record(environment_result),
        "playbook_snapshot": playbook.as_prompt(),
        "playbook_stats": playbook.stats(),
        "tool_calls": runtime.get_call_log(),
    }
    _append_jsonl(path, record)


def _save_test_failure_result(
    path: Path,
    *,
    label: str,
    index: int,
    sample: Sample,
    runtime: Stage3ExpertRuntime,
    playbook: Playbook,
    error: Exception,
) -> None:
    record: Dict[str, object] = {
        "phase": "test_failure",
        "label": label,
        "index": index,
        "sample": _sample_result_record(sample),
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "playbook_snapshot": playbook.as_prompt(),
        "playbook_stats": playbook.stats(),
        "tool_calls": runtime.get_call_log(),
    }
    _append_jsonl(path, record)


def _save_test_summary(
    path: Path,
    *,
    label: str,
    completed: int,
    failed: int,
    averages: Dict[str, float],
) -> None:
    _append_jsonl(
        path,
        {
            "phase": "test_summary",
            "label": label,
            "completed": completed,
            "failed": failed,
            "average_metrics": averages,
        },
    )


def evaluate_generator_on_test_set(
    *,
    adapter: OfflineAdapter,
    environment: Stage3TrainingEnvironment,
    runtime: Stage3ExpertRuntime,
    test_samples: List[Sample],
    label: str,
    test_results_path: Optional[Path] = None,
) -> None:
    if not test_samples:
        return

    print("\n" + "-" * 80)
    print(f"Test evaluation ({label}): {len(test_samples)} held-out samples")
    metric_totals: Dict[str, float] = {}
    completed = 0
    failed = 0

    for idx, sample in enumerate(test_samples, start=1):
        runtime.reset_call_log()
        _set_runtime_sample_genes(runtime, sample)
        try:
            generator_output = adapter.generator.generate(
                question=sample.question,
                context=sample.context,
                playbook=adapter.playbook,
                reflection=None,
            )
            environment_result = environment.evaluate(sample, generator_output)
        except Exception as exc:
            _print_sample_failure_debug(sample, runtime, exc)
            failed += 1
            if test_results_path is not None:
                _save_test_failure_result(
                    test_results_path,
                    label=label,
                    index=idx,
                    sample=sample,
                    runtime=runtime,
                    playbook=adapter.playbook,
                    error=exc,
                )
            sample_id = sample.metadata.get("sample_id", f"test_{idx}")
            print(f"  Test {idx}/{len(test_samples)}: {sample_id} | FAILED | {exc}")
            continue
        finally:
            runtime.clear_active_ranked_genes()

        completed += 1
        metrics = environment_result.metrics or {}
        for key, value in metrics.items():
            metric_totals[key] = metric_totals.get(key, 0.0) + float(value)

        called_experts = [str(call.get("expert_name")) for call in runtime.get_call_log() if call.get("expert_name")]
        if test_results_path is not None:
            _save_test_result(
                test_results_path,
                label=label,
                index=idx,
                sample=sample,
                generator_output=generator_output,
                environment_result=environment_result,
                runtime=runtime,
                playbook=adapter.playbook,
            )
        metrics_str = ", ".join(f"{key}={value:.2f}" for key, value in metrics.items())
        sample_id = sample.metadata.get("sample_id", f"test_{idx}")
        experts_str = ", ".join(called_experts) if called_experts else "none"
        print(f"  Test {idx}/{len(test_samples)}: {sample_id} | tools={experts_str} | {metrics_str}")

    if completed:
        averages = {key: value / completed for key, value in metric_totals.items()}
        averages_str = ", ".join(f"{key}={value:.2f}" for key, value in averages.items())
        print(f"Test summary ({label}): completed={completed}, failed={failed} | {averages_str}")
        if test_results_path is not None:
            _save_test_summary(
                test_results_path,
                label=label,
                completed=completed,
                failed=failed,
                averages=averages,
            )
    elif failed:
        print(f"Test summary ({label}): completed=0, failed={failed}")
        if test_results_path is not None:
            _save_test_summary(
                test_results_path,
                label=label,
                completed=0,
                failed=failed,
                averages={},
            )
    print("-" * 80)


def _print_role_debug(result: AdapterStepResult, runtime: Stage3ExpertRuntime) -> None:
    def _print_block(title: str, content: str) -> None:
        print(f"{title}:")
        text = (content or "").strip()
        if text:
            print(textwrap.indent(text, "  "))
        else:
            print("  (empty)")

    epoch = getattr(result, "epoch", 0)
    step = getattr(result, "step", 0)
    sample_id = result.sample.metadata.get("sample_id", f"sample_{step or 'unknown'}")
    print("\n" + "=" * 80)
    print(f"[Epoch {epoch} Step {step}] Sample: {sample_id}")
    _print_block("Question", result.sample.question)
    if result.sample.context:
        _print_block("Context", result.sample.context)
    _print_block("Generator reasoning", result.generator_output.reasoning)
    print(f"Generator final answer: {result.generator_output.final_answer or '(empty)'}")
    print(f"Generator bullet IDs: {', '.join(result.generator_output.bullet_ids) or '(none)'}")
    print(f"bullet IDs used: {', '.join(result.generator_output.bullet_ids_used_from_playbook) or '(none)'}")
    print("Tool calls:")
    tool_calls = runtime.get_call_log()
    if tool_calls:
        print(textwrap.indent(json.dumps(tool_calls, ensure_ascii=False, indent=2), "  "))
    else:
        print("  (none)")

    _print_block("Reflector reasoning", result.reflection.reasoning)
    _print_block("Reflector error identification", result.reflection.error_identification)
    _print_block("Reflector root cause", result.reflection.root_cause_analysis)
    _print_block("Reflector correct approach", result.reflection.correct_approach)
    _print_block("Reflector key insight", result.reflection.key_insight)
    if result.reflection.bullet_tags:
        tags = ", ".join(f"{tag.id}:{tag.tag}" for tag in result.reflection.bullet_tags)
    else:
        tags = "(none)"
    print(f"Reflector bullet tags: {tags}")

    curator_raw = result.curator_output.raw or {}
    curator_reasoning = str(curator_raw.get("reasoning", ""))
    _print_block("Curator reasoning", curator_reasoning)
    operations = curator_raw.get("operations")
    print("Curator operations:")
    if operations:
        print(textwrap.indent(json.dumps(operations, ensure_ascii=False, indent=2), "  "))
    else:
        print("  (none)")

    metrics_str = ", ".join(f"{k}={v:.2f}" for k, v in result.environment_result.metrics.items())
    print(f"Feedback: {result.environment_result.feedback}")
    print(f"Metrics: {metrics_str or '(none)'}")
    print("=" * 80)


def _print_sample_failure_debug(sample: Sample, runtime: Stage3ExpertRuntime, exc: Exception) -> None:
    metadata = sample.metadata or {}
    stage3_request = metadata.get("stage3_request") or {}
    print("\nSample failure debug:")
    print(f"  sample_id: {metadata.get('sample_id')}")
    print(f"  task_family: {metadata.get('task_family')}")
    print(f"  task_type: {metadata.get('task_type')}")
    if isinstance(stage3_request, dict):
        print(f"  candidate_experts: {stage3_request.get('candidate_experts')}")
        print(f"  oracle_experts: {stage3_request.get('oracle_experts')}")
        print(f"  candidate_labels: {stage3_request.get('candidate_labels')}")
    print("  tool_calls_before_failure:")
    call_log = runtime.get_call_log()
    if call_log:
        print(textwrap.indent(json.dumps(call_log, ensure_ascii=False, indent=2), "    "))
    else:
        print("    (none recorded)")
    print(f"  exception: {exc!r}")
    print("Traceback:")
    traceback.print_exc()


def _run_training_loop(
    *,
    samples: List[Sample],
    playbook_path: Path,
    epochs: int,
    runtime: Stage3ExpertRuntime,
    adapter: OfflineAdapter,
    test_samples: Optional[List[Sample]],
    test_every: int,
    run_initial_test: bool,
    train_shuffle_seed: int,
    results_dir: Path,
    debug_roles: bool,
    resume: bool,
) -> None:
    environment = Stage3TrainingEnvironment(runtime)
    total_samples = len(samples)
    all_results = []

    trained_samples = 0
    if resume:
        state = _load_resume_state(
            results_dir=results_dir,
            playbook_path=playbook_path,
            total_samples=total_samples,
            epochs=epochs,
            train_shuffle_seed=train_shuffle_seed,
            test_every=test_every,
            run_initial_test=run_initial_test,
        )
        trained_samples = int(state.get("trained_samples_seen") or 0)
        total_training_steps = total_samples * epochs
        print(
            f"Resuming from checkpoint: completed {trained_samples}/{total_training_steps} training samples."
        )

    train_results_path, test_results_path = _prepare_results_dir(
        results_dir,
        resume=resume,
    )

    if not resume:
        _persist_training_checkpoint(
            playbook=adapter.playbook,
            playbook_path=playbook_path,
            results_dir=results_dir,
            total_samples=total_samples,
            epochs=epochs,
            train_shuffle_seed=train_shuffle_seed,
            test_every=test_every,
            run_initial_test=run_initial_test,
            trained_samples_seen=0,
            epoch=0,
            sample_index=None,
            status="initialized",
        )

    total_training_steps = total_samples * epochs
    if trained_samples >= total_training_steps:
        print("Checkpoint already covers the full run; nothing left to do.")
        return

    try:
        if run_initial_test and test_every > 0 and trained_samples == 0:
            evaluate_generator_on_test_set(
                adapter=adapter,
                environment=environment,
                runtime=runtime,
                test_samples=test_samples or [],
                label="initial",
                test_results_path=test_results_path,
            )

        for epoch in range(1, epochs + 1):
            epoch_seed = train_shuffle_seed + epoch - 1
            epoch_completed = _epoch_resume_offset(
                total_samples=total_samples,
                trained_samples_seen=trained_samples,
                epoch=epoch,
            )
            if epoch_completed >= total_samples:
                print(f"Epoch {epoch}/{epochs} already completed; skipping.")
                continue

            print(f"Epoch {epoch}/{epochs} (shuffled order seed={epoch_seed})")
            epoch_correct = 0.0
            epoch_processed = 0

            for step_position, (idx, sample) in enumerate(
                _shuffled_epoch_samples(samples, epoch=epoch, shuffle_seed=train_shuffle_seed),
                start=1,
            ):
                if step_position <= epoch_completed:
                    continue

                print(f"Processing sample {idx}/{total_samples} (epoch {epoch}/{epochs})")
                runtime.reset_call_log()
                _set_runtime_sample_genes(runtime, sample)
                try:
                    results = adapter.run([sample], environment, epochs=1)
                except Exception as exc:
                    _persist_training_checkpoint(
                        playbook=adapter.playbook,
                        playbook_path=playbook_path,
                        results_dir=results_dir,
                        total_samples=total_samples,
                        epochs=epochs,
                        train_shuffle_seed=train_shuffle_seed,
                        test_every=test_every,
                        run_initial_test=run_initial_test,
                        trained_samples_seen=trained_samples,
                        epoch=max(epoch - 1, 0),
                        sample_index=None,
                        status="failed",
                    )
                    _print_sample_failure_debug(sample, runtime, exc)
                    raise RuntimeError(
                        f"Sample {idx} failed during training: {exc}"
                    ) from exc
                finally:
                    runtime.clear_active_ranked_genes()

                for result in results:
                    all_results.append(result)
                    _save_train_result(
                        train_results_path,
                        result,
                        runtime,
                        epoch=epoch,
                        sample_index=idx,
                        training_samples_seen=trained_samples + 1,
                    )
                    metrics = result.environment_result.metrics or {}
                    epoch_correct += float(metrics.get("exact_match", 0.0))
                    epoch_processed += 1
                    if debug_roles:
                        _print_role_debug(result, runtime)
                    prune_playbook(adapter, limit=100, helpful_keep=70)
                    trained_samples += 1
                    _persist_training_checkpoint(
                        playbook=adapter.playbook,
                        playbook_path=playbook_path,
                        results_dir=results_dir,
                        total_samples=total_samples,
                        epochs=epochs,
                        train_shuffle_seed=train_shuffle_seed,
                        test_every=test_every,
                        run_initial_test=run_initial_test,
                        trained_samples_seen=trained_samples,
                        epoch=epoch,
                        sample_index=idx,
                        status="running",
                        announce=(trained_samples % 50 == 0),
                    )
                    if trained_samples % 50 == 0:
                        _save_playbook_checkpoint(
                            adapter.playbook,
                            playbook_path,
                            training_samples_seen=trained_samples,
                        )
                    if test_every > 0 and trained_samples % test_every == 0:
                        evaluate_generator_on_test_set(
                            adapter=adapter,
                            environment=environment,
                            runtime=runtime,
                            test_samples=test_samples or [],
                            label=f"after {trained_samples} training samples",
                            test_results_path=test_results_path,
                        )

            if epoch_processed:
                epoch_accuracy = epoch_correct / epoch_processed * 100
                print(
                    f"Epoch {epoch} accuracy: {epoch_correct:.0f}/{epoch_processed} "
                    f"({epoch_accuracy:.1f}%)"
                )
            else:
                print(f"Epoch {epoch} accuracy: no samples processed.")

            _save_train_epoch_summary(
                train_results_path,
                epoch=epoch,
                processed=epoch_processed,
                exact_match_total=epoch_correct,
                training_samples_seen=trained_samples,
            )

            adapter.playbook.save_to_file(str(playbook_path))

        _save_train_run_summary(
            train_results_path,
            epochs_completed=epochs,
            training_samples_seen=trained_samples,
            playbook=adapter.playbook,
        )
        _persist_training_checkpoint(
            playbook=adapter.playbook,
            playbook_path=playbook_path,
            results_dir=results_dir,
            total_samples=total_samples,
            epochs=epochs,
            train_shuffle_seed=train_shuffle_seed,
            test_every=test_every,
            run_initial_test=run_initial_test,
            trained_samples_seen=trained_samples,
            epoch=epochs,
            sample_index=None,
            status="completed",
        )
    finally:
        summarize_run(all_results, adapter.playbook, playbook_path)


def run_training(
    samples: List[Sample],
    playbook_path: Path,
    epochs: int,
    model: str,
    agent_model: str,
    runtime: Stage3ExpertRuntime,
    test_samples: Optional[List[Sample]] = None,
    test_every: int = 10,
    run_initial_test: bool = False,
    train_shuffle_seed: int = 17,
    results_dir: Optional[Path] = None,
    debug_roles: bool = False,
    resume: bool = False,
) -> None:
    if playbook_path.exists():
        playbook = Playbook.load_from_file(str(playbook_path))
        print(f"Loaded existing playbook ({len(playbook.bullets())} bullets).")
    else:
        playbook = Playbook()
        print("Starting from an empty playbook.")

    adapter = build_adapter(runtime=runtime, agent_model=agent_model, playbook=playbook, reflector_model=model)
    _run_training_loop(
        samples=samples,
        playbook_path=playbook_path,
        epochs=epochs,
        runtime=runtime,
        adapter=adapter,
        test_samples=test_samples,
        test_every=test_every,
        run_initial_test=run_initial_test,
        train_shuffle_seed=train_shuffle_seed,
        results_dir=results_dir or _default_results_dir(playbook_path),
        debug_roles=debug_roles,
        resume=resume,
    )


def run_training_stepwise(
    samples: List[Sample],
    playbook_path: Path,
    epochs: int,
    model: str,
    agent_model: str,
    runtime: Stage3ExpertRuntime,
    test_samples: Optional[List[Sample]] = None,
    test_every: int = 10,
    run_initial_test: bool = False,
    train_shuffle_seed: int = 17,
    results_dir: Optional[Path] = None,
    debug_roles: bool = False,
    resume: bool = False,
) -> None:
    if playbook_path.exists():
        playbook = Playbook.load_from_file(str(playbook_path))
        print(f"Loaded existing playbook ({len(playbook.bullets())} bullets).")
    else:
        playbook = Playbook()
        print("Starting from an empty playbook.")

    adapter = build_adapter(runtime=runtime, agent_model=agent_model, playbook=playbook, reflector_model=model)
    _run_training_loop(
        samples=samples,
        playbook_path=playbook_path,
        epochs=epochs,
        runtime=runtime,
        adapter=adapter,
        test_samples=test_samples,
        test_every=test_every,
        run_initial_test=run_initial_test,
        train_shuffle_seed=train_shuffle_seed,
        results_dir=results_dir or _default_results_dir(playbook_path),
        debug_roles=debug_roles,
        resume=resume,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=Path,
        default=Stage3Paths.discover().stage3_exports_root / "all_grouped_train.jsonl",
        help="Path to training samples (.json, .jsonl, or .csv).",
    )
    parser.add_argument(
        "--playbook",
        type=Path,
        default=DEFAULT_PLAYBOOK_PATH,
        help=f"Playbook file to load and update (default: {DEFAULT_PLAYBOOK_PATH.name}).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs to run over the samples.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.1",
        help="LiteLLM model identifier used for reflector/curator.",
    )
    parser.add_argument(
        "--agent-model",
        type=str,
        default="gpt-5.1",
        help="Model used by the LangChain tool agent for the Generator.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate sample loading without calling the LLM.",
    )
    parser.add_argument(
        "--debug-roles",
        action="store_true",
        help="Print detailed generator, reflector, and curator outputs for each sample.",
    )
    parser.add_argument(
        "--show-internal",
        "--show-internals",
        dest="show_internal",
        action="store_true",
        help="Run samples stepwise and print internal role outputs as they are produced.",
    )
    parser.add_argument(
        "--expert-root",
        type=Path,
        default=None,
        help="Optional override for the stage-2 expert outputs root.",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path(Stage3RuntimeConfig().base_model),
        help="Path or Hugging Face identifier for the stage-2 experts' base model.",
    )
    parser.add_argument(
        "--expert-cache-size",
        type=int,
        default=2,
        help="Maximum number of expert adapters to keep loaded at once.",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization for local expert model loading. This is the default.",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Enable 4-bit quantization for local expert model loading.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of samples loaded for smoke tests or short runs.",
    )
    parser.add_argument(
        "--random-train-test-split",
        action="store_true",
        help="Shuffle the input dataset and create a train/test split before training.",
    )
    parser.add_argument(
        "--test-samples",
        type=Path,
        default=None,
        help="Optional held-out test samples (.json, .jsonl, or .csv). Cannot be used with --random-train-test-split.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train fraction to use when --random-train-test-split is enabled.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=17,
        help="Random seed used for the optional train/test split.",
    )
    parser.add_argument(
        "--split-output-dir",
        type=Path,
        default=None,
        help="Optional directory where the generated train/test JSONL files should be written.",
    )
    parser.add_argument(
        "--test-every",
        type=int,
        default=10,
        help="Evaluate the generator on the held-out test split after every N training samples. Use 0 to disable.",
    )
    parser.add_argument(
        "--run-initial-test",
        action="store_true",
        help="Also run a held-out test sweep before the first training sample. Disabled by default to avoid upfront cost.",
    )
    parser.add_argument(
        "--train-shuffle-seed",
        type=int,
        default=17,
        help="Base random seed used to reshuffle training sample order every epoch. Epoch e uses seed (train_shuffle_seed + e - 1).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory for train_results.jsonl and test_results.jsonl. Defaults to <playbook_stem>_results beside the playbook.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run from the checkpoint state stored in --results-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if getattr(args, "show_internal", False):
        args.debug_roles = True
    if args.random_train_test_split and args.test_samples is not None:
        raise SystemExit("--test-samples cannot be combined with --random-train-test-split.")

    test_samples: List[Sample] = []
    train_source_path = args.samples
    test_source_path: Optional[Path] = None
    results_dir = args.results_dir or _default_results_dir(args.playbook)
    split_metadata: Dict[str, object]
    if args.random_train_test_split:
        raw_records = _load_raw_records(args.samples)
        train_records, test_records = _random_train_test_split(
            raw_records,
            train_ratio=args.train_ratio,
            seed=args.split_seed,
        )
        split_output_dir = args.split_output_dir or args.samples.parent / f"{args.samples.stem}_split_seed{args.split_seed}"
        train_path = split_output_dir / "train.jsonl"
        test_path = split_output_dir / "test.jsonl"
        _write_jsonl(train_path, train_records)
        _write_jsonl(test_path, test_records)
        print(
            f"Random split created with seed={args.split_seed}: "
            f"train={len(train_records)} test={len(test_records)}"
        )
        print(f"Saved train split to {train_path}")
        print(f"Saved test split to {test_path}")
        samples = _records_to_samples(train_records, max_samples=args.max_samples)
        test_samples = _records_to_samples(test_records)
        train_source_path = train_path
        test_source_path = test_path
        split_metadata = {
            "mode": "random_split",
            "input_samples_path": str(args.samples.resolve()),
            "split_output_dir": str(split_output_dir.resolve()),
            "train_ratio": args.train_ratio,
            "split_seed": args.split_seed,
            "generated_train_path": str(train_path.resolve()),
            "generated_test_path": str(test_path.resolve()),
            "generated_train_record_count": len(train_records),
            "generated_test_record_count": len(test_records),
        }
        print(f"Loaded {len(samples)} training samples from {train_path}.")
        print(f"Held out {len(test_samples)} samples for testing from {test_path}.")
    elif args.test_samples is not None:
        samples = parse_samples(args.samples, max_samples=args.max_samples)
        test_samples = parse_samples(args.test_samples)
        test_source_path = args.test_samples
        split_metadata = {
            "mode": "explicit_test_file",
            "train_samples_path": str(args.samples.resolve()),
            "test_samples_path": str(args.test_samples.resolve()),
        }
        print(f"Loaded {len(samples)} training samples from {args.samples}.")
        print(f"Loaded {len(test_samples)} held-out test samples from {args.test_samples}.")
    else:
        samples = parse_samples(args.samples, max_samples=args.max_samples)
        split_metadata = {
            "mode": "train_only",
            "train_samples_path": str(args.samples.resolve()),
        }
        print(f"Loaded {len(samples)} samples from {args.samples}.")

    if (not args.dry_run or args.results_dir is not None) and not args.resume:
        _write_run_manifest(
            results_dir=results_dir,
            playbook_path=args.playbook,
            train_source_path=train_source_path,
            test_source_path=test_source_path,
            samples=samples,
            test_samples=test_samples,
            epochs=args.epochs,
            model=args.model,
            agent_model=args.agent_model,
            test_every=args.test_every,
            run_initial_test=args.run_initial_test,
            train_shuffle_seed=args.train_shuffle_seed,
            max_samples=args.max_samples,
            split_metadata=split_metadata,
        )
        print(f"Saved run manifest to {results_dir / 'run_manifest.json'}")
    elif args.resume:
        print(f"Resuming existing run in {results_dir}")

    if args.dry_run:
        print("Dry run complete (training skipped).")
        if test_samples:
            print(f"Held-out test samples available: {len(test_samples)}")
        for sample in samples[:3]:
            preview = {key: value for key, value in asdict(sample).items() if key != "metadata"}
            print(f"  Preview: {preview}")
        return

    from stage3_ace_orchestrator.engine import Stage3ExpertRuntime

    runtime_config = Stage3RuntimeConfig(
        base_model=str(args.base_model),
        expert_cache_size=args.expert_cache_size,
        use_4bit=args.use_4bit and not args.no_4bit,
        verbose=args.show_internal or args.debug_roles,
    )
    runtime = Stage3ExpertRuntime(
        Stage3Paths.discover(),
        runtime_config,
        outputs_root=args.expert_root,
    )

    try:
        if args.show_internal:
            run_training_stepwise(
                samples,
                args.playbook,
                epochs=args.epochs,
                model=args.model,
                agent_model=args.agent_model,
                runtime=runtime,
                test_samples=test_samples,
                test_every=args.test_every,
                run_initial_test=args.run_initial_test,
                train_shuffle_seed=args.train_shuffle_seed,
                results_dir=results_dir,
                debug_roles=args.debug_roles,
                resume=args.resume,
            )
        else:
            run_training(
                samples,
                args.playbook,
                epochs=args.epochs,
                model=args.model,
                agent_model=args.agent_model,
                runtime=runtime,
                test_samples=test_samples,
                test_every=args.test_every,
                run_initial_test=args.run_initial_test,
                train_shuffle_seed=args.train_shuffle_seed,
                results_dir=results_dir,
                debug_roles=args.debug_roles,
                resume=args.resume,
            )
    except Exception as exc:
        print(f"Training aborted: {exc}")
        print(
            "Ensure your orchestrator provider credentials are configured and the local expert model "
            "environment can load the stage-2 adapters. You can rerun with --resume to continue from the latest "
            "saved checkpoint, or use --dry-run to validate sample loading first."
        )


if __name__ == "__main__":
    main()