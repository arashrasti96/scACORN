from __future__ import annotations

import json
import re

from langchain.tools import tool
from pydantic import BaseModel, Field

from .engine import Stage3ExpertRuntime
from .requests import (
    NormalizedStage3Request,
    extract_ranked_gene_lists_from_question,
    extract_ranked_genes_for_profile,
    extract_ranked_genes_from_question,
)
from .registry import ExpertDescriptor


class ListExpertsInput(BaseModel):
    tissue_filter: str = Field(default="", description="Optional substring to filter experts by tissue/domain.")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum number of experts to list.")


class ExpertToolInput(BaseModel):
    question_text: str = Field(description="The biology question to ask this expert.")
    genes_csv: str = Field(
        default="",
        description="Comma- or space-separated ordered genes. Leave empty for multi-cell or multi-profile questions; when profile_id is used, the runtime supplies the canonical 200-gene list.",
    )
    profile_id: str = Field(
        default="",
        description="Cell/profile identifier such as A or B. Required for multi-cell or multi-profile questions; omit only for genuine single-cell queries.",
    )
    context: str = Field(default="", description="Optional grounded context string.")
    task_family: str = Field(default="freeform_qa", description="Stage-3 task family such as annotation_direct or tissue_inference_direct.")
    task_type: str = Field(default="", description="Optional explicit stage-2 task type override.")
    candidate_labels_csv: str = Field(default="", description="Optional comma-separated candidate labels.")


def _sanitize_tool_name(name: str) -> str:
    return "query_" + "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def _split_input_genes(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(token for token in re.split(r"[\s,]+", value.strip()) if token)


def _best_ranked_genes(payload: ExpertToolInput) -> list[str]:
    if payload.profile_id:
        profile_candidates = [
            extract_ranked_genes_for_profile(payload.question_text, payload.profile_id, limit=200),
            extract_ranked_genes_for_profile(payload.context, payload.profile_id, limit=200),
        ]
        best_profile_match = max(profile_candidates, key=len, default=[])
        if best_profile_match:
            return best_profile_match

    candidates = [
        extract_ranked_genes_from_question(payload.question_text, limit=200),
        extract_ranked_genes_from_question(payload.context, limit=200),
        extract_ranked_genes_from_question("Ordered genes: " + payload.genes_csv, limit=200),
    ]
    return max(candidates, key=len, default=[])


def _tool_request_from_input(runtime: Stage3ExpertRuntime, payload: ExpertToolInput) -> NormalizedStage3Request:
    question_profile_lists = extract_ranked_gene_lists_from_question(payload.question_text, limit=200)
    context_profile_lists = extract_ranked_gene_lists_from_question(payload.context, limit=200)
    active_profile_gene_lists = runtime.get_active_profile_gene_lists()
    active_ranked_genes = runtime.get_active_ranked_genes()
    if not payload.genes_csv.strip() and not payload.profile_id.strip():
        if len(question_profile_lists) > 1 or len(context_profile_lists) > 1:
            raise ValueError(
                "This question contains multiple cell/profile gene lists. Pass profile_id or genes_csv for the specific cell before calling an expert."
            )

    genes = _best_ranked_genes(payload)
    if active_profile_gene_lists and payload.profile_id.strip():
        canonical_genes = runtime.get_active_profile_genes(payload.profile_id)
        if canonical_genes is None:
            available_profiles = ", ".join(runtime.get_active_profile_ids()) or "none"
            raise ValueError(
                f"Could not resolve canonical genes for profile_id='{payload.profile_id}'. Available profile IDs: {available_profiles}."
            )
        genes = list(canonical_genes)
    elif not active_profile_gene_lists and active_ranked_genes:
        genes = list(active_ranked_genes)
    if payload.profile_id and not genes:
        raise ValueError(
            f"Could not find genes for profile_id='{payload.profile_id}'. Pass genes_csv explicitly or use a profile ID present in the question."
        )

    base_request = NormalizedStage3Request.from_freeform_query(
        question_text=payload.question_text,
        genes=genes,
        context=payload.context,
        task_family=payload.task_family,
    )
    return base_request.with_tool_overrides(
        candidate_labels_csv=payload.candidate_labels_csv,
        task_type=payload.task_type or None,
    )


def _verify_and_lock_active_profile_genes(
    runtime: Stage3ExpertRuntime,
    payload: ExpertToolInput,
    request: NormalizedStage3Request,
) -> NormalizedStage3Request:
    active_profile_gene_lists = runtime.get_active_profile_gene_lists()
    if not active_profile_gene_lists:
        return request

    profile_id = str(payload.profile_id or "").strip()
    if not profile_id:
        raise RuntimeError(
            "Active sample contains multiple cell/profile gene lists. Expert calls must pass profile_id so the "
            "canonical 200-gene list can be selected and verified."
        )
    if payload.genes_csv.strip():
        raise RuntimeError(
            f"Multi-cell expert call for profile_id='{profile_id}' provided genes_csv. This is invalid: multi-cell calls "
            "must pass profile_id only so the runtime can inject the canonical 200-gene list."
        )

    canonical_genes = runtime.get_active_profile_genes(profile_id)
    if canonical_genes is None:
        available_profiles = ", ".join(runtime.get_active_profile_ids()) or "none"
        raise RuntimeError(
            f"No canonical gene list is available for profile_id='{profile_id}'. Available profile IDs: {available_profiles}."
        )
    if len(canonical_genes) != runtime.config.top_genes:
        raise RuntimeError(
            f"Canonical gene list for profile_id='{profile_id}' has {len(canonical_genes)} genes; "
            f"expected {runtime.config.top_genes}. Stopping instead of issuing an incompatible stage-2 prompt."
        )

    question_profile_genes = tuple(extract_ranked_genes_for_profile(payload.question_text, profile_id, limit=runtime.config.top_genes))
    context_profile_genes = tuple(extract_ranked_genes_for_profile(payload.context, profile_id, limit=runtime.config.top_genes))

    verification_sources: list[str] = []
    verification_warnings: list[dict[str, object]] = []
    if question_profile_genes:
        verification_sources.append("question_text")
        if question_profile_genes != canonical_genes:
            verification_warnings.append(
                {
                    "source": "question_text",
                    "status": "mismatch_ignored",
                    "profile_id": profile_id,
                    "parsed_gene_count": len(question_profile_genes),
                    "canonical_gene_count": len(canonical_genes),
                }
            )
    if context_profile_genes:
        verification_sources.append("context")
        if context_profile_genes != canonical_genes:
            verification_warnings.append(
                {
                    "source": "context",
                    "status": "mismatch_ignored",
                    "profile_id": profile_id,
                    "parsed_gene_count": len(context_profile_genes),
                    "canonical_gene_count": len(canonical_genes),
                }
            )

    return (
        request.with_tool_overrides(genes_csv=", ".join(canonical_genes))
        .with_metadata_updates(
            tool_profile_id=profile_id,
            canonical_gene_verifier={
                "status": "verified_with_warnings" if verification_warnings else "verified",
                "profile_id": profile_id,
                "canonical_gene_count": len(canonical_genes),
                "verified_against": verification_sources or ["active_profile_gene_lists"],
                "warnings": verification_warnings,
            },
        )
    )


def _recoverable_multi_profile_tool_error(runtime: Stage3ExpertRuntime, error: Exception) -> str | None:
    message = str(error)
    recoverable_markers = (
        "This question contains multiple cell/profile gene lists. Pass profile_id or genes_csv",
        "Active sample contains multiple cell/profile gene lists. Expert calls must pass profile_id",
        "Multi-cell expert call for profile_id=",
        "Could not resolve canonical genes for profile_id=",
    )
    if not any(marker in message for marker in recoverable_markers):
        return None

    available_profile_ids = runtime.get_active_profile_ids()
    return json.dumps(
        {
            "tool_error": {
                "recoverable": True,
                "kind": "multi_profile_call_contract",
                "message": message,
                "available_profile_ids": available_profile_ids,
                "retry_instruction": (
                    "Retry this expert call with exactly one valid profile_id from available_profile_ids "
                    "and leave genes_csv empty."
                ),
            }
        },
        ensure_ascii=False,
    )


def build_stage3_tools(runtime: Stage3ExpertRuntime) -> list:
    tools = [build_list_experts_tool(runtime)]
    for expert in runtime.registry.values():
        tools.append(build_expert_tool(runtime, expert))
    return tools


def build_list_experts_tool(runtime: Stage3ExpertRuntime):
    def _list_stage3_experts(tissue_filter: str = "", limit: int = 20) -> str:
        return runtime.format_expert_catalog(tissue_filter=tissue_filter or None, limit=limit)

    _list_stage3_experts.__name__ = "list_stage3_experts"
    _list_stage3_experts.__doc__ = "List available local stage-2 expert adapters and their tissue/domain descriptions."
    return tool(args_schema=ListExpertsInput)(_list_stage3_experts)


def build_expert_tool(runtime: Stage3ExpertRuntime, expert: ExpertDescriptor):
    def _run_expert(
        question_text: str,
        genes_csv: str = "",
        profile_id: str = "",
        context: str = "",
        task_family: str = "freeform_qa",
        task_type: str = "",
        candidate_labels_csv: str = "",
    ) -> str:
        payload = ExpertToolInput(
            question_text=question_text,
            genes_csv=genes_csv,
            profile_id=profile_id,
            context=context,
            task_family=task_family,
            task_type=task_type,
            candidate_labels_csv=candidate_labels_csv,
        )
        try:
            request = _tool_request_from_input(runtime, payload)
            request = _verify_and_lock_active_profile_genes(runtime, payload, request)
        except Exception as exc:
            recoverable_error = _recoverable_multi_profile_tool_error(runtime, exc)
            if recoverable_error is not None:
                return recoverable_error
            raise
        result = runtime.run_expert(expert.name, request)
        public_payload = result.to_public_tool_payload()
        if public_payload.get("answer") is None:
            raise RuntimeError(
                f"Expert '{expert.name}' returned no usable answer payload for task_type='{request.task_type}'."
            )
        return json.dumps(public_payload, ensure_ascii=False)

    _run_expert.__name__ = _sanitize_tool_name(expert.name)
    _run_expert.__doc__ = (
        f"Query the local expert '{expert.name}'. {expert.description} "
        "Use this when the question may benefit from this expert's tissue/domain specialization. "
        "For multi-cell questions, always pass profile_id for the specific cell and leave genes_csv empty; the runtime will inject the canonical 200-gene list. "
        "The expert is always prompted with the trained 200-gene C2S template. "
        "The returned payload may include final_label, supporting_genes/evidence, negative_markers, context, confidence, and raw_response. "
        "Positive marker fields from the model are intentionally omitted."
    )
    return tool(args_schema=ExpertToolInput)(_run_expert)
