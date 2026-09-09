from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import itertools
import re


_REQUEST_COUNTER = itertools.count(1)
_MULTI_PROFILE_PATTERN = re.compile(
    r"(?:here\s+is\s+|given\s+)?the\s+ranked\s+gene\s+list(?:\s+for)?\s+(?:cell|profile)\s+([A-Za-z0-9_-]+)\s*:\s*(.+?)(?=(?:here\s+is\s+|given\s+)?the\s+ranked\s+gene\s+list(?:\s+for)?\s+(?:cell|profile)\s+[A-Za-z0-9_-]+\s*:|$)",
    flags=re.IGNORECASE | re.DOTALL,
)


def _split_csv_or_space(value: str | None) -> list[str]:
    if not value:
        return []
    tokens = re.split(r"[\s,]+", value.strip())
    return [token for token in tokens if token]


def _tokenize_genes(blob: str | None, *, limit: int = 200) -> list[str]:
    if not blob:
        return []
    genes = [
        token.strip().strip(".;")
        for token in re.split(r"[\s,]+", blob.strip())
        if token.strip().strip(".;")
    ]
    return genes[:limit]


def _normalize_profile_id(value: str | None) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def extract_ranked_gene_lists_from_question(text: str | None, *, limit: int = 200) -> dict[str, list[str]]:
    if not text:
        return {}

    extracted: dict[str, list[str]] = {}
    for match in _MULTI_PROFILE_PATTERN.finditer(text):
        profile_id = str(match.group(1) or "").strip()
        genes = _tokenize_genes(match.group(2), limit=limit)
        if profile_id and genes:
            extracted[profile_id] = genes
    return extracted


def extract_ranked_genes_for_profile(
    text: str | None,
    profile_id: str | None,
    *,
    limit: int = 200,
) -> list[str]:
    if not text or not profile_id:
        return []

    target = _normalize_profile_id(profile_id)
    for observed_profile_id, genes in extract_ranked_gene_lists_from_question(text, limit=limit).items():
        if _normalize_profile_id(observed_profile_id) == target:
            return genes
    return []


def extract_ranked_genes_from_question(text: str | None, *, limit: int = 200) -> list[str]:
    if not text:
        return []

    multi_profile_genes = extract_ranked_gene_lists_from_question(text, limit=limit)
    if multi_profile_genes:
        return next(iter(multi_profile_genes.values()))[:limit]

    patterns = [
        r"Here is the ranked gene list:\s*(.+?)(?:\.\s+(?:I|Based|Can|Could|Please|What)\s|\?\s|\n|$)",
        r"Ordered genes:\s*(.+?)(?:\n|$)",
        r"Cell sentence:\s*(.+?)(?:\nQuestion:|\nThe cell type|\n|$)",
        r"Given this cell gene list:\s*(.+?)(?:\.\s+I\s|\.\s+Could\s|\.\s+Please\s|\?\s|\n|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        genes = _tokenize_genes(match.group(1), limit=limit)
        if genes:
            return genes
    return []


def _default_task_type(task_family: str) -> str:
    if task_family == "annotation_direct":
        return "cell_type"
    if task_family == "marker_support_direct":
        return "marker_evidence"
    if task_family == "tissue_inference_direct":
        return "tissue_inference"
    if task_family == "differential_diagnosis_direct":
        return "differential_diagnosis"
    if task_family == "cross_expert_annotation":
        return "cell_type"
    if task_family == "ood_abstain":
        return "cell_type"
    return "freeform_qa"


STAGE2_TASK_TYPES = {
    "cell_type",
    "marker_evidence",
    "differential_diagnosis",
    "state_classification",
    "tissue_inference",
    "perturbation_inference",
    "cluster_caption",
    "freeform_qa",
    "domain_replay",
}


def normalize_task_type(task_type: str | None, task_family: str = "freeform_qa") -> str:
    candidate = str(task_type or "").strip()
    if candidate in STAGE2_TASK_TYPES:
        return candidate
    if candidate:
        mapped = _default_task_type(candidate)
        if mapped != "freeform_qa" or candidate == "freeform_qa":
            return mapped
    return _default_task_type(task_family)


@dataclass(frozen=True)
class NormalizedStage3Request:
    sample_id: str
    task_family: str
    task_type: str
    question_text: str
    genes: tuple[str, ...]
    context: str = ""
    answer_schema_name: str | None = None
    required_output_fields: tuple[str, ...] = ()
    primary_target: str | None = None
    candidate_labels: tuple[str, ...] = ()
    candidate_experts: tuple[str, ...] = ()
    oracle_experts: tuple[str, ...] = ()
    ground_truth: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_stage3_row(cls, row: dict[str, Any]) -> "NormalizedStage3Request":
        metadata = dict(row.get("metadata") or {})
        task_context = dict((row.get("ground_truth") or {}).get("task_context") or {})
        question_text = str(row.get("question_text") or row.get("question") or "Answer the grounded biology query.")
        extracted_genes = extract_ranked_genes_from_question(question_text)
        if not extracted_genes:
            extracted_genes = [str(gene) for gene in row.get("genes", []) if str(gene)]
        candidate_labels = tuple(
            metadata.get("candidate_labels")
            or task_context.get("candidate_labels")
            or []
        )
        return cls(
            sample_id=str(row.get("sample_id") or row.get("id") or f"stage3_{next(_REQUEST_COUNTER)}"),
            task_family=str(row.get("task_family") or "freeform_qa"),
            task_type=_default_task_type(str(row.get("task_family") or "freeform_qa")),
            question_text=question_text,
            genes=tuple(extracted_genes),
            context=str(row.get("context") or ""),
            answer_schema_name=row.get("answer_schema_name"),
            required_output_fields=tuple(str(field_name) for field_name in row.get("required_output_fields") or []),
            primary_target=row.get("primary_target"),
            candidate_labels=candidate_labels,
            candidate_experts=tuple(str(name) for name in row.get("candidate_experts") or []),
            oracle_experts=tuple(str(name) for name in row.get("oracle_experts") or []),
            ground_truth=dict(row.get("ground_truth") or {}),
            metadata=metadata,
        )

    @classmethod
    def from_freeform_query(
        cls,
        *,
        question_text: str,
        genes: list[str] | tuple[str, ...] | None = None,
        context: str = "",
        task_family: str = "freeform_qa",
        task_type: str | None = None,
        candidate_labels: list[str] | tuple[str, ...] | None = None,
    ) -> "NormalizedStage3Request":
        return cls(
            sample_id=f"adhoc_{next(_REQUEST_COUNTER)}",
            task_family=task_family,
            task_type=normalize_task_type(task_type, task_family),
            question_text=question_text,
            genes=tuple(genes or ()),
            context=context,
            candidate_labels=tuple(candidate_labels or ()),
        )

    def with_tool_overrides(
        self,
        *,
        question_text: str | None = None,
        genes_csv: str | None = None,
        context: str | None = None,
        task_family: str | None = None,
        task_type: str | None = None,
        candidate_labels_csv: str | None = None,
    ) -> "NormalizedStage3Request":
        genes = tuple(_split_csv_or_space(genes_csv)) if genes_csv is not None else self.genes
        candidate_labels = (
            tuple(_split_csv_or_space(candidate_labels_csv))
            if candidate_labels_csv is not None
            else self.candidate_labels
        )
        next_task_family = task_family or self.task_family
        next_task_type = normalize_task_type(task_type or self.task_type, next_task_family)
        return NormalizedStage3Request(
            sample_id=self.sample_id,
            task_family=next_task_family,
            task_type=next_task_type,
            question_text=question_text or self.question_text,
            genes=genes,
            context=self.context if context is None else context,
            answer_schema_name=self.answer_schema_name,
            required_output_fields=self.required_output_fields,
            primary_target=self.primary_target,
            candidate_labels=candidate_labels,
            candidate_experts=self.candidate_experts,
            oracle_experts=self.oracle_experts,
            ground_truth=self.ground_truth,
            metadata=self.metadata,
        )

    def with_metadata_updates(self, **updates: Any) -> "NormalizedStage3Request":
        metadata = dict(self.metadata)
        metadata.update({key: value for key, value in updates.items() if value is not None})
        return NormalizedStage3Request(
            sample_id=self.sample_id,
            task_family=self.task_family,
            task_type=self.task_type,
            question_text=self.question_text,
            genes=self.genes,
            context=self.context,
            answer_schema_name=self.answer_schema_name,
            required_output_fields=self.required_output_fields,
            primary_target=self.primary_target,
            candidate_labels=self.candidate_labels,
            candidate_experts=self.candidate_experts,
            oracle_experts=self.oracle_experts,
            ground_truth=self.ground_truth,
            metadata=metadata,
        )

    @property
    def ace_question(self) -> str:
        return self.question_text

    @property
    def ace_context(self) -> str:
        lines = []
        lines.append(f"Task family: {self.task_family}")
        lines.append(f"Task type: {self.task_type}")
        if self.genes:
            lines.append("Ordered genes: " + ", ".join(self.genes))
        profile_gene_lists = self.metadata.get("profile_gene_lists") or {}
        if isinstance(profile_gene_lists, dict) and profile_gene_lists:
            profile_ids = [str(profile_id).strip() for profile_id in profile_gene_lists if str(profile_id).strip()]
            if profile_ids:
                lines.append("Available profile IDs: " + ", ".join(profile_ids))
                lines.append(
                    "Multi-profile tool contract: every expert tool call must include exactly one profile_id from the "
                    "available profile IDs. Do not call an expert on a multi-profile sample without profile_id."
                )
        if self.context:
            lines.append("Context: " + self.context)
        if self.candidate_labels:
            lines.append("Candidate labels: " + ", ".join(self.candidate_labels))
        negative_markers = self.ground_truth.get("negative_markers") or self.metadata.get("available_negative_markers")
        if negative_markers:
            if isinstance(negative_markers, (list, tuple)):
                lines.append("Negative markers for excluding alternatives: " + ", ".join(str(marker) for marker in negative_markers))
            else:
                lines.append("Negative markers for excluding alternatives: " + str(negative_markers))
        if self.answer_schema_name:
            lines.append("Answer schema: " + self.answer_schema_name)
        if self.required_output_fields:
            lines.append("Required output fields: " + ", ".join(self.required_output_fields))
            lines.append(
                "Final answer contract: final_answer should be a grounded free-form answer that explicitly includes these elements; "
                "use final_label from candidate_labels unless abstaining, cite supporting_genes from the ordered genes, "
                "cite negative_markers that rule out alternatives, mention expert_models_used explicitly, and provide a concise rationale."
            )
        return "\n".join(lines)

    def to_stage2_sample(self) -> dict[str, Any]:
        sample = {
            "sample_id": self.sample_id,
            "task_type": self.task_type,
            "genes": list(self.genes),
            "question_text": self.question_text,
            "context": self.context,
            "metadata": dict(self.metadata),
        }
        if self.candidate_labels:
            sample["candidate_labels"] = list(self.candidate_labels)
        if self.answer_schema_name:
            sample["answer_schema_name"] = self.answer_schema_name
        if self.required_output_fields:
            sample["required_output_fields"] = list(self.required_output_fields)
        if self.primary_target:
            sample["primary_target"] = self.primary_target
        return sample
