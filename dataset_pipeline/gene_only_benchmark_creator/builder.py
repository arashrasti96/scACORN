from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence


FAIRNESS_RULES = [
    "Do not ask for donor, assay, disease, sex, developmental stage, or exact tissue metadata unless the genes themselves make that context biologically inferable.",
    "The visible model input must expose only gene lists.",
    "Use hidden grounding only to write a more natural and biologically grounded answer.",
    "Do not mention hidden metadata in the question.",
    "Keep the answer factual, concise, and evidence-based.",
]

QUESTION_GUARD_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bwhat tissue\b",
        r"\bwhich tissue\b",
        r"\bwhat assay\b",
        r"\bwhich assay\b",
        r"\bwhat disease\b",
        r"\bwhich disease\b",
        r"\bwhat sex\b",
        r"\bwhich sex\b",
        r"\bwhich donor\b",
        r"\bwhat donor\b",
        r"\bdevelopment(?:al)? stage\b",
    ]
]

ALLOWED_TISSUE_PROGRAM_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\btissue-associated biological program\b",
        r"\btissue-associated program\b",
        r"\btissue-compatible biological program\b",
        r"\btissue-compatible interpretation\b",
        r"\bcompartment-compatible biological program\b",
        r"\bcompartment-compatible interpretation\b",
    ]
]

GENE_TOKEN_EXCEPTIONS = {
    "RNA",
    "DNA",
    "JSON",
    "CELL",
    "TYPE",
    "MODEL",
    "GPT",
    "QC",
    "ECM",
    "MHC",
    "AP",
    "NF",
}

GENE_TOKEN_BLACKLIST_CHUNKS = {
    "associated",
    "cell",
    "cells",
    "compatible",
    "epithelial",
    "identity",
    "immune",
    "like",
    "negative",
    "phenotype",
    "positive",
    "program",
    "regulatory",
    "secretory",
    "secreting",
    "state",
}

IDENTITY_ALIAS_STOPWORDS = {
    "associated",
    "cell",
    "cells",
    "compatible",
    "dominant",
    "identity",
    "likely",
    "like",
    "program",
    "state",
}


@dataclass(frozen=True)
class QuestionSpec:
    name: str
    level: str
    prompt_instruction: str
    eligibility: Callable[["SourceRecord"], bool] | None = None


QUESTION_SPECS = {
    "functional_program": QuestionSpec(
        name="functional_program",
        level="cell",
        prompt_instruction=(
            "Ask about the dominant biological program suggested by a single-cell gene profile. "
            "The answer should explain the program and cite informative genes."
        ),
    ),
    "evidence_gene_reasoning": QuestionSpec(
        name="evidence_gene_reasoning",
        level="cell",
        prompt_instruction=(
            "Ask which genes are most informative for interpreting the cell identity. "
            "The answer should prioritize discriminative markers over housekeeping genes."
        ),
        eligibility=lambda record: bool(record.evidence_genes or record.positive_markers),
    ),
    "negative_exclusion": QuestionSpec(
        name="negative_exclusion",
        level="cell",
        prompt_instruction=(
            "Ask which alternative identities are less supported and why. "
            "The answer should use absent or non-prominent negative markers when available."
        ),
        eligibility=lambda record: bool(record.negative_markers),
    ),
    "confidence_assessment": QuestionSpec(
        name="confidence_assessment",
        level="cell",
        prompt_instruction=(
            "Ask how confident we should be in the interpretation of the gene profile. "
            "The answer should calibrate certainty and mention whether conflicting lineage signals are present."
        ),
    ),
    "cell_state_description": QuestionSpec(
        name="cell_state_description",
        level="cell",
        prompt_instruction=(
            "Ask for a brief description of the likely cell state represented by the gene set."
        ),
    ),
    "tissue_compatible_interpretation": QuestionSpec(
        name="tissue_compatible_interpretation",
        level="cell",
        prompt_instruction=(
            "Ask what tissue-associated or compartment-compatible biological program is suggested by the genes, "
            "but only if that context is inferable from the markers themselves."
        ),
    ),
    "informative_marker_vs_housekeeping": QuestionSpec(
        name="informative_marker_vs_housekeeping",
        level="cell",
        prompt_instruction=(
            "Ask which genes are informative markers versus likely housekeeping or generic stress genes."
        ),
    ),
    "doublet_or_mixed_signal_detection": QuestionSpec(
        name="doublet_or_mixed_signal_detection",
        level="cell",
        prompt_instruction=(
            "Ask whether the profile suggests a coherent single-cell identity or a possible mixed-lineage or doublet signal."
        ),
    ),
    "cluster_consensus": QuestionSpec(
        name="cluster_consensus",
        level="cluster",
        prompt_instruction=(
            "Ask what shared biological program is consistently represented across the cells in a cluster."
        ),
    ),
    "cluster_heterogeneity": QuestionSpec(
        name="cluster_heterogeneity",
        level="cluster",
        prompt_instruction=(
            "Ask whether the cluster looks transcriptionally coherent or mixed. "
            "The answer should explicitly discuss coherence versus heterogeneity."
        ),
    ),
    "cluster_comparison": QuestionSpec(
        name="cluster_comparison",
        level="cluster",
        prompt_instruction=(
            "Ask for the main biological differences between two clusters."
        ),
    ),
}


def _default_question_types(levels: Sequence[str]) -> tuple[str, ...]:
    level_set = set(levels)
    return tuple(
        name for name, spec in QUESTION_SPECS.items() if spec.level in level_set
    )


@dataclass(frozen=True)
class BuildConfig:
    source_root: Path
    output_dir: Path
    k: int
    seed: int = 42
    splits: tuple[str, ...] = ("test",)
    levels: tuple[str, ...] = ("cell", "cluster")
    question_types: tuple[str, ...] = ()
    model_name: str = "gpt-4o-mini"
    cluster_size: int = 3
    max_genes: int = 80
    temperature: float = 0.4
    generation_retries: int = 5
    resume: bool = False
    api_key: str | None = None

    def resolved_question_types(self) -> tuple[str, ...]:
        return self.question_types or _default_question_types(self.levels)


@dataclass(frozen=True)
class SourceRecord:
    sample_id: str
    source_name: str
    task_type: str
    genes: tuple[str, ...]
    cell_type: str
    broad_cell_class: str | None
    compartment: str | None
    tissue: str | None
    evidence_genes: tuple[str, ...]
    positive_markers: tuple[str, ...]
    negative_markers: tuple[str, ...]
    exclusions: tuple[str, ...]
    context: str | None
    metadata: dict[str, Any]
    raw_answer: str

    @classmethod
    def from_json(cls, payload: dict[str, Any], source_name: str) -> "SourceRecord":
        answer_sections = parse_structured_answer(payload.get("answer", ""))
        metadata = dict(payload.get("metadata") or {})
        genes = tuple(str(gene) for gene in payload.get("genes") or [] if str(gene).strip())
        evidence_genes = tuple(
            str(gene)
            for gene in (payload.get("evidence_genes") or answer_sections.get("EVIDENCE") or [])
            if str(gene).strip()
        )
        negative_markers = tuple(
            str(gene)
            for gene in (
                payload.get("negative_markers")
                or metadata.get("negative_markers")
                or answer_sections.get("NEGATIVE_MARKERS")
                or []
            )
            if str(gene).strip()
        )
        positive_markers = tuple(
            str(gene)
            for gene in answer_sections.get("POSITIVE_MARKERS") or []
            if str(gene).strip()
        )
        exclusions = tuple(
            str(item)
            for item in answer_sections.get("EXCLUSIONS") or []
            if str(item).strip() and str(item).strip().lower() != "none"
        )
        cell_type = str(
            payload.get("cell_type")
            or metadata.get("cell_type")
            or answer_sections.get("FINAL")
            or answer_sections.get("LABEL")
            or "unknown"
        )
        return cls(
            sample_id=str(payload.get("sample_id") or ""),
            source_name=source_name,
            task_type=str(payload.get("task_type") or "cell_type"),
            genes=genes,
            cell_type=cell_type,
            broad_cell_class=_coerce_optional_str(
                payload.get("broad_cell_class") or metadata.get("broad_cell_class")
            ),
            compartment=_coerce_optional_str(
                payload.get("compartment") or metadata.get("compartment")
            ),
            tissue=_coerce_optional_str(payload.get("tissue") or metadata.get("tissue")),
            evidence_genes=evidence_genes,
            positive_markers=positive_markers,
            negative_markers=negative_markers,
            exclusions=exclusions,
            context=_coerce_optional_str(payload.get("context") or answer_sections.get("CONTEXT")),
            metadata=metadata,
            raw_answer=str(payload.get("answer") or ""),
        )

    def hidden_grounding(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_name": self.source_name,
            "task_type": self.task_type,
            "cell_type": self.cell_type,
            "broad_cell_class": self.broad_cell_class,
            "compartment": self.compartment,
            "tissue": self.tissue,
            "context": self.context,
            "evidence_genes": list(self.evidence_genes),
            "positive_markers": list(self.positive_markers),
            "negative_markers": list(self.negative_markers),
            "exclusions": list(self.exclusions),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class BuildExample:
    question_type: str
    level: str
    public_input: dict[str, Any]
    hidden_grounding: dict[str, Any]
    source_record_ids: tuple[str, ...]


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_structured_answer(answer_text: str) -> dict[str, Any]:
    sections: dict[str, Any] = {}
    for raw_line in answer_text.splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        key = key.strip().upper()
        value = value.strip()
        if not key:
            continue
        if key in {"POSITIVE_MARKERS", "NEGATIVE_MARKERS", "EVIDENCE", "EXCLUSIONS"}:
            sections[key] = [item.strip() for item in value.split(",") if item.strip()]
        else:
            sections[key] = value
    return sections


def extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model response did not contain a JSON object.")
    return json.loads(text[start : end + 1])


def stream_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(stream_jsonl(path)) if path.exists() else []


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


class OpenAIJsonGenerator:
    def __init__(self, *, api_key: str | None = None) -> None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set and no api_key argument was provided.")
        from openai import OpenAI

        self._client = OpenAI(api_key=key)

    def generate(
        self,
        *,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> dict[str, Any]:
        text = self._generate_with_responses(
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )
        if not text:
            text = self._generate_with_chat_completions(
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
            )
        return extract_json_object(text)

    def _generate_with_responses(
        self,
        *,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> str:
        try:
            request_kwargs = {
                "model": model_name,
                "input": [
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": user_prompt}],
                    },
                ],
            }
            request_kwargs.update(_temperature_request_kwargs(model_name=model_name, temperature=temperature))
            response = self._client.responses.create(
                **request_kwargs,
            )
        except Exception:
            return ""

        output_text = getattr(response, "output_text", "")
        if output_text:
            return output_text

        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(text)
        return "\n".join(chunks)

    def _generate_with_chat_completions(
        self,
        *,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> str:
        request_kwargs = {
            "model": model_name,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        request_kwargs.update(_temperature_request_kwargs(model_name=model_name, temperature=temperature))
        response = self._client.chat.completions.create(**request_kwargs)
        return response.choices[0].message.content or ""


def _temperature_request_kwargs(*, model_name: str, temperature: float) -> dict[str, Any]:
    normalized_name = model_name.strip().lower()
    if normalized_name.startswith("gpt-5"):
        return {}
    return {"temperature": temperature}


class BenchmarkDatasetBuilder:
    def __init__(
        self,
        config: BuildConfig,
        *,
        generator: OpenAIJsonGenerator | None = None,
    ) -> None:
        self.config = config
        self.rng = random.Random(config.seed)
        self.generator = generator or OpenAIJsonGenerator(api_key=config.api_key)

    def build(self) -> dict[str, Any]:
        self._validate_config()
        records = self._load_source_records()
        if not records:
            raise RuntimeError("No source records were loaded from the export directory.")
        identity_aliases = _build_identity_alias_vocabulary(records)

        sampler = BenchmarkSampler(
            records=records,
            rng=self.rng,
            cluster_size=self.config.cluster_size,
            max_genes=self.config.max_genes,
        )
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        public_path = self.config.output_dir / "benchmark.jsonl"
        private_path = self.config.output_dir / "benchmark_hidden.jsonl"
        manifest_path = self.config.output_dir / "manifest.json"

        existing_public = read_jsonl(public_path) if self.config.resume else []
        existing_private = read_jsonl(private_path) if self.config.resume else []
        existing_ids = {row["id"] for row in existing_public}

        new_public: list[dict[str, Any]] = []
        new_private: list[dict[str, Any]] = []

        for question_type in self.config.resolved_question_types():
            spec = QUESTION_SPECS[question_type]
            generated_for_type = 0
            example_index = 0
            while generated_for_type < self.config.k:
                example = sampler.sample(question_type=question_type, example_index=example_index)
                example_index += 1
                example_id = self._build_example_id(question_type, example.source_record_ids)
                if example_id in existing_ids:
                    generated_for_type += 1
                    continue
                response = self._generate_question_and_answer(
                    spec=spec,
                    example=example,
                    identity_aliases=identity_aliases,
                )
                public_row, private_row = self._build_output_rows(
                    example_id=example_id,
                    spec=spec,
                    example=example,
                    response=response,
                )
                new_public.append(public_row)
                new_private.append(private_row)
                existing_ids.add(example_id)
                generated_for_type += 1

        combined_public = existing_public + new_public
        combined_private = existing_private + new_private
        write_jsonl(public_path, combined_public)
        write_jsonl(private_path, combined_private)

        manifest = {
            "model_name": self.config.model_name,
            "k": self.config.k,
            "seed": self.config.seed,
            "levels": list(self.config.levels),
            "question_types": list(self.config.resolved_question_types()),
            "splits": list(self.config.splits),
            "cluster_size": self.config.cluster_size,
            "max_genes": self.config.max_genes,
            "resume": self.config.resume,
            "source_root": str(self.config.source_root),
            "output_dir": str(self.config.output_dir),
            "total_examples": len(combined_public),
            "question_type_counts": dict(Counter(row["question_type"] for row in combined_public)),
            "level_counts": dict(Counter(row["level"] for row in combined_public)),
            "new_example_count": len(new_public),
            "reused_example_count": len(combined_public) - len(new_public),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest

    def _validate_config(self) -> None:
        if self.config.k <= 0:
            raise ValueError("k must be a positive integer.")
        if self.config.cluster_size < 2:
            raise ValueError("cluster_size must be at least 2.")
        if self.config.max_genes <= 0:
            raise ValueError("max_genes must be a positive integer.")
        for level in self.config.levels:
            if level not in {"cell", "cluster"}:
                raise ValueError(f"Unsupported level '{level}'.")
        for question_type in self.config.resolved_question_types():
            if question_type not in QUESTION_SPECS:
                raise ValueError(f"Unsupported question type '{question_type}'.")
            spec = QUESTION_SPECS[question_type]
            if spec.level not in set(self.config.levels):
                raise ValueError(
                    f"Question type '{question_type}' targets level '{spec.level}', which is not enabled."
                )

    def _load_source_records(self) -> list[SourceRecord]:
        records: list[SourceRecord] = []
        for path in discover_source_files(self.config.source_root, self.config.splits):
            source_name = path.parent.name
            for payload in stream_jsonl(path):
                record = SourceRecord.from_json(payload, source_name=source_name)
                if len(record.genes) < 5:
                    continue
                records.append(record)
        return records

    def _generate_question_and_answer(
        self,
        *,
        spec: QuestionSpec,
        example: BuildExample,
        identity_aliases: set[str],
    ) -> dict[str, Any]:
        validate_hidden_grounding_against_public_input(
            visible_input=example.public_input,
            hidden_grounding=example.hidden_grounding,
        )
        system_prompt = build_system_prompt(spec)
        last_error: Exception | None = None
        for attempt_index in range(self.config.generation_retries):
            user_prompt = build_user_prompt(
                question_type=spec.name,
                level=spec.level,
                visible_input=example.public_input,
                hidden_grounding=example.hidden_grounding,
                retry_feedback=str(last_error) if last_error is not None else None,
            )
            response = self.generator.generate(
                model_name=self.config.model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.config.temperature,
            )
            try:
                validate_generated_response(
                    question_type=spec.name,
                    question=response.get("question", ""),
                    answer=response.get("answer", ""),
                    visible_input=example.public_input,
                    hidden_grounding=example.hidden_grounding,
                    known_identity_aliases=identity_aliases,
                )
                return response
            except ValueError as exc:
                last_error = exc
                if attempt_index == self.config.generation_retries - 1:
                    raise
        raise RuntimeError("Unreachable generation retry state.")

    def _build_output_rows(
        self,
        *,
        example_id: str,
        spec: QuestionSpec,
        example: BuildExample,
        response: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        public_row = {
            "id": example_id,
            "level": spec.level,
            "question_type": spec.name,
            "model_input": example.public_input,
            "question": str(response["question"]).strip(),
            "answer": str(response["answer"]).strip(),
        }
        private_row = {
            "id": example_id,
            "level": spec.level,
            "question_type": spec.name,
            "source_record_ids": list(example.source_record_ids),
            "model_input": example.public_input,
            "question": public_row["question"],
            "answer": public_row["answer"],
            "hidden_grounding": example.hidden_grounding,
            "quality_flags": response.get("quality_flags") or {},
        }
        return public_row, private_row

    @staticmethod
    def _build_example_id(question_type: str, source_record_ids: Sequence[str]) -> str:
        payload = question_type + "::" + "||".join(source_record_ids)
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
        return f"{question_type}-{digest}"


class BenchmarkSampler:
    def __init__(
        self,
        *,
        records: Sequence[SourceRecord],
        rng: random.Random,
        cluster_size: int,
        max_genes: int,
    ) -> None:
        self.records = list(records)
        self.rng = rng
        self.cluster_size = cluster_size
        self.max_genes = max_genes
        self.by_cell_type = defaultdict(list)
        for record in self.records:
            self.by_cell_type[record.cell_type].append(record)

        self.coherent_groups = [group for group in self.by_cell_type.values() if len(group) >= self.cluster_size]
        self.cell_type_pairs = self._build_cell_type_pairs()

    def sample(self, *, question_type: str, example_index: int) -> BuildExample:
        spec = QUESTION_SPECS[question_type]
        if spec.level == "cell":
            return self._sample_cell(question_type=question_type, spec=spec, example_index=example_index)
        return self._sample_cluster(question_type=question_type, example_index=example_index)

    def _sample_cell(
        self,
        *,
        question_type: str,
        spec: QuestionSpec,
        example_index: int,
    ) -> BuildExample:
        eligible = [record for record in self.records if spec.eligibility is None or spec.eligibility(record)]
        if not eligible:
            raise RuntimeError(f"No eligible cell records for question type '{question_type}'.")
        ordered = list(eligible)
        self.rng.shuffle(ordered)
        record = ordered[example_index % len(ordered)]
        public_input = {"genes": self._limit_genes(record.genes)}
        hidden_grounding = {"record": record.hidden_grounding()}
        return BuildExample(
            question_type=question_type,
            level="cell",
            public_input=public_input,
            hidden_grounding=hidden_grounding,
            source_record_ids=(record.sample_id,),
        )

    def _sample_cluster(self, *, question_type: str, example_index: int) -> BuildExample:
        if question_type == "cluster_comparison":
            return self._sample_cluster_comparison(example_index=example_index)
        if question_type == "cluster_heterogeneity" and example_index % 2 == 1:
            return self._sample_mixed_cluster(example_index=example_index)
        return self._sample_coherent_cluster(question_type=question_type, example_index=example_index)

    def _sample_coherent_cluster(self, *, question_type: str, example_index: int) -> BuildExample:
        if not self.coherent_groups:
            raise RuntimeError("No coherent clusters could be formed with the selected cluster size.")
        groups = list(self.coherent_groups)
        self.rng.shuffle(groups)
        group = groups[example_index % len(groups)]
        records = tuple(group[: self.cluster_size])
        public_input = {"cells": [self._limit_genes(record.genes) for record in records]}
        hidden_grounding = {
            "cluster_mode": "coherent",
            "records": [record.hidden_grounding() for record in records],
            "cluster_label": records[0].cell_type,
        }
        return BuildExample(
            question_type=question_type,
            level="cluster",
            public_input=public_input,
            hidden_grounding=hidden_grounding,
            source_record_ids=tuple(record.sample_id for record in records),
        )

    def _sample_mixed_cluster(self, *, example_index: int) -> BuildExample:
        if len(self.cell_type_pairs) < 1:
            raise RuntimeError("No distinct cell-type pairs were available for mixed-cluster sampling.")
        left_label, right_label = self.cell_type_pairs[example_index % len(self.cell_type_pairs)]
        left_records = tuple(self.by_cell_type[left_label][: max(1, self.cluster_size // 2)])
        right_records = tuple(self.by_cell_type[right_label][: self.cluster_size - len(left_records)])
        records = left_records + right_records
        public_input = {"cells": [self._limit_genes(record.genes) for record in records]}
        hidden_grounding = {
            "cluster_mode": "mixed",
            "records": [record.hidden_grounding() for record in records],
            "component_labels": [left_label, right_label],
        }
        return BuildExample(
            question_type="cluster_heterogeneity",
            level="cluster",
            public_input=public_input,
            hidden_grounding=hidden_grounding,
            source_record_ids=tuple(record.sample_id for record in records),
        )

    def _sample_cluster_comparison(self, *, example_index: int) -> BuildExample:
        if len(self.cell_type_pairs) < 1:
            raise RuntimeError("No distinct cluster pairs were available for comparison sampling.")
        left_label, right_label = self.cell_type_pairs[example_index % len(self.cell_type_pairs)]
        left_records = tuple(self.by_cell_type[left_label][: self.cluster_size])
        right_records = tuple(self.by_cell_type[right_label][: self.cluster_size])
        if len(left_records) < self.cluster_size or len(right_records) < self.cluster_size:
            raise RuntimeError("Cluster comparison sampling requires two coherent groups of cluster_size cells.")
        public_input = {
            "cluster_A": [self._limit_genes(record.genes) for record in left_records],
            "cluster_B": [self._limit_genes(record.genes) for record in right_records],
        }
        hidden_grounding = {
            "cluster_A": {
                "label": left_label,
                "records": [record.hidden_grounding() for record in left_records],
            },
            "cluster_B": {
                "label": right_label,
                "records": [record.hidden_grounding() for record in right_records],
            },
        }
        return BuildExample(
            question_type="cluster_comparison",
            level="cluster",
            public_input=public_input,
            hidden_grounding=hidden_grounding,
            source_record_ids=tuple(record.sample_id for record in left_records + right_records),
        )

    def _build_cell_type_pairs(self) -> list[tuple[str, str]]:
        eligible_labels = [label for label, group in self.by_cell_type.items() if len(group) >= self.cluster_size]
        pairs: list[tuple[str, str]] = []
        for index, left_label in enumerate(eligible_labels):
            left_record = self.by_cell_type[left_label][0]
            for right_label in eligible_labels[index + 1 :]:
                right_record = self.by_cell_type[right_label][0]
                if left_record.compartment and right_record.compartment:
                    if left_record.compartment == right_record.compartment and left_record.cell_type == right_record.cell_type:
                        continue
                pairs.append((left_label, right_label))
        self.rng.shuffle(pairs)
        return pairs

    def _limit_genes(self, genes: Sequence[str]) -> list[str]:
        return list(genes[: self.max_genes])


def discover_source_files(source_root: Path, splits: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for split in splits:
        if split == "all":
            paths.extend(sorted(source_root.glob("*/cell_annotation_rationale.jsonl")))
            continue
        paths.extend(sorted(source_root.glob(f"*/cell_annotation_rationale_{split}.jsonl")))
    if not paths:
        raise RuntimeError(
            f"No cell_annotation_rationale JSONL files were found under '{source_root}' for splits {tuple(splits)}."
        )
    return paths


def build_system_prompt(spec: QuestionSpec) -> str:
    rules = "\n".join(f"- {rule}" for rule in FAIRNESS_RULES)
    return (
        "You are building a benchmark dataset for reasoning over single-cell gene-expression profiles. "
        "Write a natural user question and a grounded expert answer.\n"
        f"Question type instruction: {spec.prompt_instruction}\n"
        "Rules:\n"
        f"{rules}\n"
        "The answer must explicitly cite at least one evidence gene from the hidden grounding.\n"
        "In the answer, every gene or marker name you mention must come only from the hidden-grounding evidence-gene or negative-marker whitelist.\n"
        "Use only exact whitelist strings. Do not invent aliases, lineage shorthand, pathway shorthand, cell-type shorthand, or slash-compressed forms such as TCR, CD3/T, or TRBC1/TRBC2 unless that exact string appears in the whitelist.\n"
        "If you mention a lineage or biological program in prose, write it as normal lowercase words rather than uppercase shorthand tokens. For example, prefer 't-cell-associated program' over 'CD3/T-CELL' and 'nk-associated program' over 'NK-ASSOCIATED'.\n"
        "Any cell-type, lineage, compartment, or identity label in the answer must come from the hidden-grounding identity whitelist or a simple morphological variant of it. Do not invent more specific labels than the hidden grounding supports.\n"
        "Do not mention any visible-input gene unless it also appears in that hidden-grounding whitelist.\n"
        "Return strict JSON with keys question, answer, and quality_flags."
    )


def build_user_prompt(
    *,
    question_type: str,
    level: str,
    visible_input: dict[str, Any],
    hidden_grounding: dict[str, Any],
    retry_feedback: str | None = None,
) -> str:
    grounding_gene_whitelist = sorted(_collect_allowed_grounding_genes(hidden_grounding))
    grounding_identity_whitelist = sorted(_collect_allowed_grounding_identity_aliases(hidden_grounding))
    payload = {
        "question_type": question_type,
        "level": level,
        "visible_input": visible_input,
        "hidden_grounding": hidden_grounding,
        "answer_grounding_gene_whitelist": grounding_gene_whitelist,
        "answer_grounding_identity_whitelist": grounding_identity_whitelist,
        "grounding_instruction": (
            "When writing the answer, only mention genes from answer_grounding_gene_whitelist. "
            "If a visible gene is not in the whitelist, do not cite it. "
            "Evidence genes are expected to be visible in the input gene list or lists for the corresponding sample or samples. "
            "Use exact whitelist strings only: do not rewrite them as aliases, do not collapse multiple genes into slash forms, and do not introduce lineage shorthand such as TCR or CD3/T unless that exact token is in the whitelist. "
            "The whitelist contains only evidence genes and negative markers; positive markers outside that set are not allowed in the answer. "
            "Write non-gene biological descriptions in normal lowercase prose, not in uppercase shorthand labels. "
            "Any cell-type or lineage description must stay within answer_grounding_identity_whitelist; do not invent more specific labels than those identities support."
        ),
        "required_output_schema": {
            "question": "string",
            "answer": "string",
            "quality_flags": {
                "uses_evidence_genes": True,
                "answer_only_uses_grounding_genes": True,
                "answer_only_uses_grounding_identities": True,
                "requires_hidden_metadata_in_question": False,
                "notes": "string",
            },
        },
    }
    if retry_feedback:
        retry_payload = {
            "validator_error": retry_feedback,
            "required_fix": (
                "Regenerate the answer from scratch. Remove every forbidden gene or marker mention, "
                "and only use exact evidence-gene or negative-marker strings from answer_grounding_gene_whitelist."
            ),
        }
        marker = "mentioned genes outside hidden grounding:"
        if marker in retry_feedback:
            banned_terms = [term.strip() for term in retry_feedback.split(marker, 1)[1].split(",") if term.strip()]
            retry_payload["do_not_repeat_terms"] = banned_terms
            retry_payload["term_handling_instruction"] = (
                "Do not reuse any banned term. If you need to describe a lineage or program, "
                "rewrite it as normal lowercase prose without all-caps shorthand or slash-compressed labels."
            )
        identity_marker = "mentioned identity labels outside hidden grounding:"
        if identity_marker in retry_feedback:
            banned_identities = [
                term.strip() for term in retry_feedback.split(identity_marker, 1)[1].split(",") if term.strip()
            ]
            retry_payload["do_not_repeat_identity_terms"] = banned_identities
            retry_payload["identity_handling_instruction"] = (
                "Do not reuse any banned identity term. Restrict cell-type and lineage descriptions "
                "to answer_grounding_identity_whitelist or simpler wording directly supported by it."
            )
        payload["retry_feedback"] = retry_payload
    return json.dumps(payload, indent=2)


def validate_generated_response(
    *,
    question_type: str,
    question: str,
    answer: str,
    visible_input: dict[str, Any],
    hidden_grounding: dict[str, Any],
    known_identity_aliases: set[str] | None = None,
) -> None:
    if not question or not answer:
        raise ValueError(f"Generated response for '{question_type}' is missing a question or answer.")
    if _question_violates_fairness(question_type=question_type, question=question):
        raise ValueError(
            f"Generated question for '{question_type}' violates the metadata fairness rule: {question}"
        )
    evidence_genes = {gene.upper() for gene in _collect_grounding_evidence_genes(hidden_grounding)}
    allowed_answer_genes = _collect_allowed_grounding_genes(hidden_grounding)
    answer_upper = answer.upper()
    if evidence_genes and not any(re.search(rf"\b{re.escape(gene)}\b", answer_upper) for gene in evidence_genes):
        raise ValueError(
            f"Generated answer for '{question_type}' did not cite any evidence genes from the hidden grounding."
        )
    disallowed_answer_mentions = sorted(
        gene for gene in _extract_gene_mentions(answer) if gene not in allowed_answer_genes
    )
    if disallowed_answer_mentions:
        raise ValueError(
            f"Generated answer for '{question_type}' mentioned genes outside hidden grounding: {', '.join(disallowed_answer_mentions[:8])}"
        )
    allowed_identity_aliases = _collect_allowed_grounding_identity_aliases(hidden_grounding)
    mentioned_identity_aliases = _extract_identity_mentions(
        answer,
        candidate_aliases=known_identity_aliases or allowed_identity_aliases,
    )
    disallowed_identity_mentions = sorted(
        alias for alias in mentioned_identity_aliases if alias not in allowed_identity_aliases
    )
    if disallowed_identity_mentions:
        raise ValueError(
            f"Generated answer for '{question_type}' mentioned identity labels outside hidden grounding: {', '.join(disallowed_identity_mentions[:8])}"
        )
    validate_public_input(visible_input)


def validate_hidden_grounding_against_public_input(
    *,
    visible_input: dict[str, Any],
    hidden_grounding: dict[str, Any],
) -> None:
    missing_evidence = _collect_evidence_genes_missing_from_input(
        visible_input=visible_input,
        hidden_grounding=hidden_grounding,
    )
    if not missing_evidence:
        return
    parts = [f"{scope}: {', '.join(genes[:6])}" for scope, genes in sorted(missing_evidence.items())]
    raise ValueError(
        "Hidden grounding evidence genes were not present in the visible input gene set(s): "
        + "; ".join(parts[:4])
    )


def validate_public_input(payload: dict[str, Any]) -> None:
    allowed_top_level = {"genes", "cells", "cluster_A", "cluster_B"}
    unknown_fields = set(payload) - allowed_top_level
    if unknown_fields:
        raise ValueError(f"Public model_input contains unsupported fields: {sorted(unknown_fields)}")
    for value in payload.values():
        _validate_gene_container(value)


def _validate_gene_container(value: Any) -> None:
    if isinstance(value, list):
        if not value:
            raise ValueError("Public model_input contains an empty gene container.")
        if all(isinstance(item, str) for item in value):
            return
        if all(isinstance(item, list) for item in value):
            for item in value:
                if not item or not all(isinstance(gene, str) for gene in item):
                    raise ValueError("Cluster inputs must be nested lists of gene strings.")
            return
    raise ValueError("Public model_input must contain only gene lists.")


def _collect_evidence_genes_missing_from_input(
    *,
    visible_input: dict[str, Any],
    hidden_grounding: dict[str, Any],
) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}

    def compare(scope: str, visible_genes: Sequence[str], record: dict[str, Any]) -> None:
        visible_gene_set = _normalize_gene_set(visible_genes)
        evidence_gene_set = _normalize_gene_set(record.get("evidence_genes") or [])
        absent = sorted(gene for gene in evidence_gene_set if gene not in visible_gene_set)
        if absent:
            missing[scope] = absent

    if "record" in hidden_grounding:
        compare("record", visible_input.get("genes") or [], hidden_grounding["record"])
        return missing

    if "records" in hidden_grounding:
        visible_cells = visible_input.get("cells") or []
        records = hidden_grounding.get("records") or []
        if len(visible_cells) != len(records):
            raise ValueError(
                f"Visible cluster cell count {len(visible_cells)} did not match hidden grounding record count {len(records)}."
            )
        for index, (visible_genes, record) in enumerate(zip(visible_cells, records)):
            compare(f"cells[{index}]", visible_genes, record)
        return missing

    for cluster_name in ("cluster_A", "cluster_B"):
        cluster = hidden_grounding.get(cluster_name)
        if not cluster:
            continue
        visible_cluster = visible_input.get(cluster_name) or []
        records = cluster.get("records") or []
        if len(visible_cluster) != len(records):
            raise ValueError(
                f"Visible {cluster_name} cell count {len(visible_cluster)} did not match hidden grounding record count {len(records)}."
            )
        for index, (visible_genes, record) in enumerate(zip(visible_cluster, records)):
            compare(f"{cluster_name}[{index}]", visible_genes, record)
    return missing


def _normalize_gene_set(genes: Sequence[str]) -> set[str]:
    return {
        str(gene).strip().upper()
        for gene in genes
        if str(gene).strip() and str(gene).strip().lower() != "none"
    }


def _collect_grounding_evidence_genes(hidden_grounding: dict[str, Any]) -> list[str]:
    genes: list[str] = []
    if "record" in hidden_grounding:
        genes.extend(hidden_grounding["record"].get("evidence_genes") or [])
    if "records" in hidden_grounding:
        for record in hidden_grounding["records"]:
            genes.extend(record.get("evidence_genes") or [])
    for cluster_name in ("cluster_A", "cluster_B"):
        cluster = hidden_grounding.get(cluster_name)
        if not cluster:
            continue
        for record in cluster.get("records") or []:
            genes.extend(record.get("evidence_genes") or [])
    return [gene for gene in genes if gene]


def _collect_allowed_grounding_genes(hidden_grounding: dict[str, Any]) -> set[str]:
    genes: set[str] = set()

    def collect_from_record(record: dict[str, Any]) -> None:
        for field in ("evidence_genes", "negative_markers"):
            for gene in record.get(field) or []:
                normalized = str(gene).strip().upper()
                if normalized:
                    genes.add(normalized)

    if "record" in hidden_grounding:
        collect_from_record(hidden_grounding["record"])
    if "records" in hidden_grounding:
        for record in hidden_grounding["records"]:
            collect_from_record(record)
    for cluster_name in ("cluster_A", "cluster_B"):
        cluster = hidden_grounding.get(cluster_name)
        if not cluster:
            continue
        for record in cluster.get("records") or []:
            collect_from_record(record)
    return genes


def _build_identity_alias_vocabulary(records: Sequence[SourceRecord]) -> set[str]:
    aliases: set[str] = set()
    for record in records:
        hidden_record = record.hidden_grounding()
        aliases.update(_collect_identity_aliases_from_record(hidden_record))
    return aliases


def _collect_allowed_grounding_identity_aliases(hidden_grounding: dict[str, Any]) -> set[str]:
    aliases: set[str] = set()

    if "record" in hidden_grounding:
        aliases.update(_collect_identity_aliases_from_record(hidden_grounding["record"]))
    if "records" in hidden_grounding:
        for record in hidden_grounding["records"]:
            aliases.update(_collect_identity_aliases_from_record(record))
    for cluster_name in ("cluster_A", "cluster_B"):
        cluster = hidden_grounding.get(cluster_name)
        if not cluster:
            continue
        cluster_label = cluster.get("label")
        aliases.update(_expand_identity_aliases(cluster_label))
        for record in cluster.get("records") or []:
            aliases.update(_collect_identity_aliases_from_record(record, cluster_label=cluster_label))
    return aliases


def _collect_identity_aliases_from_record(
    record: dict[str, Any],
    *,
    cluster_label: str | None = None,
) -> set[str]:
    aliases: set[str] = set()
    metadata = record.get("metadata") or {}
    provenance = metadata.get("provenance") or {}

    terms = [
        cluster_label,
        record.get("cell_type"),
        record.get("broad_cell_class"),
        record.get("compartment"),
        provenance.get("canonical_label_for_grounding"),
    ]
    for term in terms:
        aliases.update(_expand_identity_aliases(term))

    broad_cell_class = _normalize_identity_alias(record.get("broad_cell_class"))
    compartment = _normalize_identity_alias(record.get("compartment"))
    compartment_variants = _expand_identity_aliases(compartment)
    if broad_cell_class and compartment_variants:
        aliases.add(f"{next(iter(sorted(compartment_variants)))} {broad_cell_class}")
        broad_stem = _identity_stem(broad_cell_class)
        if broad_stem:
            for compartment_variant in compartment_variants:
                aliases.add(f"{broad_stem} {compartment_variant}")
                aliases.add(f"{compartment_variant} {broad_stem}")

    return {alias for alias in aliases if _is_meaningful_identity_alias(alias)}


def _expand_identity_aliases(term: Any) -> set[str]:
    normalized = _normalize_identity_alias(term)
    if not normalized:
        return set()

    aliases = {normalized}
    if "-" in normalized:
        aliases.add(normalized.replace("-", " "))
    if " " in normalized:
        aliases.add(normalized.replace(" ", "-"))

    if normalized.endswith(" cell"):
        aliases.add(f"{normalized}-like")
        aliases.add(f"{normalized} like")

    if normalized == "epithelium":
        aliases.add("epithelial")
    if normalized == "stroma":
        aliases.add("stromal")

    return {alias for alias in aliases if _is_meaningful_identity_alias(alias)}


def _normalize_identity_alias(term: Any) -> str:
    if term is None:
        return ""
    text = str(term).strip().lower()
    if not text or text == "none":
        return ""
    text = text.replace("–", "-").replace("—", "-").replace("_", " ")
    text = re.sub(r"[^a-z0-9\-/\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _identity_stem(alias: str) -> str:
    if alias.endswith(" cells"):
        stem = alias[: -len(" cells")].strip()
    elif alias.endswith(" cell"):
        stem = alias[: -len(" cell")].strip()
    else:
        return ""
    if len(stem) <= 1:
        return ""
    return stem


def _is_meaningful_identity_alias(alias: str) -> bool:
    normalized = _normalize_identity_alias(alias)
    if len(normalized) < 3:
        return False
    tokens = [token for token in re.split(r"[\s\-/]+", normalized) if token]
    if not tokens:
        return False
    if all(token in IDENTITY_ALIAS_STOPWORDS for token in tokens):
        return False
    return True


def _extract_identity_mentions(text: str, *, candidate_aliases: set[str]) -> set[str]:
    normalized_text = _normalize_identity_alias(text)
    mentions: set[str] = set()
    for alias in sorted(candidate_aliases, key=len, reverse=True):
        if not alias:
            continue
        tokens = [token for token in re.split(r"[\s\-]+", alias) if token]
        if not tokens:
            continue
        pattern = r"(?<![a-z0-9])" + r"[-\s]+".join(re.escape(token) for token in tokens) + r"(?![a-z0-9])"
        if re.search(pattern, normalized_text):
            mentions.add(alias)
    return mentions


def _extract_gene_mentions(text: str) -> set[str]:
    def flush_token(raw_token: list[str], mentions: set[str]) -> None:
        candidate_raw = "".join(raw_token).strip("/")
        if len(candidate_raw) < 3:
            return
        uppercase_count = sum(1 for char in candidate_raw if char.isupper())
        has_digit = any(char.isdigit() for char in candidate_raw)
        if uppercase_count < 2 and not has_digit:
            return
        alpha_chunks = [chunk.lower() for chunk in re.findall(r"[A-Za-z]+", candidate_raw)]
        if any(chunk in GENE_TOKEN_BLACKLIST_CHUNKS for chunk in alpha_chunks if chunk != "orf"):
            return
        candidate = candidate_raw.upper()
        if candidate not in GENE_TOKEN_EXCEPTIONS:
            mentions.add(candidate)

    mentions: set[str] = set()
    token: list[str] = []
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


def _question_violates_fairness(*, question_type: str, question: str) -> bool:
    if question_type == "tissue_compatible_interpretation":
        if any(pattern.search(question) for pattern in ALLOWED_TISSUE_PROGRAM_PATTERNS):
            return False
    return any(pattern.search(question) for pattern in QUESTION_GUARD_PATTERNS)