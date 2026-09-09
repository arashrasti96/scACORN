from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Stage3Paths


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_expert_profiles(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = _read_json(path)
    experts = payload.get("experts") if isinstance(payload, dict) else None
    if not isinstance(experts, dict):
        return {}
    return {
        str(name): profile
        for name, profile in experts.items()
        if isinstance(profile, dict)
    }


def _normalize_manifest_path(raw_path: str | None, *, repo_root: Path, fallback: Path | None = None) -> Path | None:
    if not raw_path:
        return fallback

    candidate = Path(raw_path)
    if candidate.exists():
        return candidate

    marker = "/agentic-context-engine/"
    if marker in raw_path:
        suffix = raw_path.split(marker, 1)[1]
        remapped = repo_root / suffix
        if remapped.exists():
            return remapped

    if fallback is not None and fallback.exists():
        return fallback
    return candidate


def _derive_tissue_label(dataset_name: str) -> str:
    cleaned = dataset_name
    if cleaned.endswith("_cell_annotation"):
        cleaned = cleaned[: -len("_cell_annotation")]
    prefixes = (
        "tabula_sapiens_",
        "tabula_sapiens_immune_",
        "human_cell_atlas_",
    )
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    return cleaned.replace("_", " ")


def _short_expert_summary(dataset_name: str, tissue_label: str, manifest: dict[str, Any]) -> str:
    counts = manifest.get("train_task_counts") or {}
    task_fragment = ", ".join(sorted(counts)) if counts else "cell-type grounding"
    return (
        f"Expert for dataset '{dataset_name}' focused on tissue/domain '{tissue_label}'. "
        f"Adapter trained for {task_fragment} with stage-1 domain context preserved."
    )


def _render_profile_description(
    dataset_name: str,
    tissue_label: str,
    manifest: dict[str, Any],
    profile: dict[str, Any] | None,
) -> str:
    fallback = _short_expert_summary(dataset_name, tissue_label, manifest)
    if not profile:
        return fallback

    explicit = profile.get("description")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    parts: list[str] = []
    specialization = profile.get("specialization")
    if isinstance(specialization, str) and specialization.strip():
        parts.append(specialization.strip())

    preferred_queries = profile.get("preferred_queries")
    if isinstance(preferred_queries, list) and preferred_queries:
        rendered_queries = ", ".join(str(item).strip() for item in preferred_queries if str(item).strip())
        if rendered_queries:
            parts.append(f"Best used for: {rendered_queries}.")

    routing_hints = profile.get("routing_hints")
    if isinstance(routing_hints, list) and routing_hints:
        rendered_hints = "; ".join(str(item).strip() for item in routing_hints if str(item).strip())
        if rendered_hints:
            parts.append(f"Routing hints: {rendered_hints}.")

    strengths = profile.get("strengths")
    if isinstance(strengths, list) and strengths:
        rendered_strengths = ", ".join(str(item).strip() for item in strengths if str(item).strip())
        if rendered_strengths:
            parts.append(f"Strengths: {rendered_strengths}.")

    limitations = profile.get("limitations")
    if isinstance(limitations, list) and limitations:
        rendered_limitations = ", ".join(str(item).strip() for item in limitations if str(item).strip())
        if rendered_limitations:
            parts.append(f"Limitations: {rendered_limitations}.")

    return " ".join(parts) if parts else fallback


@dataclass(frozen=True)
class ExpertDescriptor:
    name: str
    dataset_name: str
    tissue_label: str
    description: str
    profile: dict[str, Any]
    run_dir: Path
    adapter_dir: Path
    task_manifest_path: Path
    training_summary_path: Path | None
    stage1_adapter_path: Path | None
    stage1_adapter_name: str
    stage2_adapter_name: str
    supported_task_types: tuple[str, ...] = field(default_factory=tuple)
    task_manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return self.name.replace("_", " ")

    @classmethod
    def from_run_dir(
        cls,
        run_dir: Path,
        *,
        repo_root: Path,
        expert_profiles: dict[str, dict[str, Any]] | None = None,
    ) -> "ExpertDescriptor":
        task_manifest_path = run_dir / "task_manifest.json"
        manifest = _read_json(task_manifest_path)
        dataset_context = manifest.get("dataset_context") or {}
        dataset_name = (
            dataset_context.get("dataset_type_resolved")
            or dataset_context.get("dataset_type_requested")
            or run_dir.name
        )
        tissue_label = _derive_tissue_label(dataset_name)
        adapter_dir = run_dir / "adapter"
        raw_stage1_path = manifest.get("stage1_adapter") or manifest.get("args", {}).get("stage1_adapter")
        stage1_adapter_path = _normalize_manifest_path(raw_stage1_path, repo_root=repo_root, fallback=None)
        training_summary_path = run_dir / "training_summary.json"
        supported_task_types = tuple(sorted((manifest.get("train_task_counts") or {}).keys()))
        profile = dict((expert_profiles or {}).get(dataset_name) or {})
        return cls(
            name=dataset_name,
            dataset_name=dataset_name,
            tissue_label=tissue_label,
            description=_render_profile_description(dataset_name, tissue_label, manifest, profile),
            profile=profile,
            run_dir=run_dir,
            adapter_dir=adapter_dir,
            task_manifest_path=task_manifest_path,
            training_summary_path=training_summary_path if training_summary_path.exists() else None,
            stage1_adapter_path=stage1_adapter_path,
            stage1_adapter_name=manifest.get("stage1_adapter_name", "stage1_domain"),
            stage2_adapter_name=manifest.get("stage2_adapter_name", "default"),
            supported_task_types=supported_task_types,
            task_manifest=manifest,
        )


def discover_experts(
    outputs_root: Path,
    *,
    repo_root: Path,
    expert_profiles_path: Path | None = None,
    limit: int | None = None,
) -> list[ExpertDescriptor]:
    expert_profiles = _load_expert_profiles(expert_profiles_path)
    experts: list[ExpertDescriptor] = []
    for run_dir in sorted(path for path in outputs_root.iterdir() if path.is_dir()):
        task_manifest_path = run_dir / "task_manifest.json"
        adapter_dir = run_dir / "adapter"
        if not task_manifest_path.exists() or not adapter_dir.exists():
            continue
        experts.append(
            ExpertDescriptor.from_run_dir(
                run_dir,
                repo_root=repo_root,
                expert_profiles=expert_profiles,
            )
        )
        if limit is not None and len(experts) >= limit:
            break
    return experts


def load_expert_registry(paths: Stage3Paths | None = None, *, outputs_root: Path | None = None, limit: int | None = None) -> dict[str, ExpertDescriptor]:
    resolved_paths = paths or Stage3Paths.discover()
    root = outputs_root or resolved_paths.stage2_outputs_root
    experts = discover_experts(
        root,
        repo_root=resolved_paths.repo_root,
        expert_profiles_path=resolved_paths.expert_profiles_path,
        limit=limit,
    )
    return {expert.name: expert for expert in experts}