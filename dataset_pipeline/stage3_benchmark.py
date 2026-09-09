from __future__ import annotations

import json
import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import PipelineConfig
from .io_utils import ensure_dir, write_json
from .marker_catalog import candidate_labels_for_gold


GROUPED_SPLITS = ("train", "val", "test")
DIRECTORY_SUFFIX = "_cell_annotation"
ANSWER_SCHEMA_NAME = "stage3_orchestration_v1"
PROFILE_BUNDLE_ANSWER_SCHEMA_NAME = "stage3_profile_bundle_v1"
REQUIRED_OUTPUT_FIELDS = [
    "final_label",
    "alternative_labels",
    "predicted_tissue",
    "supporting_genes",
    "confidence",
    "abstain",
]
PROFILE_BUNDLE_REQUIRED_OUTPUT_FIELDS = [
    "profile_labels",
    "profile_supporting_genes",
    "combined_markers",
    "combined_summary",
    "abstain",
]

DIRECT_CONTEXT_KEYS = (
    "tissue",
    "assay",
    "disease",
    "sex",
    "development_stage",
    "suspension_type",
)

NONLEAK_CONTEXT_KEYS = (
    "assay",
    "disease",
    "sex",
    "development_stage",
    "suspension_type",
)

QUESTION_SUFFIX = (
    "Return JSON with keys final_label, alternative_labels, predicted_tissue, "
    "supporting_genes, confidence, and abstain. Use at most 5 supporting_genes, "
    "and choose genes from the input only."
)
PROFILE_BUNDLE_SUFFIX = (
    "Return JSON with keys profile_labels, profile_supporting_genes, combined_markers, "
    "combined_summary, and abstain. Set profile_labels to an object keyed by profile ID. "
    "Set profile_supporting_genes to an object keyed by profile ID, with at most 5 genes per profile, "
    "chosen from that profile's ordered input only. Set combined_markers to a deduplicated list drawn from the inputs."
)

ANNOTATION_QUESTION_TEMPLATES = (
    "Identify the most likely cell type for this ranked single-cell gene-expression profile. If uncertain, include up to 3 ranked alternative_labels and infer the most likely tissue context if possible.",
    "Determine which cell type best matches this ordered single-cell gene-expression signature. If the call is uncertain, provide up to 3 ranked alternative_labels and infer the likely tissue context when possible.",
    "Based on this ranked gene list from one cell, predict the most likely cell annotation. If needed, add up to 3 ranked alternative_labels and the most likely tissue context.",
    "Use this single-cell expression ranking to assign the best-supported cell type. When uncertainty remains, report up to 3 ranked alternative_labels and the most likely tissue context.",
    "From the ordered genes in this single-cell profile, infer the most likely cell identity. If the evidence is mixed, include up to 3 ranked alternative_labels and infer tissue context if possible.",
    "Decide which cell label is most consistent with this ranked transcriptomic profile. If you are not fully certain, include up to 3 ranked alternative_labels and the most likely tissue context.",
    "What cell annotation best explains this ranked gene-expression profile for a single cell? If the answer is uncertain, add up to 3 ranked alternative_labels and infer the tissue context when possible.",
    "Which cell type is the best match for this ordered single-cell gene list? If ambiguity remains, provide up to 3 ranked alternative_labels and the most likely tissue context.",
    "Provide the most plausible cell annotation for this ranked expression profile from one cell. If needed, include up to 3 ranked alternative_labels and infer the likely tissue context.",
    "Classify this single-cell gene-expression profile by its most likely cell type. If confidence is limited, report up to 3 ranked alternative_labels and the most likely tissue context.",
    "Interpret this ranked set of expressed genes and report the most likely cell type. If the profile is ambiguous, include up to 3 ranked alternative_labels and infer tissue context if possible.",
    "Use the gene-expression ordering in this single cell to infer the best cell-type label. When uncertain, add up to 3 ranked alternative_labels and the most likely tissue context.",
    "Assign the most likely cell label to this ranked single-cell expression profile. If the evidence does not support a single clear answer, provide up to 3 ranked alternative_labels and infer tissue context when possible.",
    "Read this ordered transcriptomic profile and identify the most likely cell type. If necessary, include up to 3 ranked alternative_labels and the most likely tissue context.",
    "Predict the primary cell identity represented by this ranked gene-expression profile. If the decision is uncertain, report up to 3 ranked alternative_labels and infer tissue context if possible.",
    "Determine the best-supported cell annotation for this one-cell ranked gene list. When the profile is not decisive, include up to 3 ranked alternative_labels and the most likely tissue context.",
    "Which cell label is most consistent with this ranked single-cell profile? If uncertainty remains, list up to 3 ranked alternative_labels and infer the likely tissue context.",
    "Infer the cell annotation that best fits this ordered expression signature from a single cell. If ambiguous, include up to 3 ranked alternative_labels and the most likely tissue context.",
    "Select the most likely cell type for this ranked gene-expression pattern. If the top call is not definitive, provide up to 3 ranked alternative_labels and infer tissue context when possible.",
    "Report the best cell-type call for this single-cell ranked gene list. If the evidence allows multiple possibilities, include up to 3 ranked alternative_labels and the most likely tissue context.",
)

TISSUE_QUESTION_TEMPLATES = (
    "Infer the most likely tissue of origin for this ranked single-cell gene-expression profile. Set final_label to the inferred tissue name and copy the same value into predicted_tissue.",
    "Determine which tissue best matches this ordered single-cell gene-expression signature. Use the inferred tissue name for both final_label and predicted_tissue.",
    "Based on this ranked gene list from one cell, predict the most likely tissue of origin. Put the same tissue name in final_label and predicted_tissue.",
    "Use this single-cell expression ranking to identify the most likely tissue context. Set final_label and predicted_tissue to the same inferred tissue value.",
    "From the ordered genes in this single-cell profile, infer the tissue of origin. The inferred tissue should appear in both final_label and predicted_tissue.",
    "Decide which tissue label is most consistent with this ranked transcriptomic profile. Copy that tissue name into both final_label and predicted_tissue.",
    "What tissue or compartment is most strongly supported by this ranked gene-expression profile? Use the same tissue name for final_label and predicted_tissue.",
    "Which tissue is the best match for this ordered single-cell gene list? Set both final_label and predicted_tissue to that tissue.",
    "Provide the most plausible tissue of origin for this ranked expression profile from one cell. Use the inferred tissue in both final_label and predicted_tissue.",
    "Classify this single-cell gene-expression profile by its most likely tissue context. Write the same tissue name into final_label and predicted_tissue.",
    "Interpret this ranked set of expressed genes and report the most likely tissue of origin. The answer should be copied into both final_label and predicted_tissue.",
    "Use the gene-expression ordering in this single cell to infer the best tissue label. Set final_label and predicted_tissue to the same inferred tissue.",
    "Assign the most likely tissue label to this ranked single-cell expression profile. Use that same tissue value for final_label and predicted_tissue.",
    "Read this ordered transcriptomic profile and identify the most likely tissue context. Repeat the inferred tissue in final_label and predicted_tissue.",
    "Predict the primary tissue of origin represented by this ranked gene-expression profile. Put the same tissue name in both final_label and predicted_tissue.",
    "Determine the best-supported tissue assignment for this one-cell ranked gene list. The chosen tissue should fill both final_label and predicted_tissue.",
    "Which tissue label is most consistent with this ranked single-cell profile? Use that same tissue name for final_label and predicted_tissue.",
    "Infer the tissue context that best fits this ordered expression signature from a single cell. Copy the answer into both final_label and predicted_tissue.",
    "Select the most likely tissue for this ranked gene-expression pattern. Set final_label and predicted_tissue to the same tissue value.",
    "Report the best tissue call for this single-cell ranked gene list. Use the inferred tissue name for both final_label and predicted_tissue.",
)

DIFFERENTIAL_QUESTION_TEMPLATES = (
    "Choose the best-matching cell label from the following candidate set: {candidate_panel}. Also infer the most likely tissue context if possible.",
    "From this candidate label panel, select the cell type that best fits the ranked profile: {candidate_panel}. Also infer the likely tissue context when possible.",
    "Decide which candidate label best explains this ordered single-cell expression signature: {candidate_panel}. Also infer the most likely tissue context if possible.",
    "Use the ranked gene-expression profile to choose the strongest label from this candidate set: {candidate_panel}. Also infer tissue context when possible.",
    "Among these candidate labels, determine the best match for this single-cell profile: {candidate_panel}. Also report the most likely tissue context if possible.",
    "Pick the best-supported cell annotation from the following options: {candidate_panel}. Also infer the most likely tissue context when possible.",
    "Resolve this differential diagnosis by choosing one label from: {candidate_panel}. Also infer the likely tissue context if possible.",
    "Which candidate label is the best fit for this ranked gene-expression pattern: {candidate_panel}? Also infer tissue context when possible.",
    "Select the most plausible cell label from this candidate panel: {candidate_panel}. Also infer the most likely tissue context if possible.",
    "Use this ordered gene list to pick the best label from the candidate set: {candidate_panel}. Also infer tissue context when possible.",
    "Determine the top cell-type match from these candidates: {candidate_panel}. Also infer the most likely tissue context if possible.",
    "Choose the single best diagnosis from the following candidate labels: {candidate_panel}. Also infer tissue context when possible.",
    "Based on this ranked profile, select the best-matching label from: {candidate_panel}. Also infer the most likely tissue context if possible.",
    "From the candidate labels below, identify the one most consistent with this cell: {candidate_panel}. Also infer tissue context when possible.",
    "Pick the strongest cell-type call from this provided panel: {candidate_panel}. Also infer the likely tissue context if possible.",
    "Choose the best candidate label for this ranked single-cell expression signature: {candidate_panel}. Also infer tissue context when possible.",
    "Which option in this candidate panel best explains the profile: {candidate_panel}? Also infer the most likely tissue context if possible.",
    "Use this single-cell ranked gene list to discriminate among these candidates: {candidate_panel}. Also infer tissue context when possible.",
    "Identify the best label from the following candidate set for this cell: {candidate_panel}. Also infer the likely tissue context if possible.",
    "Return the best-matching candidate label from this panel: {candidate_panel}. Also infer the most likely tissue context when possible.",
)

CROSS_EXPERT_QUESTION_TEMPLATES = (
    "The tissue of origin is hidden. Infer the most likely tissue context and final cell identity for this cell, and include up to 3 alternative_labels if uncertainty remains.",
    "The source tissue is unknown. Infer both the most likely tissue context and the final cell identity, and add up to 3 alternative_labels if needed.",
    "Treat this as a hidden-tissue annotation problem. Infer the likely tissue context and the final cell label, with up to 3 alternative_labels if uncertainty remains.",
    "The tissue context has been withheld. Infer the most likely tissue and final cell identity, adding up to 3 alternative_labels if needed.",
    "You do not know the tissue of origin in advance. Infer the best tissue context and final cell annotation, and include up to 3 alternative_labels if the case is ambiguous.",
    "This cell may require resolving ambiguity across tissues. Infer the most likely tissue context and final cell identity, and provide up to 3 alternative_labels if needed.",
    "The tissue is intentionally hidden. Infer the tissue context and final cell-type answer, and include up to 3 alternative_labels when uncertainty remains.",
    "Solve this hidden-tissue cell annotation task by inferring the most likely tissue and final cell label, with up to 3 alternative_labels if necessary.",
    "Because the tissue of origin is unknown, infer the most likely tissue context and final cell identity, and include up to 3 alternative_labels if needed.",
    "The tissue context is not provided. Infer the likely tissue and final cell annotation, adding up to 3 alternative_labels if uncertainty remains.",
    "Infer the most likely tissue context and final cell identity for this ambiguous profile, and include up to 3 alternative_labels if needed.",
    "The correct tissue must be inferred rather than assumed. Infer the tissue context and final cell label, with up to 3 alternative_labels if needed.",
    "Infer the most likely tissue and final cell identity for this hidden-tissue cell, and include up to 3 alternative_labels if uncertainty remains.",
    "This is a tissue-hidden annotation problem. Infer the likely tissue context and final cell identity, and add up to 3 alternative_labels when needed.",
    "For a cell with unknown tissue of origin, infer the most likely tissue context and final cell annotation, with up to 3 alternative_labels if needed.",
    "The tissue is concealed. Infer the most likely tissue and final cell-type label, and include up to 3 alternative_labels if the case is uncertain.",
    "Resolve this hidden-origin cell by inferring the best tissue context and final cell identity, and provide up to 3 alternative_labels if uncertainty remains.",
    "A tissue label is not given. Infer the likely tissue and final cell annotation, with up to 3 alternative_labels if needed.",
    "The origin tissue is unknown. Infer the most likely tissue context and final cell identity, adding up to 3 alternative_labels if required.",
    "Solve this hidden-tissue annotation task by inferring both the most likely tissue context and the final cell label, and include up to 3 alternative_labels if uncertainty remains.",
)

OOD_QUESTION_TEMPLATES = (
    "If the available evidence is insufficient for a reliable answer, abstain instead of forcing an answer. When abstaining, set final_label and predicted_tissue to unknown.",
    "If the profile does not support a reliable decision, abstain rather than guessing. On abstention, set final_label and predicted_tissue to unknown.",
    "If the cell cannot be assigned with adequate confidence, abstain instead of forcing a label. When abstaining, use unknown for final_label and predicted_tissue.",
    "If there is not enough evidence for a reliable answer, abstain. On abstention, set both final_label and predicted_tissue to unknown.",
    "If the profile is not suitable for a confident decision, abstain rather than overcalling. When abstaining, final_label and predicted_tissue must both be unknown.",
    "If this cell cannot be reliably explained from the observed evidence, abstain instead of inventing an answer. Use unknown for both final_label and predicted_tissue when abstaining.",
    "If the cell is not adequately supported by the evidence, abstain. When abstaining, set final_label and predicted_tissue to unknown.",
    "If the profile does not permit a reliable match, abstain rather than forcing a prediction. On abstention, use unknown for final_label and predicted_tissue.",
    "If the evidence remains inconclusive, abstain instead of returning a forced answer. Set final_label and predicted_tissue to unknown when abstaining.",
    "If the cell does not support a reliable label or tissue call, abstain. In that case, final_label and predicted_tissue should both be unknown.",
    "If the cell is outside the range of reliable inference from the provided evidence, abstain rather than guessing. Use unknown for final_label and predicted_tissue when abstaining.",
    "If no reliable conclusion can be drawn from the profile, abstain. Set final_label and predicted_tissue to unknown on abstention.",
    "If this cell does not align clearly enough with a confident answer, abstain instead of forcing a call. When abstaining, use unknown for final_label and predicted_tissue.",
    "If the cell is not well explained by the observed evidence, abstain. On abstention, set both final_label and predicted_tissue to unknown.",
    "If there is no credible fit supported by the profile, abstain instead of overcommitting. Use unknown for final_label and predicted_tissue when abstaining.",
    "If the cell falls outside what can be answered reliably from the input, abstain. When abstaining, set final_label and predicted_tissue to unknown.",
    "If the profile does not justify a confident answer, abstain rather than forcing an answer. Use unknown for final_label and predicted_tissue on abstention.",
    "If the evidence does not clearly support a label or tissue assignment, abstain. In that case, set final_label and predicted_tissue to unknown.",
    "If the profile is too uncertain for a reliable conclusion, abstain instead of guessing. When abstaining, use unknown for final_label and predicted_tissue.",
    "If no appropriate answer can be supported by this cell profile, abstain. Set both final_label and predicted_tissue to unknown when abstaining.",
)

MARKER_QUESTION_TEMPLATES = (
    "Identify the marker genes from this ranked single-cell gene-expression profile that best support the top cell annotation, and return the annotation as well.",
    "From this ordered single-cell gene list, report the most informative supporting markers for the best cell-type call and include the annotation.",
    "Use this ranked single-cell profile to name the strongest supporting marker genes and the most likely cell label.",
    "Given this ordered gene-expression signature for one cell, return the best-supported annotation together with the input genes that most justify it.",
    "Read this ranked single-cell expression profile and report both the top annotation and the key supporting marker genes from the input.",
    "Determine the most likely cell label for this one-cell gene ranking and list the marker genes that provide the strongest support.",
    "From this ordered transcriptomic profile, return the most plausible cell identity and the supporting markers drawn from the input genes.",
    "Use the ordered genes in this profile to identify the likely cell type and the marker genes that best justify that choice.",
    "Report the strongest annotation for this single-cell gene list and the marker genes from the input that support it most clearly.",
    "Which cell type and marker genes are best supported by this ranked single-cell expression profile? Return both in JSON.",
)

PROFILE_BUNDLE_QUESTION_TEMPLATES = (
    "You are given {profile_panel}, each represented by an ordered single-cell gene-expression profile. Infer the most likely annotation for each profile and combine the answers into one structured response.",
    "For {profile_panel}, use the ordered gene sets to assign one cell label per profile and synthesize a combined answer across the profiles.",
    "Treat {profile_panel} as separate single-cell profiles. Return the best annotation for each profile and a combined summary grounded in their gene sets.",
    "Based on the ordered gene lists for {profile_panel}, identify the most likely cell type for each profile and provide one combined response.",
    "Use the ranked genes for {profile_panel} to produce per-profile cell labels and a combined summary of the overall answer.",
    "Resolve the cell identity for {profile_panel} using the ordered gene sets, then combine the per-profile answers in one JSON object.",
    "Given the ordered gene-expression profiles for {profile_panel}, return one label per profile and a combined marker summary across the profiles.",
    "Infer the most likely annotation for each of {profile_panel} and synthesize the combined answer from those individual profile-level calls.",
    "Using the ordered genes for {profile_panel}, provide structured per-profile labels and a combined answer that summarizes the profiles together.",
    "Read the ordered gene sets for {profile_panel}, answer each profile at the cell level, and combine those answers into one final structured output.",
)


@dataclass
class DatasetIndex:
    dataset_name: str
    tissue_label: str
    display_tissue: str
    split_paths: dict[str, Path]
    cell_type_counts: Counter = field(default_factory=Counter)
    broad_class_counts: Counter = field(default_factory=Counter)

    @property
    def cell_type_set(self) -> set[str]:
        return set(self.cell_type_counts)

    @property
    def broad_class_set(self) -> set[str]:
        return set(self.broad_class_counts)


@dataclass
class Stage3Sample:
    sample_id: str
    source_sample_id: str
    task_family: str
    split: str
    source_dataset: str
    source_tissue: str
    genes: list[str]
    question_text: str
    context: str
    answer_schema_name: str
    required_output_fields: list[str]
    primary_target: str
    ground_truth: dict
    candidate_experts: list[str]
    oracle_experts: list[str]
    metadata: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Stage3Writer:
    def __init__(self, output_root: Path) -> None:
        self.output_root = ensure_dir(output_root)
        self._family_handles: dict[tuple[str, str], object] = {}
        self._merged_handles: dict[str, object] = {}
        self.family_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.merged_counts: dict[str, int] = defaultdict(int)

    def write_sample(self, sample: Stage3Sample) -> None:
        row = json.dumps(sample.to_dict(), ensure_ascii=False)

        family_key = (sample.task_family, sample.split)
        if family_key not in self._family_handles:
            family_dir = ensure_dir(self.output_root / sample.task_family)
            family_path = family_dir / f"grouped_{sample.split}.jsonl"
            self._family_handles[family_key] = family_path.open("w", encoding="utf-8")
        family_handle = self._family_handles[family_key]
        family_handle.write(row + "\n")
        self.family_counts[sample.task_family][sample.split] += 1

        if sample.split not in self._merged_handles:
            merged_path = self.output_root / f"all_grouped_{sample.split}.jsonl"
            self._merged_handles[sample.split] = merged_path.open("w", encoding="utf-8")
        merged_handle = self._merged_handles[sample.split]
        merged_handle.write(row + "\n")
        self.merged_counts[sample.split] += 1

    def close(self) -> None:
        for handle in self._family_handles.values():
            handle.close()
        for handle in self._merged_handles.values():
            handle.close()

    def build_manifest(self) -> dict:
        manifest = {"families": {}, "merged": {}}
        for family, split_map in self.family_counts.items():
            manifest["families"][family] = {}
            for split, count in split_map.items():
                manifest["families"][family][split] = {
                    "rows": count,
                    "output_path": str(self.output_root / family / f"grouped_{split}.jsonl"),
                }
        for split, count in self.merged_counts.items():
            manifest["merged"][split] = {
                "rows": count,
                "output_path": str(self.output_root / f"all_grouped_{split}.jsonl"),
            }
        return manifest


def _short_tissue_name(value: str | None) -> str:
    if not value:
        return "unknown"
    value = value.strip()
    if value.endswith(" organ"):
        return value[: -len(" organ")]
    return value


def _stable_unique(items: list[str]) -> list[str]:
    seen = set()
    unique = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def _stable_order(items: list[str], seed: str) -> list[str]:
    unique = _stable_unique(items)
    return sorted(unique, key=lambda value: hashlib.sha1(f"{seed}|{value}".encode("utf-8")).hexdigest())


def _question_template_index(selection_seed: str, template_count: int) -> int:
    digest = hashlib.sha1(f"question|{selection_seed}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % template_count


def _question_from_templates(
    templates: tuple[str, ...],
    selection_seed: str,
    **format_kwargs,
) -> str:
    template = templates[_question_template_index(selection_seed, len(templates))]
    return template.format(**format_kwargs) + " " + QUESTION_SUFFIX


def _make_stage3_id(source_sample_id: str, task_family: str, split: str, extra: str = "") -> str:
    digest = hashlib.sha1(f"{source_sample_id}|{task_family}|{split}|{extra}".encode("utf-8")).hexdigest()[:12]
    return f"{task_family}:{split}:{digest}"


def _sanitize_genes(genes: list[str], top_k: int | None = None) -> list[str]:
    sanitized = []
    for gene in genes:
        if gene is None:
            continue
        gene_text = str(gene).strip()
        if not gene_text or gene_text.lower() == "none":
            continue
        sanitized.append(gene_text)
    if top_k is not None:
        return sanitized[:top_k]
    return sanitized


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _context_from_metadata(metadata: dict, keys: tuple[str, ...]) -> str:
    claims = []
    for key in keys:
        value = metadata.get(key)
        if value:
            claims.append(f"{key}={value}")
    return "; ".join(claims)


def _supporting_genes(row: dict, config: PipelineConfig) -> list[str]:
    genes = row.get("evidence_genes") or row.get("genes") or []
    return _sanitize_genes(list(genes), top_k=config.defaults.evidence_gene_count)


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


def _marker_targets(row: dict, config: PipelineConfig) -> list[str]:
    answer_fields = _parse_answer_fields(row.get("answer"))
    genes = set(_sanitize_genes(row.get("genes", [])))
    support_genes = _sanitize_genes(
        answer_fields.get("EVIDENCE", answer_fields.get("POSITIVE_MARKERS", "")).split(",")
    )
    matched_support = [gene for gene in support_genes if gene in genes]
    if matched_support:
        return matched_support[: config.defaults.evidence_gene_count]
    return _supporting_genes(row, config)


def _discover_dataset_dirs(exports_root: Path, dataset_name: str | None = None) -> list[Path]:
    discovered = []
    for entry in sorted(exports_root.iterdir()):
        if not entry.is_dir() or not entry.name.endswith(DIRECTORY_SUFFIX):
            continue
        if dataset_name and entry.name != dataset_name:
            continue
        discovered.append(entry)
    return discovered


def _parse_dataset_tissue(dataset_name: str) -> str:
    prefix = "tabula_sapiens_"
    suffix = "_cell_annotation"
    name = dataset_name
    if name.startswith(prefix):
        name = name[len(prefix) :]
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    return name.replace("_", " ")


def _load_dataset_index(dataset_dir: Path) -> DatasetIndex | None:
    split_paths: dict[str, Path] = {}
    cell_type_counts: Counter = Counter()
    broad_class_counts: Counter = Counter()
    tissue_label: str | None = None

    for split in GROUPED_SPLITS:
        split_path = dataset_dir / f"cell_annotation_rationale_grouped_{split}.jsonl"
        if not split_path.exists():
            continue
        split_paths[split] = split_path
        for row in _iter_jsonl(split_path):
            metadata = row.get("metadata", {})
            cell_type = metadata.get("cell_type")
            broad_class = metadata.get("broad_cell_class")
            if cell_type:
                cell_type_counts[cell_type] += 1
            if broad_class:
                broad_class_counts[broad_class] += 1
            if tissue_label is None:
                tissue_label = metadata.get("tissue") or metadata.get("tissue_general")

    if not split_paths:
        return None

    if not tissue_label:
        tissue_label = _parse_dataset_tissue(dataset_dir.name)

    return DatasetIndex(
        dataset_name=dataset_dir.name,
        tissue_label=tissue_label,
        display_tissue=_short_tissue_name(tissue_label),
        split_paths=split_paths,
        cell_type_counts=cell_type_counts,
        broad_class_counts=broad_class_counts,
    )


def _load_dataset_indices(exports_root: Path, dataset_name: str | None = None) -> dict[str, DatasetIndex]:
    dataset_indices = {}
    for dataset_dir in _discover_dataset_dirs(exports_root, dataset_name=dataset_name):
        index = _load_dataset_index(dataset_dir)
        if index is None:
            continue
        dataset_indices[index.dataset_name] = index
    return dataset_indices


def _clear_stage3_outputs(output_root: Path) -> None:
    if not output_root.exists():
        return
    for file_path in output_root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix not in {".json", ".jsonl"}:
            continue
        file_path.unlink()


def _dataset_overlap_score(left: DatasetIndex, right: DatasetIndex) -> tuple[int, int, str]:
    shared_labels = len(left.cell_type_set & right.cell_type_set)
    shared_broad_classes = len(left.broad_class_set & right.broad_class_set)
    return (shared_labels, shared_broad_classes, right.dataset_name)


def _build_alternative_labels(row: dict, dataset_index: DatasetIndex, config: PipelineConfig) -> list[str]:
    metadata = row.get("metadata", {})
    gold_label = metadata.get("cell_type")
    alternatives = []
    for label in candidate_labels_for_gold(gold_label, metadata):
        if label != gold_label:
            alternatives.append(label)
    for label, _count in dataset_index.cell_type_counts.most_common():
        if label != gold_label:
            alternatives.append(label)
    max_count = max(1, config.defaults.differential_candidate_count - 1)
    return _stable_unique(alternatives)[:max_count]


def _build_differential_panel(row: dict, dataset_index: DatasetIndex, config: PipelineConfig) -> list[str]:
    metadata = row.get("metadata", {})
    gold_label = metadata.get("cell_type")
    if not gold_label:
        return []
    panel = [gold_label] + _build_alternative_labels(row, dataset_index, config)
    max_count = max(2, config.defaults.differential_candidate_count)
    return _stable_order(panel, row.get("sample_id", gold_label))[:max_count]


def _select_cross_experts(
    row: dict,
    source_dataset: str,
    dataset_indices: dict[str, DatasetIndex],
    max_experts: int = 4,
) -> tuple[list[str], list[str]]:
    metadata = row.get("metadata", {})
    gold_label = metadata.get("cell_type")
    broad_class = metadata.get("broad_cell_class")
    source_index = dataset_indices[source_dataset]

    label_matches = []
    broad_matches = []
    fallback = []
    for dataset_name, dataset_index in dataset_indices.items():
        if dataset_name == source_dataset:
            continue
        if gold_label and gold_label in dataset_index.cell_type_set:
            label_matches.append(dataset_name)
            continue
        if broad_class and broad_class in dataset_index.broad_class_set:
            broad_matches.append(dataset_name)
            continue
        fallback.append(dataset_name)

    fallback = [
        dataset_name
        for dataset_name, _shared_labels, _shared_broad, _name in sorted(
            (
                (candidate, *_dataset_overlap_score(source_index, dataset_indices[candidate]))
                for candidate in fallback
            ),
            key=lambda item: (-item[1], -item[2], item[0]),
        )
    ]

    seed = row.get("sample_id", source_dataset)
    ordered_related = _stable_unique(label_matches + broad_matches + fallback)
    ordered_related = _stable_order(ordered_related, seed)
    candidate_experts = [source_dataset] + ordered_related[: max_experts - 1]
    oracle_experts = []
    for dataset_name in candidate_experts:
        if dataset_name == source_dataset:
            oracle_experts.append(dataset_name)
            continue
        dataset_index = dataset_indices[dataset_name]
        if gold_label and gold_label in dataset_index.cell_type_set:
            oracle_experts.append(dataset_name)
            continue
        if broad_class and broad_class in dataset_index.broad_class_set:
            oracle_experts.append(dataset_name)
    return candidate_experts, _stable_unique(oracle_experts)


def _select_ood_experts(
    row: dict,
    source_dataset: str,
    dataset_indices: dict[str, DatasetIndex],
    count: int = 3,
) -> list[str]:
    metadata = row.get("metadata", {})
    gold_label = metadata.get("cell_type")
    broad_class = metadata.get("broad_cell_class")
    if not gold_label and not broad_class:
        return []

    seed = row.get("sample_id", source_dataset)
    candidates = []
    for dataset_name, dataset_index in dataset_indices.items():
        if dataset_name == source_dataset:
            continue
        if gold_label and gold_label in dataset_index.cell_type_set:
            continue
        if broad_class and broad_class in dataset_index.broad_class_set:
            continue
        candidates.append(dataset_name)
    ordered = _stable_order(candidates, seed)
    return ordered[:count]


def _question_for_annotation(selection_seed: str) -> str:
    return _question_from_templates(ANNOTATION_QUESTION_TEMPLATES, selection_seed)


def _question_for_tissue(selection_seed: str) -> str:
    return _question_from_templates(TISSUE_QUESTION_TEMPLATES, selection_seed)


def _question_for_differential(candidate_panel: list[str], selection_seed: str) -> str:
    panel_text = ", ".join(candidate_panel)
    return _question_from_templates(
        DIFFERENTIAL_QUESTION_TEMPLATES,
        selection_seed,
        candidate_panel=panel_text,
    )


def _question_for_cross_expert(
    selection_seed: str,
) -> str:
    return _question_from_templates(CROSS_EXPERT_QUESTION_TEMPLATES, selection_seed)


def _question_for_ood(
    selection_seed: str,
) -> str:
    return _question_from_templates(OOD_QUESTION_TEMPLATES, selection_seed)


def _question_for_marker(selection_seed: str) -> str:
    return _question_from_templates(MARKER_QUESTION_TEMPLATES, selection_seed)


def _profile_panel_text(profile_ids: tuple[str, ...]) -> str:
    if len(profile_ids) == 2:
        return f"profiles {profile_ids[0]} and {profile_ids[1]}"
    return f"profiles {', '.join(profile_ids[:-1])}, and {profile_ids[-1]}"


def _question_for_profile_bundle(profile_ids: tuple[str, ...], selection_seed: str) -> str:
    template = PROFILE_BUNDLE_QUESTION_TEMPLATES[_question_template_index(selection_seed, len(PROFILE_BUNDLE_QUESTION_TEMPLATES))]
    return template.format(profile_panel=_profile_panel_text(profile_ids)) + " " + PROFILE_BUNDLE_SUFFIX


def _bundle_profile_count(selection_seed: str) -> int:
    digest = hashlib.sha1(f"bundle|{selection_seed}".encode("utf-8")).hexdigest()
    return 2 + (int(digest[:2], 16) % 2)


def _select_profile_bundle_rows(rows: list[dict], start_index: int, desired_count: int, top_genes: int) -> list[dict]:
    selected = []
    for row in rows[start_index:]:
        metadata = row.get("metadata", {})
        genes = _sanitize_genes(row.get("genes", []), top_k=top_genes)
        if not metadata.get("cell_type") or not genes:
            continue
        selected.append(row)
        if len(selected) >= desired_count:
            break
    if len(selected) < 2:
        return []
    return selected


def _profile_bundle_context(profile_rows: list[dict], profile_ids: tuple[str, ...], config: PipelineConfig) -> str:
    lines = []
    for profile_id, row in zip(profile_ids, profile_rows):
        genes = _sanitize_genes(row.get("genes", []), top_k=config.defaults.top_genes)
        metadata = row.get("metadata", {})
        lines.append(f"Profile {profile_id} ordered genes: {', '.join(genes)}")
        nonleak_context = _context_from_metadata(metadata, NONLEAK_CONTEXT_KEYS)
        if nonleak_context:
            lines.append(f"Profile {profile_id} context: {nonleak_context}")
    return "\n".join(lines)


def _build_annotation_sample(row: dict, split: str, dataset_index: DatasetIndex, config: PipelineConfig) -> Stage3Sample | None:
    metadata = row.get("metadata", {})
    gold_label = metadata.get("cell_type")
    tissue_label = _short_tissue_name(metadata.get("tissue") or dataset_index.tissue_label)
    genes = _sanitize_genes(row.get("genes", []), top_k=config.defaults.top_genes)
    if not gold_label or not genes:
        return None

    alternatives = _build_alternative_labels(row, dataset_index, config)
    supporting_genes = _supporting_genes(row, config)
    context = row.get("context") or _context_from_metadata(metadata, DIRECT_CONTEXT_KEYS)
    sample_id = _make_stage3_id(row["sample_id"], "annotation_direct", split)
    return Stage3Sample(
        sample_id=sample_id,
        source_sample_id=row["sample_id"],
        task_family="annotation_direct",
        split=split,
        source_dataset=dataset_index.dataset_name,
        source_tissue=tissue_label,
        genes=genes,
        question_text=_question_for_annotation(sample_id),
        context=context,
        answer_schema_name=ANSWER_SCHEMA_NAME,
        required_output_fields=list(REQUIRED_OUTPUT_FIELDS),
        primary_target="final_label",
        ground_truth={
            "final_label": gold_label,
            "alternative_labels": alternatives,
            "predicted_tissue": tissue_label,
            "supporting_genes": supporting_genes,
            "abstain": False,
        },
        candidate_experts=[dataset_index.dataset_name],
        oracle_experts=[dataset_index.dataset_name],
        metadata={
            "candidate_labels": [gold_label] + alternatives,
            "candidate_expert_tissues": [dataset_index.display_tissue],
            "broad_cell_class": metadata.get("broad_cell_class") or "",
            "compartment": metadata.get("compartment") or "",
        },
        provenance={
            "builder": "stage3_annotation_direct",
            "source_split": split,
            "source_task": "cell_annotation_rationale_grouped",
        },
    )


def _build_tissue_sample(row: dict, split: str, dataset_index: DatasetIndex, config: PipelineConfig) -> Stage3Sample | None:
    metadata = row.get("metadata", {})
    tissue_label = _short_tissue_name(metadata.get("tissue") or dataset_index.tissue_label)
    genes = _sanitize_genes(row.get("genes", []), top_k=config.defaults.top_genes)
    if not tissue_label or not genes:
        return None

    supporting_genes = _supporting_genes(row, config)
    context = _context_from_metadata(metadata, tuple(key for key in DIRECT_CONTEXT_KEYS if key != "tissue"))
    sample_id = _make_stage3_id(row["sample_id"], "tissue_inference_direct", split)
    return Stage3Sample(
        sample_id=sample_id,
        source_sample_id=row["sample_id"],
        task_family="tissue_inference_direct",
        split=split,
        source_dataset=dataset_index.dataset_name,
        source_tissue=tissue_label,
        genes=genes,
        question_text=_question_for_tissue(sample_id),
        context=context,
        answer_schema_name=ANSWER_SCHEMA_NAME,
        required_output_fields=list(REQUIRED_OUTPUT_FIELDS),
        primary_target="predicted_tissue",
        ground_truth={
            "final_label": tissue_label,
            "alternative_labels": [],
            "predicted_tissue": tissue_label,
            "supporting_genes": supporting_genes,
            "abstain": False,
        },
        candidate_experts=[dataset_index.dataset_name],
        oracle_experts=[dataset_index.dataset_name],
        metadata={
            "candidate_expert_tissues": [dataset_index.display_tissue],
            "broad_cell_class": metadata.get("broad_cell_class") or "",
            "compartment": metadata.get("compartment") or "",
        },
        provenance={
            "builder": "stage3_tissue_inference_direct",
            "source_split": split,
            "source_task": "cell_annotation_rationale_grouped",
        },
    )


def _build_differential_sample(row: dict, split: str, dataset_index: DatasetIndex, config: PipelineConfig) -> Stage3Sample | None:
    metadata = row.get("metadata", {})
    gold_label = metadata.get("cell_type")
    tissue_label = _short_tissue_name(metadata.get("tissue") or dataset_index.tissue_label)
    genes = _sanitize_genes(row.get("genes", []), top_k=config.defaults.top_genes)
    if not gold_label or not genes:
        return None

    candidate_panel = _build_differential_panel(row, dataset_index, config)
    if len(candidate_panel) < 2:
        return None

    alternatives = [label for label in candidate_panel if label != gold_label]
    supporting_genes = _supporting_genes(row, config)
    context = row.get("context") or _context_from_metadata(metadata, DIRECT_CONTEXT_KEYS)
    sample_id = _make_stage3_id(row["sample_id"], "differential_diagnosis_direct", split, extra="|".join(candidate_panel))
    return Stage3Sample(
        sample_id=sample_id,
        source_sample_id=row["sample_id"],
        task_family="differential_diagnosis_direct",
        split=split,
        source_dataset=dataset_index.dataset_name,
        source_tissue=tissue_label,
        genes=genes,
        question_text=_question_for_differential(candidate_panel, sample_id),
        context=context,
        answer_schema_name=ANSWER_SCHEMA_NAME,
        required_output_fields=list(REQUIRED_OUTPUT_FIELDS),
        primary_target="final_label",
        ground_truth={
            "final_label": gold_label,
            "alternative_labels": alternatives,
            "predicted_tissue": tissue_label,
            "supporting_genes": supporting_genes,
            "abstain": False,
        },
        candidate_experts=[dataset_index.dataset_name],
        oracle_experts=[dataset_index.dataset_name],
        metadata={
            "candidate_labels": candidate_panel,
            "candidate_expert_tissues": [dataset_index.display_tissue],
            "broad_cell_class": metadata.get("broad_cell_class") or "",
            "compartment": metadata.get("compartment") or "",
        },
        provenance={
            "builder": "stage3_differential_diagnosis_direct",
            "source_split": split,
            "source_task": "cell_annotation_rationale_grouped",
        },
    )


def _build_marker_sample(row: dict, split: str, dataset_index: DatasetIndex, config: PipelineConfig) -> Stage3Sample | None:
    metadata = row.get("metadata", {})
    gold_label = metadata.get("cell_type")
    tissue_label = _short_tissue_name(metadata.get("tissue") or dataset_index.tissue_label)
    genes = _sanitize_genes(row.get("genes", []), top_k=config.defaults.top_genes)
    if not gold_label or not genes:
        return None

    alternatives = _build_alternative_labels(row, dataset_index, config)
    supporting_genes = _marker_targets(row, config)
    context = row.get("context") or _context_from_metadata(metadata, DIRECT_CONTEXT_KEYS)
    sample_id = _make_stage3_id(row["sample_id"], "marker_support_direct", split)
    return Stage3Sample(
        sample_id=sample_id,
        source_sample_id=row["sample_id"],
        task_family="marker_support_direct",
        split=split,
        source_dataset=dataset_index.dataset_name,
        source_tissue=tissue_label,
        genes=genes,
        question_text=_question_for_marker(sample_id),
        context=context,
        answer_schema_name=ANSWER_SCHEMA_NAME,
        required_output_fields=list(REQUIRED_OUTPUT_FIELDS),
        primary_target="supporting_genes",
        ground_truth={
            "final_label": gold_label,
            "alternative_labels": alternatives,
            "predicted_tissue": tissue_label,
            "supporting_genes": supporting_genes,
            "abstain": False,
        },
        candidate_experts=[dataset_index.dataset_name],
        oracle_experts=[dataset_index.dataset_name],
        metadata={
            "candidate_labels": [gold_label] + alternatives,
            "candidate_expert_tissues": [dataset_index.display_tissue],
            "broad_cell_class": metadata.get("broad_cell_class") or "",
            "compartment": metadata.get("compartment") or "",
        },
        provenance={
            "builder": "stage3_marker_support_direct",
            "source_split": split,
            "source_task": "cell_annotation_rationale_grouped",
        },
    )


def _build_profile_bundle_sample(
    rows: list[dict],
    row_index: int,
    split: str,
    dataset_index: DatasetIndex,
    config: PipelineConfig,
) -> Stage3Sample | None:
    seed_row = rows[row_index]
    desired_count = _bundle_profile_count(seed_row.get("sample_id", f"{dataset_index.dataset_name}:{row_index}"))
    profile_rows = _select_profile_bundle_rows(rows, row_index, desired_count, config.defaults.top_genes)
    if len(profile_rows) < 2:
        return None

    profile_ids = tuple(chr(ord("A") + index) for index in range(len(profile_rows)))
    profile_labels = {}
    profile_supporting_genes = {}
    combined_markers = []
    profile_gene_sets = []
    for profile_id, row in zip(profile_ids, profile_rows):
        metadata = row.get("metadata", {})
        genes = _sanitize_genes(row.get("genes", []), top_k=config.defaults.top_genes)
        label = metadata.get("cell_type")
        if not label or not genes:
            return None
        supporting_genes = _marker_targets(row, config)
        profile_labels[profile_id] = label
        profile_supporting_genes[profile_id] = supporting_genes
        combined_markers.extend(supporting_genes)
        profile_gene_sets.append(
            {
                "profile_id": profile_id,
                "source_sample_id": row["sample_id"],
                "genes": genes,
            }
        )

    sample_id = _make_stage3_id(
        seed_row["sample_id"],
        "profile_bundle_synthesis",
        split,
        extra="|".join(profile_labels.values()),
    )
    combined_summary = "; ".join(f"Profile {profile_id}={profile_labels[profile_id]}" for profile_id in profile_ids)
    return Stage3Sample(
        sample_id=sample_id,
        source_sample_id=seed_row["sample_id"],
        task_family="profile_bundle_synthesis",
        split=split,
        source_dataset=dataset_index.dataset_name,
        source_tissue=dataset_index.display_tissue,
        genes=[],
        question_text=_question_for_profile_bundle(profile_ids, sample_id),
        context=_profile_bundle_context(profile_rows, profile_ids, config),
        answer_schema_name=PROFILE_BUNDLE_ANSWER_SCHEMA_NAME,
        required_output_fields=list(PROFILE_BUNDLE_REQUIRED_OUTPUT_FIELDS),
        primary_target="profile_labels",
        ground_truth={
            "profile_labels": profile_labels,
            "profile_supporting_genes": profile_supporting_genes,
            "combined_markers": _stable_unique(combined_markers),
            "combined_summary": combined_summary,
            "abstain": False,
        },
        candidate_experts=[dataset_index.dataset_name],
        oracle_experts=[dataset_index.dataset_name],
        metadata={
            "profile_gene_sets": profile_gene_sets,
            "profile_count": len(profile_ids),
            "candidate_expert_tissues": [dataset_index.display_tissue],
        },
        provenance={
            "builder": "stage3_profile_bundle_synthesis",
            "source_split": split,
            "source_task": "cell_annotation_rationale_grouped",
            "source_sample_ids": [row["sample_id"] for row in profile_rows],
        },
    )


def _build_cross_expert_sample(
    row: dict,
    split: str,
    dataset_index: DatasetIndex,
    dataset_indices: dict[str, DatasetIndex],
    config: PipelineConfig,
) -> Stage3Sample | None:
    metadata = row.get("metadata", {})
    gold_label = metadata.get("cell_type")
    tissue_label = _short_tissue_name(metadata.get("tissue") or dataset_index.tissue_label)
    genes = _sanitize_genes(row.get("genes", []), top_k=config.defaults.top_genes)
    if not gold_label or not genes:
        return None

    candidate_experts, oracle_experts = _select_cross_experts(row, dataset_index.dataset_name, dataset_indices)
    if len(candidate_experts) < 2:
        return None

    alternatives = _build_alternative_labels(row, dataset_index, config)
    supporting_genes = _supporting_genes(row, config)
    context = _context_from_metadata(metadata, NONLEAK_CONTEXT_KEYS)
    sample_id = _make_stage3_id(row["sample_id"], "cross_expert_annotation", split, extra="|".join(candidate_experts))
    return Stage3Sample(
        sample_id=sample_id,
        source_sample_id=row["sample_id"],
        task_family="cross_expert_annotation",
        split=split,
        source_dataset=dataset_index.dataset_name,
        source_tissue=tissue_label,
        genes=genes,
        question_text=_question_for_cross_expert(sample_id),
        context=context,
        answer_schema_name=ANSWER_SCHEMA_NAME,
        required_output_fields=list(REQUIRED_OUTPUT_FIELDS),
        primary_target="final_label",
        ground_truth={
            "final_label": gold_label,
            "alternative_labels": alternatives,
            "predicted_tissue": tissue_label,
            "supporting_genes": supporting_genes,
            "abstain": False,
        },
        candidate_experts=candidate_experts,
        oracle_experts=oracle_experts,
        metadata={
            "candidate_labels": [gold_label] + alternatives,
            "candidate_expert_tissues": [dataset_indices[expert].display_tissue for expert in candidate_experts],
            "oracle_tissues": [dataset_indices[expert].display_tissue for expert in oracle_experts],
            "broad_cell_class": metadata.get("broad_cell_class") or "",
            "compartment": metadata.get("compartment") or "",
        },
        provenance={
            "builder": "stage3_cross_expert_annotation",
            "source_split": split,
            "source_task": "cell_annotation_rationale_grouped",
            "oracle_expert_rule": "source_tissue_plus_shared_label_or_broad_class",
        },
    )


def _build_ood_sample(
    row: dict,
    split: str,
    dataset_index: DatasetIndex,
    dataset_indices: dict[str, DatasetIndex],
    config: PipelineConfig,
) -> Stage3Sample | None:
    metadata = row.get("metadata", {})
    tissue_label = _short_tissue_name(metadata.get("tissue") or dataset_index.tissue_label)
    genes = _sanitize_genes(row.get("genes", []), top_k=config.defaults.top_genes)
    if not genes:
        return None

    candidate_experts = _select_ood_experts(row, dataset_index.dataset_name, dataset_indices)
    if len(candidate_experts) < 3:
        return None

    supporting_genes = _supporting_genes(row, config)
    context = _context_from_metadata(metadata, NONLEAK_CONTEXT_KEYS)
    sample_id = _make_stage3_id(row["sample_id"], "ood_abstain", split, extra="|".join(candidate_experts))
    return Stage3Sample(
        sample_id=sample_id,
        source_sample_id=row["sample_id"],
        task_family="ood_abstain",
        split=split,
        source_dataset=dataset_index.dataset_name,
        source_tissue=tissue_label,
        genes=genes,
        question_text=_question_for_ood(sample_id),
        context=context,
        answer_schema_name=ANSWER_SCHEMA_NAME,
        required_output_fields=list(REQUIRED_OUTPUT_FIELDS),
        primary_target="abstain",
        ground_truth={
            "final_label": "unknown",
            "alternative_labels": [],
            "predicted_tissue": "unknown",
            "supporting_genes": supporting_genes,
            "abstain": True,
        },
        candidate_experts=candidate_experts,
        oracle_experts=[],
        metadata={
            "candidate_expert_tissues": [dataset_indices[expert].display_tissue for expert in candidate_experts],
            "ood_source_label": metadata.get("cell_type") or "",
            "ood_source_tissue": tissue_label,
            "broad_cell_class": metadata.get("broad_cell_class") or "",
        },
        provenance={
            "builder": "stage3_ood_abstain",
            "source_split": split,
            "source_task": "cell_annotation_rationale_grouped",
            "oracle_expert_rule": "no_candidate_expert_contains_gold_label_or_broad_class",
        },
    )


def _validate_sample(sample: Stage3Sample, known_experts: set[str], seen_ids: set[str]) -> None:
    failures = []
    if sample.sample_id in seen_ids:
        failures.append(f"duplicate_sample_id={sample.sample_id}")
    seen_ids.add(sample.sample_id)

    if not set(sample.candidate_experts).issubset(known_experts):
        failures.append(f"unknown_expert_in_candidates={sample.sample_id}")
    if not set(sample.oracle_experts).issubset(set(sample.candidate_experts)):
        failures.append(f"oracle_not_subset_of_candidates={sample.sample_id}")

    gene_set = set(sample.genes)
    for gene in sample.ground_truth.get("supporting_genes", []):
        if gene not in gene_set:
            failures.append(f"support_gene_missing_from_input={sample.sample_id}:{gene}")

    if sample.task_family == "cross_expert_annotation" and len(sample.candidate_experts) < 2:
        failures.append(f"cross_expert_missing_candidates={sample.sample_id}")
    if sample.task_family == "ood_abstain":
        if sample.oracle_experts:
            failures.append(f"ood_should_not_have_oracle_experts={sample.sample_id}")
        if not sample.ground_truth.get("abstain"):
            failures.append(f"ood_should_abstain={sample.sample_id}")
    if sample.task_family == "profile_bundle_synthesis":
        profile_gene_sets = sample.metadata.get("profile_gene_sets") or []
        if len(profile_gene_sets) < 2:
            failures.append(f"profile_bundle_requires_multiple_gene_sets={sample.sample_id}")
        genes_by_profile = {
            item.get("profile_id"): set(item.get("genes") or [])
            for item in profile_gene_sets
        }
        for profile_id, markers in (sample.ground_truth.get("profile_supporting_genes") or {}).items():
            available = genes_by_profile.get(profile_id, set())
            for marker in markers:
                if marker not in available:
                    failures.append(f"profile_support_gene_missing_from_input={sample.sample_id}:{profile_id}:{marker}")
        combined_available = set().union(*genes_by_profile.values()) if genes_by_profile else set()
        for marker in sample.ground_truth.get("combined_markers", []):
            if marker not in combined_available:
                failures.append(f"combined_marker_missing_from_bundle_input={sample.sample_id}:{marker}")

    if failures:
        preview = "; ".join(failures[:10])
        raise ValueError(f"Stage-3 validation failed: {preview}")


def build_stage3_dataset(
    config: PipelineConfig,
    *,
    dataset_name: str | None = None,
    exports_root: str | Path | None = None,
    output_root: str | Path | None = None,
    limit_per_split: int | None = None,
) -> dict:
    resolved_exports_root = Path(exports_root) if exports_root else config.paths.exports_dir
    resolved_output_root = Path(output_root) if output_root else (config.paths.data_dir / "stage3_exports")

    dataset_indices = _load_dataset_indices(resolved_exports_root, dataset_name=dataset_name)
    if not dataset_indices:
        raise ValueError(f"No grouped tissue exports found under {resolved_exports_root}")

    _clear_stage3_outputs(resolved_output_root)
    writer = Stage3Writer(resolved_output_root)
    source_tissue_counts: Counter = Counter()
    known_experts = set(dataset_indices)
    seen_ids: set[str] = set()
    family_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def maybe_write(sample: Stage3Sample | None) -> None:
        if sample is None:
            return
        current_count = family_counts[sample.task_family][sample.split]
        if limit_per_split is not None and current_count >= limit_per_split:
            return
        _validate_sample(sample, known_experts, seen_ids)
        writer.write_sample(sample)
        family_counts[sample.task_family][sample.split] += 1
        source_tissue_counts[sample.source_tissue] += 1

    def all_limits_reached() -> bool:
        if limit_per_split is None:
            return False
        expected_families = {
            "annotation_direct",
            "tissue_inference_direct",
            "differential_diagnosis_direct",
            "marker_support_direct",
            "profile_bundle_synthesis",
            "cross_expert_annotation",
            "ood_abstain",
        }
        for family in expected_families:
            for split in GROUPED_SPLITS:
                if family_counts[family][split] < limit_per_split:
                    return False
        return True

    try:
        for dataset_index in dataset_indices.values():
            for split in GROUPED_SPLITS:
                split_path = dataset_index.split_paths.get(split)
                if split_path is None:
                    continue
                rows = list(_iter_jsonl(split_path))
                for row_index, row in enumerate(rows):
                    maybe_write(_build_annotation_sample(row, split, dataset_index, config))
                    maybe_write(_build_tissue_sample(row, split, dataset_index, config))
                    maybe_write(_build_differential_sample(row, split, dataset_index, config))
                    maybe_write(_build_marker_sample(row, split, dataset_index, config))
                    maybe_write(_build_profile_bundle_sample(rows, row_index, split, dataset_index, config))
                    maybe_write(_build_cross_expert_sample(row, split, dataset_index, dataset_indices, config))
                    maybe_write(_build_ood_sample(row, split, dataset_index, dataset_indices, config))
                    if all_limits_reached():
                        break
                if all_limits_reached():
                    break
            if all_limits_reached():
                break
    finally:
        writer.close()

    manifest = writer.build_manifest()
    schema_path = resolved_output_root / "answer_schema.json"
    write_json(
        schema_path,
        {
            "schemas": {
                ANSWER_SCHEMA_NAME: {
                    "required_output_fields": REQUIRED_OUTPUT_FIELDS,
                    "notes": {
                        "final_label": "Primary answer label for the task family.",
                        "alternative_labels": "Ranked alternatives when uncertainty remains.",
                        "predicted_tissue": "Most likely tissue context or unknown.",
                        "supporting_genes": "Genes drawn from the input only.",
                        "confidence": "Model-calibrated confidence field reserved for downstream scoring.",
                        "abstain": "True when the model should refuse to force an answer.",
                    },
                },
                PROFILE_BUNDLE_ANSWER_SCHEMA_NAME: {
                    "required_output_fields": PROFILE_BUNDLE_REQUIRED_OUTPUT_FIELDS,
                    "notes": {
                        "profile_labels": "Object keyed by profile ID with the best cell label for each gene set.",
                        "profile_supporting_genes": "Object keyed by profile ID with supporting genes from that profile's input only.",
                        "combined_markers": "Deduplicated markers synthesized across the profiles.",
                        "combined_summary": "Compact textual combination of the per-profile answers.",
                        "abstain": "True when the model should refuse to force an answer.",
                    },
                },
            },
        },
    )
    manifest["answer_schema_path"] = str(schema_path)
    write_json(resolved_output_root / "manifest.json", manifest)

    summary = {
        "exports_root": str(resolved_exports_root),
        "output_root": str(resolved_output_root),
        "dataset_count": len(dataset_indices),
        "datasets": sorted(dataset_indices),
        "counts_by_family": {family: dict(split_map) for family, split_map in family_counts.items()},
        "source_tissue_counts": dict(source_tissue_counts),
        "total_rows": sum(sum(split_map.values()) for split_map in family_counts.values()),
        "manifest_path": str(resolved_output_root / "manifest.json"),
    }
    write_json(resolved_output_root / "summary.json", summary)
    return summary