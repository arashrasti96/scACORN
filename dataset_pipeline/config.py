import json
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_STATE_KEYS = [
    "cell_state",
    "state",
    "activation_state",
    "disease_state",
    "condition",
    "status",
]


@dataclass
class PathsConfig:
    root: Path
    data_dir: Path
    raw_dir: Path
    normalized_dir: Path
    exports_dir: Path
    reports_dir: Path


@dataclass
class DefaultsConfig:
    organism: str = "homo_sapiens"
    max_cells_per_dataset: int = 50000
    top_genes: int = 200
    evidence_gene_count: int = 5
    differential_candidate_count: int = 4
    min_label_frequency: int = 25
    state_label_keys: list[str] = field(default_factory=lambda: list(DEFAULT_STATE_KEYS))
    context_allowlist: list[str] = field(
        default_factory=lambda: [
            "organism",
            "tissue",
            "tissue_general",
            "cell_type",
            "cell_type_ontology_term_id",
            "tissue_ontology_term_id",
            "disease",
            "disease_ontology_term_id",
            "assay",
            "assay_ontology_term_id",
            "donor_id",
            "sex",
            "development_stage",
            "suspension_type",
        ]
    )


@dataclass
class DatasetSpec:
    name: str
    source: str
    enabled: bool = True
    census_version: str = "latest"
    organism: str = "homo_sapiens"
    dataset_id: str | None = None
    collection_name_contains: str | None = None
    dataset_title_contains: str | None = None
    obs_value_filter: str | None = None
    obs_columns: list[str] = field(default_factory=list)
    task_types: list[str] = field(default_factory=list)
    required_obs_columns: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class PipelineConfig:
    paths: PathsConfig
    defaults: DefaultsConfig
    datasets: list[DatasetSpec]


def _resolve_paths(root: Path, payload: dict) -> PathsConfig:
    paths = payload.get("paths", {})
    data_dir = root / paths.get("data_dir", "data")
    raw_dir = root / paths.get("raw_dir", "data/raw")
    normalized_dir = root / paths.get("normalized_dir", "data/normalized")
    exports_dir = root / paths.get("exports_dir", "data/exports")
    reports_dir = root / paths.get("reports_dir", "data/reports")
    return PathsConfig(
        root=root,
        data_dir=data_dir,
        raw_dir=raw_dir,
        normalized_dir=normalized_dir,
        exports_dir=exports_dir,
        reports_dir=reports_dir,
    )


def load_catalog(path: str | Path) -> PipelineConfig:
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    defaults_payload = payload.get("defaults", {})
    defaults = DefaultsConfig(
        organism=defaults_payload.get("organism", "homo_sapiens"),
        max_cells_per_dataset=int(defaults_payload.get("max_cells_per_dataset", 50000)),
        top_genes=int(defaults_payload.get("top_genes", 200)),
        evidence_gene_count=int(defaults_payload.get("evidence_gene_count", 5)),
        differential_candidate_count=int(defaults_payload.get("differential_candidate_count", 4)),
        min_label_frequency=int(defaults_payload.get("min_label_frequency", 25)),
        state_label_keys=list(defaults_payload.get("state_label_keys", DEFAULT_STATE_KEYS)),
        context_allowlist=list(defaults_payload.get("context_allowlist", DefaultsConfig().context_allowlist)),
    )

    datasets = [DatasetSpec(**item) for item in payload.get("datasets", [])]
    return PipelineConfig(
        paths=_resolve_paths(path.parent, payload),
        defaults=defaults,
        datasets=datasets,
    )