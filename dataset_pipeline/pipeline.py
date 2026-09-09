from pathlib import Path

from .builders import build_examples
from .catalog import enabled_datasets, get_dataset_spec
from .census_source import audit_h5ad_metadata, download_dataset
from .config import PipelineConfig
from .export import export_intermediate, export_stage2, export_verification_report
from .io_utils import ensure_dir, write_json
from .normalize import normalize_h5ad
from .verifier import verify_examples


def dataset_raw_path(config: PipelineConfig, dataset_name: str) -> Path:
    return config.paths.raw_dir / f"{dataset_name}.h5ad"


def dataset_normalized_path(config: PipelineConfig, dataset_name: str) -> Path:
    return config.paths.normalized_dir / f"{dataset_name}.normalized.jsonl"


def dataset_examples_path(config: PipelineConfig, dataset_name: str) -> Path:
    return config.paths.normalized_dir / f"{dataset_name}.examples.jsonl"


def dataset_report_path(config: PipelineConfig, dataset_name: str) -> Path:
    return config.paths.reports_dir / f"{dataset_name}.verification.json"


def download_from_catalog(config: PipelineConfig, dataset_name: str | None = None) -> list[Path]:
    ensure_dir(config.paths.raw_dir)
    paths = []
    for spec in enabled_datasets(config):
        if dataset_name and spec.name != dataset_name:
            continue
        output_path = dataset_raw_path(config, spec.name)
        download_dataset(spec, output_path, max_cells=config.defaults.max_cells_per_dataset)
        paths.append(output_path)
    return paths


def audit_downloads(config: PipelineConfig, dataset_name: str | None = None) -> dict:
    ensure_dir(config.paths.reports_dir)
    report = {"datasets": {}}
    for spec in enabled_datasets(config):
        if dataset_name and spec.name != dataset_name:
            continue
        raw_path = dataset_raw_path(config, spec.name)
        report["datasets"][spec.name] = audit_h5ad_metadata(
            raw_path,
            required_columns=spec.required_obs_columns,
            state_label_keys=config.defaults.state_label_keys,
        )
    return report


def build_verified_examples(config: PipelineConfig, dataset_name: str | None = None) -> dict:
    ensure_dir(config.paths.normalized_dir)
    ensure_dir(config.paths.exports_dir)
    ensure_dir(config.paths.reports_dir)

    summary = {"datasets": {}}
    for spec in enabled_datasets(config):
        if dataset_name and spec.name != dataset_name:
            continue

        raw_path = dataset_raw_path(config, spec.name)
        records = normalize_h5ad(raw_path, spec, config)
        examples = build_examples(records, config, spec.task_types)
        verified_examples, verification_results = verify_examples(examples, config)

        export_intermediate(verified_examples, dataset_examples_path(config, spec.name))
        stage2_manifest = export_stage2(verified_examples, config.paths.exports_dir / spec.name)
        verification_payload = export_verification_report(verification_results, dataset_report_path(config, spec.name))

        summary["datasets"][spec.name] = {
            "raw_path": str(raw_path),
            "records": len(records),
            "examples_before_verification": len(examples),
            "examples_after_verification": len(verified_examples),
            "stage2_manifest": stage2_manifest,
            "verification": {
                "total_rows": verification_payload["total_rows"],
                "passed_rows": verification_payload["passed_rows"],
                "failed_rows": verification_payload["failed_rows"],
                "field_failures": verification_payload["field_failures"],
            },
        }

    write_json(config.paths.reports_dir / "pipeline_summary.json", summary)
    return summary