import argparse
import json
from pathlib import Path

from .catalog import enabled_datasets
from .census_source import search_census_datasets
from .config import load_catalog
from .gene_only_benchmark_creator.audit import main as audit_gene_only_benchmark_main
from .gene_only_benchmark_creator.cli import main as build_gene_only_benchmark_main
from .io_utils import write_json
from .pipeline import audit_downloads, build_verified_examples, download_from_catalog
from .runtime import ensure_h5ad_runtime, runtime_report
from .stage3_benchmark import build_stage3_dataset


def _catalog_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--catalog",
        type=str,
        default=str(Path(__file__).resolve().parent / "approved_datasets.json"),
        help="Path to the standalone dataset catalog JSON file.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone CELLxGENE dataset pipeline for stage-2 task-adapter data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search-census", help="Search candidate CELLxGENE datasets.")
    search_parser.add_argument("--collection-name-contains", type=str, default=None)
    search_parser.add_argument("--dataset-title-contains", type=str, default=None)
    search_parser.add_argument("--census-version", type=str, default="latest")

    subparsers.add_parser("doctor", help="Report interpreter and H5AD runtime health.")

    list_parser = subparsers.add_parser("list-approved", help="List approved datasets from the local catalog.")
    _catalog_arg(list_parser)

    download_parser = subparsers.add_parser("download", help="Download approved CELLxGENE datasets into the standalone raw directory.")
    _catalog_arg(download_parser)
    download_parser.add_argument("--dataset", type=str, default=None, help="Optional dataset name from the catalog.")

    audit_parser = subparsers.add_parser("audit", help="Audit downloaded datasets for task metadata coverage.")
    _catalog_arg(audit_parser)
    audit_parser.add_argument("--dataset", type=str, default=None, help="Optional dataset name from the catalog.")

    build_parser = subparsers.add_parser("build", help="Normalize, build, verify, and export datasets.")
    _catalog_arg(build_parser)
    build_parser.add_argument("--dataset", type=str, default=None, help="Optional dataset name from the catalog.")

    stage3_parser = subparsers.add_parser(
        "build-stage3",
        help="Build grouped stage-3 orchestration benchmark datasets from existing exports.",
    )
    _catalog_arg(stage3_parser)
    stage3_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional export dataset folder name, for example tabula_sapiens_bladder_cell_annotation.",
    )
    stage3_parser.add_argument(
        "--exports-root",
        type=str,
        default=None,
        help="Optional override for the grouped stage-2 exports root.",
    )
    stage3_parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional override for the stage-3 export root.",
    )
    stage3_parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="Optional cap applied independently to each task family and split, useful for smoke tests.",
    )

    gene_benchmark_parser = subparsers.add_parser(
        "build-gene-benchmark",
        help="Build a gene-only benchmark dataset with GPT-generated natural questions and answers.",
    )
    gene_benchmark_parser.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the gene-only benchmark builder.",
    )

    gene_benchmark_audit_parser = subparsers.add_parser(
        "audit-gene-benchmark",
        help="Audit an existing gene-only benchmark for answer gene mentions outside hidden grounding.",
    )
    gene_benchmark_audit_parser.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the gene-only benchmark audit command.",
    )

    run_all_parser = subparsers.add_parser("run-all", help="Download, audit, build, verify, and export in one command.")
    _catalog_arg(run_all_parser)
    run_all_parser.add_argument("--dataset", type=str, default=None, help="Optional dataset name from the catalog.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "search-census":
        matches = search_census_datasets(
            census_version=args.census_version,
            collection_name_contains=args.collection_name_contains,
            dataset_title_contains=args.dataset_title_contains,
        )
        print(json.dumps(matches, indent=2))
        return

    if args.command == "doctor":
        print(json.dumps(runtime_report(), indent=2))
        return

    if args.command == "build-gene-benchmark":
        forwarded_args = list(args.remainder)
        if forwarded_args[:1] == ["--"]:
            forwarded_args = forwarded_args[1:]
        build_gene_only_benchmark_main(forwarded_args)
        return

    if args.command == "audit-gene-benchmark":
        forwarded_args = list(args.remainder)
        if forwarded_args[:1] == ["--"]:
            forwarded_args = forwarded_args[1:]
        audit_gene_only_benchmark_main(forwarded_args)
        return

    if args.command in {"download", "audit", "build", "run-all"}:
        ensure_h5ad_runtime()

    config = load_catalog(args.catalog)

    if args.command == "list-approved":
        rows = []
        for spec in enabled_datasets(config):
            rows.append(
                {
                    "name": spec.name,
                    "source": spec.source,
                    "task_types": spec.task_types,
                    "required_obs_columns": spec.required_obs_columns,
                }
            )
        print(rows)
        return

    if args.command == "download":
        paths = download_from_catalog(config, dataset_name=args.dataset)
        print({"downloaded": [str(path) for path in paths]})
        return

    if args.command == "audit":
        report = audit_downloads(config, dataset_name=args.dataset)
        output_path = config.paths.reports_dir / "audit_report.json"
        write_json(output_path, report)
        print({"audit_report": str(output_path)})
        return

    if args.command == "build":
        summary = build_verified_examples(config, dataset_name=args.dataset)
        print(summary)
        return

    if args.command == "build-stage3":
        summary = build_stage3_dataset(
            config,
            dataset_name=args.dataset,
            exports_root=args.exports_root,
            output_root=args.output_root,
            limit_per_split=args.limit_per_split,
        )
        print(summary)
        return

    if args.command == "run-all":
        download_from_catalog(config, dataset_name=args.dataset)
        report = audit_downloads(config, dataset_name=args.dataset)
        write_json(config.paths.reports_dir / "audit_report.json", report)
        summary = build_verified_examples(config, dataset_name=args.dataset)
        print(summary)
        return