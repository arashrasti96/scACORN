from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .builder import BenchmarkDatasetBuilder, BuildConfig, QUESTION_SPECS


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
DEFAULT_SOURCE_ROOT = PIPELINE_DIR / "data" / "exports"
DEFAULT_OUTPUT_ROOT = PIPELINE_DIR / "data" / "gene_only_benchmarks"


def _load_repo_dotenv(start: Path | None = None, *, override: bool = False) -> Path | None:
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent

    repo_root = None
    for candidate in (current, *current.parents):
        if (candidate / "ace" / "Agent.py").exists() and (candidate / "examples").exists():
            repo_root = candidate
            break
    if repo_root is None:
        return None

    env_path = repo_root / ".env"
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value
    return env_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a gene-only benchmark dataset with GPT-generated natural questions and answers."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "gpt4o_mini_benchmark_v1",
        help="Directory where benchmark.jsonl, benchmark_hidden.jsonl, and manifest.json will be written.",
    )
    parser.add_argument(
        "-k",
        "--k",
        type=int,
        required=True,
        help="Number of examples to generate per selected question type.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Root directory containing export subdirectories with cell_annotation_rationale JSONL files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        help="Source splits to load. Use values such as test, val, train, or all.",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=["cell", "cluster"],
        default=["cell", "cluster"],
        help="Benchmark levels to include.",
    )
    parser.add_argument(
        "--question-types",
        nargs="+",
        choices=sorted(QUESTION_SPECS),
        help="Optional explicit question types. Defaults to all question types for the selected levels.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model name used to generate natural questions and answers.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for deterministic sampling.")
    parser.add_argument(
        "--cluster-size",
        type=int,
        default=3,
        help="Number of cells to include in each cluster-level example.",
    )
    parser.add_argument(
        "--max-genes",
        type=int,
        default=80,
        help="Maximum number of genes to expose per cell in the public model input.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature for GPT generation.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse any already-generated examples found in the output directory.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional explicit OpenAI API key. By default OPENAI_API_KEY is used.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    _load_repo_dotenv(Path(__file__))
    parser = build_parser()
    args = parser.parse_args(argv)
    config = BuildConfig(
        source_root=args.source_root,
        output_dir=args.output_dir,
        k=args.k,
        seed=args.seed,
        splits=tuple(args.splits),
        levels=tuple(args.levels),
        question_types=tuple(args.question_types or ()),
        model_name=args.model,
        cluster_size=args.cluster_size,
        max_genes=args.max_genes,
        temperature=args.temperature,
        resume=args.resume,
        api_key=args.api_key,
    )
    manifest = BenchmarkDatasetBuilder(config).build()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()