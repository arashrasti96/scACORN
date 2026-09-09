#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
CONTRASTIVE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = CONTRASTIVE_ROOT

sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != SCRIPT_DIR]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CONTRASTIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTRASTIVE_ROOT))

from ace import Playbook  # noqa: E402
from stage3_ace_orchestrator.config import Stage3Paths, Stage3RuntimeConfig  # noqa: E402
from stage3_ace_orchestrator.engine import Stage3ExpertRuntime  # noqa: E402
from stage3_ace_orchestrator.orchestrator import build_stage3_generator  # noqa: E402
from stage3_ace_orchestrator.requests import NormalizedStage3Request  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the stage-3 ACE orchestrator")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Stage-3 JSONL dataset. Defaults to all_grouped_val.jsonl under stage3_exports.",
    )
    parser.add_argument("--llm-model", default="gpt-5.1", help="Orchestrator model")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample limit")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                yield json.loads(line)


def parse_answer_payload(text: str) -> dict[str, Any] | None:
    trimmed = text.strip()
    if not trimmed:
        return None
    if not trimmed.startswith("{"):
        start = trimmed.find("{")
        end = trimmed.rfind("}")
        if start != -1 and end != -1 and end > start:
            trimmed = trimmed[start : end + 1]
    try:
        parsed = json.loads(trimmed)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_answer_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, list):
        normalized = [_normalize_answer_value(item) for item in value]
        if all(not isinstance(item, (dict, list)) for item in normalized):
            return sorted(normalized, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True))
        return normalized
    if isinstance(value, dict):
        return {
            str(key): _normalize_answer_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    return value


def _values_match(expected: Any, observed: Any) -> bool:
    return _normalize_answer_value(expected) == _normalize_answer_value(observed)


def evaluate_answer(request: NormalizedStage3Request, parsed_answer: dict[str, Any] | None, raw_answer: str) -> dict[str, Any]:
    expected = request.ground_truth.get(request.primary_target or "")
    observed = None if parsed_answer is None else parsed_answer.get(request.primary_target or "")
    if request.primary_target == "abstain":
        expected_bool = bool(expected)
        observed_bool = bool(observed)
        hit = expected_bool == observed_bool
    elif expected is not None and observed is not None:
        hit = _values_match(expected, observed)
    else:
        normalized_expected = str(expected).strip().lower() if expected is not None else None
        hit = normalized_expected is not None and normalized_expected in raw_answer.strip().lower()
    return {
        "primary_target": request.primary_target,
        "expected": expected,
        "observed": observed,
        "exact_hit": bool(hit),
    }


def evaluate_routing(request: NormalizedStage3Request, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    called_experts = []
    for call in tool_calls:
        expert_name = call.get("expert_name")
        if expert_name and expert_name not in called_experts:
            called_experts.append(expert_name)
    candidate_overlap = sorted(set(called_experts) & set(request.candidate_experts))
    oracle_overlap = sorted(set(called_experts) & set(request.oracle_experts))
    return {
        "called_experts": called_experts,
        "candidate_overlap": candidate_overlap,
        "oracle_overlap": oracle_overlap,
        "candidate_overlap_count": len(candidate_overlap),
        "oracle_overlap_count": len(oracle_overlap),
    }


def main() -> None:
    args = parse_args()
    paths = Stage3Paths.discover()
    dataset_path = Path(args.dataset) if args.dataset else (paths.stage3_exports_root / "all_grouped_val.jsonl")
    output_path = Path(args.output) if args.output else (SCRIPT_DIR / "outputs" / f"benchmark_{dataset_path.stem}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    runtime = Stage3ExpertRuntime(paths, Stage3RuntimeConfig())
    generator = build_stage3_generator(runtime, model=args.llm_model)
    rows = []
    for index, row in enumerate(iter_jsonl(dataset_path), start=1):
        if args.max_samples is not None and index > args.max_samples:
            break
        request = NormalizedStage3Request.from_stage3_row(row)
        runtime.reset_call_log()
        generator_output = generator.generate(
            question=request.ace_question,
            context=request.ace_context,
            playbook=Playbook(),
        )
        parsed_answer = parse_answer_payload(generator_output.final_answer)
        answer_metrics = evaluate_answer(request, parsed_answer, generator_output.final_answer)
        routing_metrics = evaluate_routing(request, runtime.get_call_log())
        rows.append(
            {
                "sample_id": request.sample_id,
                "task_family": request.task_family,
                "answer_metrics": answer_metrics,
                "routing_metrics": routing_metrics,
                "tool_calls": runtime.get_call_log(),
                "generator_final_answer": generator_output.final_answer,
            }
        )
        print(f"[{len(rows)}] {request.sample_id} exact_hit={answer_metrics['exact_hit']}", flush=True)

    summary = {
        "dataset": str(dataset_path),
        "rows": len(rows),
        "answer_accuracy": (
            sum(1 for row in rows if row["answer_metrics"]["exact_hit"]) / len(rows)
            if rows else 0.0
        ),
        "candidate_routing_hit_rate": (
            sum(1 for row in rows if row["routing_metrics"]["candidate_overlap_count"] > 0) / len(rows)
            if rows else 0.0
        ),
        "oracle_routing_hit_rate": (
            sum(1 for row in rows if row["routing_metrics"]["oracle_overlap_count"] > 0) / len(rows)
            if rows else 0.0
        ),
        "samples": rows,
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ["dataset", "rows", "answer_accuracy", "candidate_routing_hit_rate", "oracle_routing_hit_rate"]}, indent=2))
    print(f"Saved benchmark report to {output_path}")


if __name__ == "__main__":
    main()