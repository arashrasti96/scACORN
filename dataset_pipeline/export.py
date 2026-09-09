from collections import Counter, defaultdict
from pathlib import Path

from .builders import to_stage2_jsonl_row
from .io_utils import ensure_dir, write_json, write_jsonl
from .schemas import TaskExample, VerificationResult


def export_intermediate(examples: list[TaskExample], path: Path) -> None:
    write_jsonl(path, [example.to_dict() for example in examples])


def export_stage2(examples: list[TaskExample], export_dir: Path) -> dict:
    ensure_dir(export_dir)
    by_task = defaultdict(list)
    for example in examples:
        by_task[example.task_type].append(example)

    manifest = {"tasks": {}}
    for task_type, task_examples in by_task.items():
        rows = [to_stage2_jsonl_row(example) for example in task_examples]
        output_path = export_dir / f"{task_type}.jsonl"
        write_jsonl(output_path, rows)
        manifest["tasks"][task_type] = {
            "rows": len(rows),
            "output_path": str(output_path),
        }
    return manifest


def export_verification_report(results: list[VerificationResult], path: Path) -> dict:
    field_failures = Counter()
    task_counts = Counter()
    passed_count = 0
    for result in results:
        task_counts[result.task_type] += 1
        if result.passed:
            passed_count += 1
        for issue in result.issues:
            field_failures[issue.field_name] += 1

    payload = {
        "total_rows": len(results),
        "passed_rows": passed_count,
        "failed_rows": len(results) - passed_count,
        "task_counts": dict(task_counts),
        "field_failures": dict(field_failures),
        "results": [result.to_dict() for result in results],
    }
    write_json(path, payload)
    return payload