#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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
    parser = argparse.ArgumentParser(description="Run one stage-3 ACE orchestration query")
    parser.add_argument("--question", required=True, help="Biology question for the orchestrator")
    parser.add_argument("--genes", default="", help="Comma-separated ordered genes")
    parser.add_argument("--context", default="", help="Optional grounded context")
    parser.add_argument("--task-family", default="freeform_qa", help="Stage-3 task family label")
    parser.add_argument("--task-type", default="", help="Optional explicit stage-2 task type")
    parser.add_argument("--candidate-labels", default="", help="Optional comma-separated candidate labels")
    parser.add_argument("--llm-model", default="gpt-5.1", help="Orchestrator model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = Stage3ExpertRuntime(Stage3Paths.discover(), Stage3RuntimeConfig())
    generator = build_stage3_generator(runtime, model=args.llm_model)
    request = NormalizedStage3Request.from_freeform_query(
        question_text=args.question,
        genes=[token for token in args.genes.replace(",", " ").split() if token],
        context=args.context,
        task_family=args.task_family,
        task_type=args.task_type or None,
        candidate_labels=[token.strip() for token in args.candidate_labels.split(",") if token.strip()],
    )
    runtime.reset_call_log()
    result = generator.generate(
        question=request.ace_question,
        context=request.ace_context,
        playbook=Playbook(),
    )
    payload = {
        "generator": result.raw,
        "final_answer": result.final_answer,
        "tool_calls": runtime.get_call_log(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()