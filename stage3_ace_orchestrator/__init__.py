"""Stage-3 ACE orchestrator package."""

from typing import TYPE_CHECKING

from .config import Stage3Paths, Stage3RuntimeConfig, find_repo_root
from .requests import NormalizedStage3Request
from .registry import ExpertDescriptor, discover_experts, load_expert_registry

if TYPE_CHECKING:
    from .engine import ExpertExecutionResult, Stage3ExpertRuntime
    from .orchestrator import build_stage3_generator, build_stage3_langchain_agent

__all__ = [
    "ExpertDescriptor",
    "ExpertExecutionResult",
    "NormalizedStage3Request",
    "Stage3Paths",
    "Stage3ExpertRuntime",
    "Stage3RuntimeConfig",
    "build_stage3_generator",
    "build_stage3_langchain_agent",
    "discover_experts",
    "find_repo_root",
    "load_expert_registry",
]


def __getattr__(name: str):
    if name in {"ExpertExecutionResult", "Stage3ExpertRuntime"}:
        from .engine import ExpertExecutionResult, Stage3ExpertRuntime

        return {
            "ExpertExecutionResult": ExpertExecutionResult,
            "Stage3ExpertRuntime": Stage3ExpertRuntime,
        }[name]
    if name in {"build_stage3_generator", "build_stage3_langchain_agent"}:
        from .orchestrator import build_stage3_generator, build_stage3_langchain_agent

        return {
            "build_stage3_generator": build_stage3_generator,
            "build_stage3_langchain_agent": build_stage3_langchain_agent,
        }[name]
    raise AttributeError(name)