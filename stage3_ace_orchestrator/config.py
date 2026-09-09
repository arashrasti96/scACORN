from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward until the standalone BioAgent repository root is found."""
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / "ace" / "Agent.py").exists() and (candidate / "stage3_ace_orchestrator").exists():
            return candidate
    raise FileNotFoundError("Could not locate the BioAgent repository root from the stage-3 package")


def load_repo_dotenv(start: Path | None = None, *, override: bool = False) -> Path | None:
    """Load the repo-root .env file if it exists and return its path."""
    repo_root = find_repo_root(start)
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


@dataclass(frozen=True)
class Stage3Paths:
    repo_root: Path
    contrastive_root: Path
    stage2_root: Path
    stage2_outputs_root: Path
    stage3_dataset_root: Path
    stage3_exports_root: Path
    stage3_package_root: Path
    expert_profiles_path: Path

    @classmethod
    def discover(cls) -> "Stage3Paths":
        repo_root = find_repo_root()
        contrastive_root = repo_root
        stage2_root = contrastive_root / "stage2_task_adapter"
        return cls(
            repo_root=repo_root,
            contrastive_root=contrastive_root,
            stage2_root=stage2_root,
            stage2_outputs_root=stage2_root / "outputs",
            stage3_dataset_root=contrastive_root / "dataset_pipeline",
            stage3_exports_root=contrastive_root / "dataset_pipeline" / "data" / "stage3_exports",
            stage3_package_root=contrastive_root / "stage3_ace_orchestrator",
            expert_profiles_path=contrastive_root / "stage3_ace_orchestrator" / "expert_profiles.json",
        )


@dataclass(frozen=True)
class Stage3RuntimeConfig:
    base_model: str = os.getenv("GEMMA_MODEL_PATH", "vandijklab/C2S-Scale-Gemma-2-2B")
    top_genes: int = 200
    max_seq_len: int = 2048
    max_new_tokens: int = 256
    temperature: float = 0.0
    use_4bit: bool = False
    max_experts_per_query: int = 2
    expert_cache_size: int = 2
    registry_limit: int | None = None
    verbose: bool = False
