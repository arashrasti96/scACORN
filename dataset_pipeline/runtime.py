import sys
from pathlib import Path


def find_recommended_python() -> Path | None:
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / ".venv" / "bin" / "python"
        if candidate.exists():
            return candidate
    return None


def _format_environment_hint() -> str:
    current_python = Path(sys.executable)
    lines = [f"Current interpreter: {current_python}"]
    recommended_python = find_recommended_python()
    if recommended_python is not None and recommended_python != current_python:
        lines.append(f"Recommended interpreter: {recommended_python}")
        if current_python.name == recommended_python.name:
            lines.append(
                "The shell prompt name can be misleading when both environments are named '.venv'. "
                "The authoritative path is the interpreter shown above."
            )
        lines.append("Re-run the command with the recommended interpreter or activate that virtual environment first.")
    return "\n".join(lines)


def ensure_h5ad_runtime() -> None:
    try:
        import numpy  # noqa: F401
        import h5py  # noqa: F401
        import anndata  # noqa: F401
    except Exception as exc:
        details = [
            "H5AD-backed commands require a working numpy/h5py/anndata runtime.",
            _format_environment_hint(),
            f"Original error: {exc}",
        ]
        message = str(exc).lower()
        if "libcpupower.so.0" in message:
            details.append("Detected missing native library: libcpupower.so.0")
        raise RuntimeError("\n".join(details)) from exc


def runtime_report() -> dict:
    current_python = str(Path(sys.executable))
    recommended_python = find_recommended_python()
    report = {
        "current_python": current_python,
        "recommended_python": None if recommended_python is None else str(recommended_python),
        "h5ad_runtime_ok": False,
        "h5ad_runtime_error": None,
    }
    try:
        ensure_h5ad_runtime()
        report["h5ad_runtime_ok"] = True
    except RuntimeError as exc:
        report["h5ad_runtime_error"] = str(exc)
    return report