"""C2S + LoRA adapter embedding method (user's trained model).

Loads the pretrained Gemma-2 model, applies a saved LoRA adapter checkpoint
and projection head, then extracts both feature-space and projection-space
embeddings.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add parent Contrastive_learning directory to path so we can import utils
_SCRIPT_DIR = Path(__file__).resolve().parent
_CL_DIR = _SCRIPT_DIR.parent.parent  # …/Contrastive_learning
if str(_CL_DIR) not in sys.path:
    sys.path.insert(0, str(_CL_DIR))

from utils import ProjectionHead, build_c2s_prompt, get_hidden_states  # noqa: E402

_DEFAULT_MODEL = os.getenv(
    "GEMMA_MODEL_PATH",
    "vandijklab/C2S-Scale-Gemma-2-2B",
)

_DEFAULT_OUTPUTS_DIR = _SCRIPT_DIR.parent / "outputs"


def name() -> str:
    return "C2S + LoRA (contrastive)"


def _find_best_checkpoint(run_dir: Path) -> Path:
    best = run_dir / "checkpoints" / "best"
    if best.is_dir():
        return best
    last = run_dir / "checkpoints" / "last"
    if last.is_dir():
        return last
    raise FileNotFoundError(f"No checkpoint found in {run_dir}")


def _find_matching_run(dataset_name: str) -> Path | None:
    """Find a per-dataset adapter run whose name contains the dataset name."""
    if not _DEFAULT_OUTPUTS_DIR.is_dir():
        return None
    for run_dir in sorted(_DEFAULT_OUTPUTS_DIR.iterdir()):
        # e.g. dataset_name = "tabula_sapiens_heart_cell_annotation"
        #      run_dir.name = "gemma_stage1_tabula_sapiens_heart_cell_annotation"
        if dataset_name in run_dir.name:
            try:
                return _find_best_checkpoint(run_dir)
            except FileNotFoundError:
                continue
    return None


def embed(
    genes_lists: list[list[str]],
    checkpoint_dir: str | None = None,
    run_name: str | None = None,
    dataset_name: str | None = None,
    model_path: str = _DEFAULT_MODEL,
    batch_size: int = 16,
    max_seq_len: int = 512,
    organism: str = "Homo sapiens",
    return_projections: bool = False,
    **kwargs,
) -> np.ndarray:
    """Produce embeddings from the trained LoRA adapter.

    Parameters
    ----------
    checkpoint_dir:
        Explicit path to a checkpoint directory containing
        ``gemma_lora_stage1/`` (adapter) and ``projection_head.pt``.
    run_name:
        Name of the training run under ``outputs/``. Used to locate the
        best checkpoint if ``checkpoint_dir`` is not given.
    dataset_name:
        Name of the dataset being evaluated. When set and no explicit
        checkpoint/run is specified, the code will auto-match a
        per-dataset adapter whose run name contains this string.
    model_path:
        Base model path (before LoRA).
    return_projections:
        If True, return projected embeddings (128-d). Otherwise return
        raw hidden states (feature space).

    Returns shape ``(n_cells, dim)``.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Resolve checkpoint directory
    if checkpoint_dir:
        ckpt = Path(checkpoint_dir)
    elif run_name:
        ckpt = _find_best_checkpoint(_DEFAULT_OUTPUTS_DIR / run_name)
    else:
        # Try to match a per-dataset adapter first
        ckpt = None
        if dataset_name:
            ckpt = _find_matching_run(dataset_name)
        # Fall back to the shared multi-dataset adapter (2B model runs)
        if ckpt is None:
            for cand in sorted(_DEFAULT_OUTPUTS_DIR.iterdir()):
                # Skip 14B/27B runs for auto-discover — prefer 2B per-dataset
                if "14B" in cand.name or "27B" in cand.name:
                    continue
                try:
                    ckpt = _find_best_checkpoint(cand)
                    break
                except FileNotFoundError:
                    continue
        if ckpt is None:
            raise FileNotFoundError(
                f"No training runs with checkpoints found in {_DEFAULT_OUTPUTS_DIR}"
            )

    adapter_dir = ckpt / "gemma_lora_stage1"
    proj_head_path = ckpt / "projection_head.pt"
    metadata_path = ckpt / "metadata.json"

    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter not found: {adapter_dir}")

    # Read the adapter config to determine the correct base model
    adapter_config_path = adapter_dir / "adapter_config.json"
    if adapter_config_path.is_file():
        with open(adapter_config_path) as f:
            adapter_cfg = json.load(f)
        saved_base = adapter_cfg.get("base_model_name_or_path")
        if saved_base:
            model_path = saved_base

    # Read metadata for hidden_size / projection_dim
    meta = {}
    if metadata_path.is_file():
        with open(metadata_path) as f:
            meta = json.load(f)
    hidden_size = meta.get("hidden_size", 2304)
    proj_dim = meta.get("args", {}).get("projection_dim", 128)

    print(f"  Loading base model: {model_path}")
    print(f"  Adapter: {adapter_dir}")
    print(f"  hidden_size={hidden_size}, proj_dim={proj_dim}")

    # Load base model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA adapter
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    # Load projection head
    proj_head = ProjectionHead(
        input_dim=hidden_size,
        hidden_dim=hidden_size // 2,
        output_dim=proj_dim,
    )
    if proj_head_path.is_file():
        proj_head.load_state_dict(
            torch.load(proj_head_path, map_location="cpu")
        )
    device = next(model.parameters()).device
    proj_head = proj_head.to(device)
    proj_head.eval()

    all_features: list[np.ndarray] = []
    all_projections: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(genes_lists), batch_size):
            batch_genes = genes_lists[start : start + batch_size]
            prompts = [
                build_c2s_prompt(g, organism=organism) for g in batch_genes
            ]
            hidden = get_hidden_states(
                model, tokenizer, prompts, max_seq_len, enable_grad=False
            )
            projected = proj_head(hidden)
            all_features.append(hidden.float().cpu().numpy())
            all_projections.append(projected.float().cpu().numpy())

    features = np.concatenate(all_features, axis=0)
    projections = np.concatenate(all_projections, axis=0)

    return projections if return_projections else features
