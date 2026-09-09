"""Cell2Sentence base (no LoRA) embedding method.

Loads the pretrained C2S-Scale Gemma-2 model **without** any LoRA adapter
and extracts last-token hidden states as cell embeddings.  This represents
the "pre-finetune" LLM baseline.
"""

from __future__ import annotations

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

from utils import build_c2s_prompt, get_hidden_states  # noqa: E402

_DEFAULT_MODEL = os.getenv(
    "GEMMA_MODEL_PATH",
    "vandijklab/C2S-Scale-Gemma-2-27B",
)


def name() -> str:
    return "C2S Base (no LoRA)"


def embed(
    genes_lists: list[list[str]],
    model_path: str = _DEFAULT_MODEL,
    batch_size: int = 16,
    max_seq_len: int = 2048,
    organism: str = "Homo sapiens",
    **kwargs,
) -> np.ndarray:
    """Produce embeddings using the pretrained Gemma model (no adapter).

    Returns shape ``(n_cells, hidden_size)``.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    all_hidden: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(genes_lists), batch_size):
            batch_genes = genes_lists[start : start + batch_size]
            prompts = [
                build_c2s_prompt(g, organism=organism) for g in batch_genes
            ]
            hidden = get_hidden_states(
                model, tokenizer, prompts, max_seq_len, enable_grad=False
            )
            all_hidden.append(hidden.float().cpu().numpy())

    return np.concatenate(all_hidden, axis=0)
