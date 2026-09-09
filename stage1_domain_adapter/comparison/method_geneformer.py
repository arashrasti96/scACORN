"""Geneformer zero-shot embedding method.

Converts ranked gene lists into Geneformer's rank-value token format and
extracts cell embeddings from the pretrained model.

Requires the ``geneformer`` package::

    git lfs install
    git clone https://huggingface.co/ctheodoris/Geneformer
    cd Geneformer && pip install .
"""

from __future__ import annotations

import numpy as np


def name() -> str:
    return "Geneformer (zero-shot)"


def _load_gene_mappings():
    """Load Geneformer's gene-name → Ensembl-ID and token dictionaries."""
    try:
        from geneformer import TranscriptomeTokenizer
    except ImportError:
        raise ImportError(
            "Geneformer is required.  Install with:\n"
            "  git lfs install\n"
            "  git clone https://huggingface.co/ctheodoris/Geneformer\n"
            "  cd Geneformer && pip install ."
        )

    import pickle
    from pathlib import Path

    # Locate dictionaries shipped with geneformer
    import geneformer
    pkg_dir = Path(geneformer.__file__).parent

    # gene_name_id_dict: gene_symbol → ensembl_id
    name_id_path = pkg_dir / "gene_name_id_dict.pkl"
    if not name_id_path.exists():
        # Older versions may use a different filename
        name_id_path = pkg_dir / "gene_name_id_dict_gc30M.pkl"
    with open(name_id_path, "rb") as f:
        gene_name_to_ens = pickle.load(f)

    # token_dictionary: ensembl_id → token_id
    token_path = pkg_dir / "token_dictionary.pkl"
    if not token_path.exists():
        token_path = pkg_dir / "token_dictionary_gc30M.pkl"
    with open(token_path, "rb") as f:
        ens_to_token = pickle.load(f)

    return gene_name_to_ens, ens_to_token


def _genes_to_token_ids(
    genes: list[str],
    gene_name_to_ens: dict[str, str],
    ens_to_token: dict[str, int],
) -> list[int]:
    """Map gene symbols → Ensembl IDs → Geneformer token IDs."""
    ids = []
    for g in genes:
        ens = gene_name_to_ens.get(g)
        if ens is not None and ens in ens_to_token:
            ids.append(ens_to_token[ens])
    return ids


def embed(
    genes_lists: list[list[str]],
    model_name: str = "ctheodoris/Geneformer",
    batch_size: int = 32,
    **kwargs,
) -> np.ndarray:
    """Produce Geneformer zero-shot embeddings from ranked gene lists.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier or local path for Geneformer.
    batch_size:
        Cells per forward pass.

    Returns shape ``(n_cells, hidden_size)``.
    """
    import torch
    from transformers import AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Geneformer model
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True
    ).to(device)
    model.eval()

    # Load gene mappings from the geneformer package
    gene_name_to_ens, ens_to_token = _load_gene_mappings()

    # Determine pad token id (Geneformer uses 0 as pad)
    pad_token_id = 0

    all_embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(genes_lists), batch_size):
            batch_genes = genes_lists[start : start + batch_size]

            # Convert gene symbols to token IDs
            batch_ids = []
            for genes in batch_genes:
                ids = _genes_to_token_ids(genes, gene_name_to_ens, ens_to_token)
                if not ids:
                    ids = [pad_token_id]
                batch_ids.append(ids)

            # Pad to max length in batch
            max_len = max(len(ids) for ids in batch_ids)
            input_ids_list = []
            attn_masks = []
            for ids in batch_ids:
                pad_len = max_len - len(ids)
                attn = [1] * len(ids) + [0] * pad_len
                padded = ids + [pad_token_id] * pad_len
                input_ids_list.append(padded)
                attn_masks.append(attn)

            input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
            attention_mask = torch.tensor(attn_masks, dtype=torch.long, device=device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state

            # Mean-pool over non-padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).float()
            summed = (last_hidden * mask_expanded).sum(dim=1)
            counts = mask_expanded.sum(dim=1).clamp(min=1)
            cell_emb = summed / counts

            all_embeddings.append(cell_emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0).astype(np.float32)
