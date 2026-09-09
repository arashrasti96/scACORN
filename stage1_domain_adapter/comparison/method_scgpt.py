"""scGPT zero-shot embedding method.

Builds a pseudo-AnnData object from ordered gene lists (using inverse-rank
as synthetic expression counts) and calls scGPT's ``embed_data`` to produce
cell embeddings.
"""

from __future__ import annotations

import numpy as np


def name() -> str:
    return "scGPT (zero-shot)"


def _build_pseudo_anndata(
    genes_lists: list[list[str]],
) -> "anndata.AnnData":
    """Build an AnnData from ranked gene lists with pseudo-counts.

    Gene at rank 0 → value = N, rank N-1 → value = 1.  Genes not
    present in a cell get value 0.
    """
    import anndata as ad
    import scipy.sparse as sp

    # Collect full vocabulary
    vocab: dict[str, int] = {}
    for genes in genes_lists:
        for g in genes:
            if g not in vocab:
                vocab[g] = len(vocab)

    n_cells = len(genes_lists)
    n_genes = len(vocab)
    gene_names = [""] * n_genes
    for g, idx in vocab.items():
        gene_names[idx] = g

    # Build sparse count matrix using inverse-rank pseudo-counts
    rows, cols, vals = [], [], []
    for i, genes in enumerate(genes_lists):
        n = len(genes)
        for rank, g in enumerate(genes):
            rows.append(i)
            cols.append(vocab[g])
            vals.append(float(n - rank))

    X = sp.csr_matrix(
        (vals, (rows, cols)), shape=(n_cells, n_genes), dtype=np.float32
    )

    import pandas as pd

    adata = ad.AnnData(
        X=X,
        var=pd.DataFrame(index=gene_names),
    )
    return adata


def embed(
    genes_lists: list[list[str]],
    model_dir: str = "scGPT_human",
    batch_size: int = 64,
    **kwargs,
) -> np.ndarray:
    """Produce scGPT zero-shot embeddings.

    Parameters
    ----------
    model_dir:
        Path or name of the scGPT pre-trained model directory. If not an
        absolute path, will attempt to download the ``whole-human`` model.
    batch_size:
        Cells per forward pass.

    Returns shape ``(n_cells, embed_dim)``.
    """
    import torch
    from pathlib import Path

    adata = _build_pseudo_anndata(genes_lists)

    # Try importing scGPT
    try:
        from scgpt.model import TransformerModel
        from scgpt.tokenizer.gene_tokenizer import GeneVocab
        from scgpt.utils import set_seed
    except ImportError:
        raise ImportError(
            "scGPT is required.  Install with: pip install scgpt"
        )

    model_path = Path(model_dir)
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"scGPT model directory not found: {model_dir}. "
            "Download the whole-human model from "
            "https://github.com/bowang-lab/scGPT#pretrained-scgpt-model-zoo"
        )

    # Load gene vocabulary from the model
    vocab_file = model_path / "vocab.json"
    vocab = GeneVocab.from_file(vocab_file)
    pad_token = "<pad>"
    pad_value = -2

    # Filter genes to those in scGPT vocabulary
    gene_ids_in_vocab = [
        vocab[g] for g in adata.var_names if g in vocab
    ]
    genes_in_vocab = [g for g in adata.var_names if g in vocab]

    if not genes_in_vocab:
        raise ValueError("No genes in the data match the scGPT vocabulary.")

    # Subset AnnData to vocab genes
    adata = adata[:, genes_in_vocab].copy()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model config
    import json

    model_config_file = model_path / "args.json"
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)

    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]

    ntokens = len(vocab)

    model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        vocab=vocab,
        pad_value=pad_value,
        pad_token=pad_token,
    )

    # Load pretrained weights
    model_file = model_path / "best_model.pt"
    model.load_state_dict(
        torch.load(model_file, map_location=device)
    )
    model.to(device)
    model.eval()

    # Tokenize and embed
    import scipy.sparse as sp

    if sp.issparse(adata.X):
        count_matrix = adata.X.toarray()
    else:
        count_matrix = np.array(adata.X)

    all_embeddings = []
    with torch.no_grad():
        for start in range(0, len(count_matrix), batch_size):
            batch_counts = count_matrix[start : start + batch_size]
            batch_gene_ids = np.array(gene_ids_in_vocab)

            # Prepare input tensors
            gene_ids_tensor = torch.from_numpy(
                np.tile(batch_gene_ids, (len(batch_counts), 1))
            ).long().to(device)

            values_tensor = torch.from_numpy(batch_counts).float().to(device)

            # Forward pass to get cell embeddings
            src_key_padding_mask = torch.zeros_like(
                gene_ids_tensor, dtype=torch.bool
            )
            output = model._encode(
                gene_ids_tensor, values_tensor, src_key_padding_mask
            )
            # Mean-pool over gene tokens to get cell embedding
            cell_emb = output.mean(dim=1)
            all_embeddings.append(cell_emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0).astype(np.float32)
