"""PCA baseline for single-cell embedding comparison.

Encodes each cell as a fixed-length gene-rank feature vector, then applies
PCA to produce a 128-d embedding.  CPU-only; no deep learning.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def name() -> str:
    return "PCA (rank-vector)"


def embed(
    genes_lists: list[list[str]],
    n_components: int = 128,
    **kwargs,
) -> np.ndarray:
    """Produce embeddings for a list of cells.

    Each cell's gene list is encoded as a vector over the shared gene
    vocabulary where the value at each position is the *inverse rank*
    (gene at rank 0 → value = N, gene at rank N-1 → value = 1, absent → 0).
    PCA then reduces this to ``n_components`` dimensions.

    Returns shape ``(n_cells, n_components)``.
    """
    # Build shared vocabulary across all cells
    vocab: dict[str, int] = {}
    for genes in genes_lists:
        for g in genes:
            if g not in vocab:
                vocab[g] = len(vocab)

    # Encode each cell as inverse-rank vector
    n_cells = len(genes_lists)
    n_genes = len(vocab)
    X = np.zeros((n_cells, n_genes), dtype=np.float32)
    for i, genes in enumerate(genes_lists):
        n = len(genes)
        for rank, g in enumerate(genes):
            X[i, vocab[g]] = n - rank  # inverse rank

    n_comp = min(n_components, n_cells, n_genes)
    pca = PCA(n_components=n_comp, random_state=42)
    return pca.fit_transform(X)
