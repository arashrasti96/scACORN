"""SentenceTransformer baseline for single-cell embedding comparison.

Uses a lightweight pre-trained sentence-transformer model
(``all-MiniLM-L6-v2``) to embed cell sentences.
"""

from __future__ import annotations

import numpy as np


def name() -> str:
    return "SentenceTransformer (all-MiniLM-L6-v2)"


def embed(
    genes_lists: list[list[str]],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    **kwargs,
) -> np.ndarray:
    """Produce embeddings for a list of cells.

    Each cell's gene list is space-joined into a "sentence" and encoded
    by the specified SentenceTransformer model.

    Returns shape ``(n_cells, embed_dim)``.
    """
    from sentence_transformers import SentenceTransformer

    sentences = [" ".join(genes) for genes in genes_lists]

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)
