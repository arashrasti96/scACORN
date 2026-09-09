"""TF-IDF + SVD baseline for single-cell embedding comparison.

Treats each cell's ordered gene list as a sentence (space-joined), applies
TF-IDF vectorization, then reduces to 128d via Truncated SVD.  CPU-only.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def name() -> str:
    return "TF-IDF + SVD"


def embed(
    genes_lists: list[list[str]],
    n_components: int = 128,
    **kwargs,
) -> np.ndarray:
    """Produce embeddings for a list of cells.

    Returns shape ``(n_cells, n_components)``.
    """
    sentences = [" ".join(genes) for genes in genes_lists]
    print(f"      [tfidf] vectorizing {len(sentences)} sentences ...", flush=True)

    tfidf = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[A-Za-z0-9\-]+",
        lowercase=False,
    )
    X = tfidf.fit_transform(sentences)
    print(f"      [tfidf] sparse matrix {X.shape}, nnz={X.nnz}", flush=True)

    n_comp = min(n_components, X.shape[0] - 1, X.shape[1] - 1)
    print(f"      [tfidf] SVD to {n_comp}d (randomized) ...", flush=True)
    svd = TruncatedSVD(
        n_components=max(1, n_comp), algorithm="randomized", random_state=42,
    )
    return svd.fit_transform(X).astype(np.float32)
