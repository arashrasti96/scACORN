"""Evaluation metrics for the comparison framework.

Provides Recall@K (cosine), NMI, ARI, and Silhouette — all computed
from embeddings and ground-truth labels.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# Recall@K  (ported from utils.print_recall_at_k)
# ---------------------------------------------------------------------------

def recall_at_k(
    embeddings: np.ndarray,
    labels: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float | None]:
    """Compute Recall@K using cosine-distance nearest-neighbour search.

    Returns a dict like ``{"R@1": 0.82, "R@5": 0.95, "R@10": 0.98}``.
    """
    n = len(labels)
    if n < 2:
        return {f"R@{k}": None for k in ks}

    max_k = min(max(ks), n - 1)
    if max_k < 1:
        return {f"R@{k}": None for k in ks}

    nn = NearestNeighbors(n_neighbors=max_k + 1, metric="cosine")
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)

    # Drop self-neighbour at rank 0
    indices = indices[:, 1:]
    nn_labels = labels[indices]
    matches = nn_labels == labels[:, None]

    results: dict[str, float | None] = {}
    for k in ks:
        key = f"R@{k}"
        if k > max_k:
            results[key] = None
        else:
            results[key] = float(matches[:, :k].any(axis=1).mean())
    return results


# ---------------------------------------------------------------------------
# Clustering metrics
# ---------------------------------------------------------------------------

def clustering_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
) -> dict[str, float | None]:
    """Run KMeans and compute NMI, ARI, and Silhouette Score.

    The number of clusters is set to the number of unique labels.
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    n_samples = len(labels)

    if n_clusters < 2 or n_samples < n_clusters:
        return {"NMI": None, "ARI": None, "Silhouette": None}

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    pred = km.fit_predict(embeddings)

    nmi = normalized_mutual_info_score(labels, pred)
    ari = adjusted_rand_score(labels, pred)
    sil = silhouette_score(embeddings, pred, metric="cosine")

    return {
        "NMI": float(nmi),
        "ARI": float(ari),
        "Silhouette": float(sil),
    }


# ---------------------------------------------------------------------------
# Combined evaluation
# ---------------------------------------------------------------------------

def evaluate_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
    seed: int = 42,
) -> dict[str, float | None]:
    """Run all metrics (Recall@K, NMI, ARI, Silhouette) and return a flat dict."""
    results = recall_at_k(embeddings, labels, ks=ks)
    results.update(clustering_metrics(embeddings, labels, seed=seed))
    return results
