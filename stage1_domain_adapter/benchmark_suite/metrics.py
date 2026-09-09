from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors


def self_retrieval_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float | None]:
    sample_count = len(labels)
    if sample_count < 2:
        return {f"R{rank}": None for rank in ks}

    max_k = min(max(ks), sample_count - 1)
    if max_k < 1:
        return {f"R{rank}": None for rank in ks}

    index = NearestNeighbors(n_neighbors=max_k + 1, metric="cosine")
    index.fit(embeddings)
    _, indices = index.kneighbors(embeddings)
    neighbor_labels = labels[indices[:, 1:]]
    return recall_from_neighbor_labels(neighbor_labels, labels, ks)


def clustering_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
) -> dict[str, float | None]:
    unique_labels = np.unique(labels)
    cluster_count = len(unique_labels)
    sample_count = len(labels)

    if cluster_count < 2 or sample_count <= cluster_count:
        return {"NMI": None, "ARI": None, "Silhouette": None}

    clusterer = KMeans(n_clusters=cluster_count, random_state=seed, n_init=10)
    predicted = clusterer.fit_predict(embeddings)

    return {
        "NMI": float(normalized_mutual_info_score(labels, predicted)),
        "ARI": float(adjusted_rand_score(labels, predicted)),
        "Silhouette": float(silhouette_score(embeddings, predicted, metric="cosine")),
    }


def knn_label_transfer_metrics(
    reference_embeddings: np.ndarray,
    reference_labels: np.ndarray,
    query_embeddings: np.ndarray,
    query_labels: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
    classifier_k: int = 5,
    same_set: bool = False,
) -> dict[str, float | None]:
    if len(reference_labels) == 0 or len(query_labels) == 0:
        results = {f"R{rank}": None for rank in ks}
        results.update({"knn_acc": None, "knn_macro_f1": None})
        return results

    extra_neighbor = 1 if same_set else 0
    max_k = min(max(max(ks), classifier_k), len(reference_labels) - extra_neighbor)
    if max_k < 1:
        results = {f"R{rank}": None for rank in ks}
        results.update({"knn_acc": None, "knn_macro_f1": None})
        return results

    index = NearestNeighbors(n_neighbors=max_k + extra_neighbor, metric="cosine")
    index.fit(reference_embeddings)
    distances, indices = index.kneighbors(query_embeddings)

    if same_set:
        distances = distances[:, 1:]
        indices = indices[:, 1:]

    neighbor_labels = reference_labels[indices]
    results = recall_from_neighbor_labels(neighbor_labels, query_labels, ks)

    effective_k = min(classifier_k, neighbor_labels.shape[1])
    if effective_k < 1:
        results.update({"knn_acc": None, "knn_macro_f1": None})
        return results

    predictions = weighted_knn_predictions(
        neighbor_labels=neighbor_labels[:, :effective_k],
        neighbor_distances=distances[:, :effective_k],
    )
    results["knn_acc"] = float(accuracy_score(query_labels, predictions))
    results["knn_macro_f1"] = float(f1_score(query_labels, predictions, average="macro"))
    return results


def recall_from_neighbor_labels(
    neighbor_labels: np.ndarray,
    true_labels: np.ndarray,
    ks: tuple[int, ...],
) -> dict[str, float | None]:
    matches = neighbor_labels == true_labels[:, None]
    results: dict[str, float | None] = {}
    for rank in ks:
        key = f"R{rank}"
        if rank > neighbor_labels.shape[1]:
            results[key] = None
            continue
        results[key] = float(matches[:, :rank].any(axis=1).mean())
    return results


def weighted_knn_predictions(
    neighbor_labels: np.ndarray,
    neighbor_distances: np.ndarray,
) -> np.ndarray:
    predictions: list[str] = []
    for label_row, distance_row in zip(neighbor_labels, neighbor_distances, strict=True):
        votes: defaultdict[str, float] = defaultdict(float)
        for label, distance in zip(label_row, distance_row, strict=True):
            votes[str(label)] += 1.0 / max(float(distance), 1e-8)
        predictions.append(max(votes.items(), key=lambda item: item[1])[0])
    return np.asarray(predictions)
