"""Unsupervised clustering features via K-Means.

Groups tracks into acoustic clusters independent of human-assigned genres
and uses the resulting cluster IDs as categorical features.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import (
    ACOUSTIC_FEATURE_COLS,
    KMEANS_K_RANGE,
    KMEANS_RANDOM_STATE,
    RANDOM_SEED,
)
from src.utils.helpers import get_logger, save_joblib

logger = get_logger(__name__)


def _find_optimal_k(
    X: np.ndarray,
    k_range: range = KMEANS_K_RANGE,
    random_state: int = KMEANS_RANDOM_STATE,
) -> int:
    """Determine optimal K using the elbow method + silhouette score.

    Returns the K with the highest silhouette score.
    """
    logger.info(f"Searching for optimal K in {k_range}...")
    best_k = k_range.start
    best_score = -1

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        logger.info(f"  K={k:2d}  silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k

    logger.info(f"  Optimal K={best_k} (silhouette={best_score:.4f})")
    return best_k


def extract_cluster_features(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    n_clusters: Optional[int] = None,
    fit: bool = True,
    kmeans_model: Optional[KMeans] = None,
    scaler: Optional[StandardScaler] = None,
    save_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, List[str], Optional[KMeans], Optional[StandardScaler]]:
    """Run K-Means on acoustic features and produce cluster-based features.

    Features produced:
        - cluster_id: integer cluster assignment (then one-hot encoded)
        - cluster_distance: distance to assigned cluster centroid
          (measures how "typical" the song is for its cluster)

    Args:
        df: Input DataFrame.
        numeric_cols: Columns to use for clustering.
            Defaults to ACOUSTIC_FEATURE_COLS.
        n_clusters: Number of clusters. If None, determined by silhouette.
        fit: If True, fit new KMeans. If False, use provided kmeans_model.
        kmeans_model: Pre-fitted KMeans model.
        scaler: Pre-fitted StandardScaler for the clustering features.
        save_dir: Directory to save fitted models.

    Returns:
        (df with cluster features, cluster feature column names,
         KMeans model, StandardScaler)
    """
    logger.info("Extracting K-Means cluster features...")

    if numeric_cols is None:
        numeric_cols = [c for c in ACOUSTIC_FEATURE_COLS if c in df.columns]

    df = df.copy()
    X = df[numeric_cols].fillna(0).values

    # Scale before clustering
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    elif scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        raise ValueError("Either fit=True or scaler must be provided")

    # Determine K
    if fit and n_clusters is None:
        n_clusters = _find_optimal_k(X_scaled)
    elif n_clusters is None:
        n_clusters = kmeans_model.n_clusters

    logger.info(f"  Using K={n_clusters}")

    # Fit or predict
    if fit:
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=KMEANS_RANDOM_STATE,
            n_init=10,
            max_iter=300,
        )
        cluster_ids = kmeans.fit_predict(X_scaled)
    elif kmeans_model is not None:
        kmeans = kmeans_model
        cluster_ids = kmeans.predict(X_scaled)
    else:
        raise ValueError("Either fit=True or kmeans_model must be provided")

    # Cluster assignment
    df["cluster_id"] = cluster_ids

    # Distance to centroid (how "typical" the song is)
    centroids = kmeans.cluster_centers_
    distances = np.linalg.norm(X_scaled - centroids[cluster_ids], axis=1)
    df["cluster_distance"] = distances

    # One-hot encode cluster IDs
    cluster_dummies = pd.get_dummies(df["cluster_id"], prefix="cluster")
    df = pd.concat([df, cluster_dummies], axis=1)

    # Collect feature column names
    cluster_cols = ["cluster_distance"] + list(cluster_dummies.columns)

    logger.info(f"  Total cluster features: {len(cluster_cols)}")
    logger.info(f"  Cluster sizes: {pd.Series(cluster_ids).value_counts().sort_index().to_dict()}")

    if save_dir and fit:
        save_dir = Path(save_dir)
        save_joblib(kmeans, save_dir / "kmeans_model.joblib")
        save_joblib(scaler, save_dir / "cluster_scaler.joblib")
        logger.info(f"  Saved KMeans model & scaler → {save_dir}")

    return df, cluster_cols, kmeans if fit else None, scaler if fit else None