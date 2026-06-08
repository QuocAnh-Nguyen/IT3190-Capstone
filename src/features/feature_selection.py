"""Feature selection via Random Forest importance and PCA."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from src.config import (
    PCA_VARIANCE_THRESHOLD,
    RANDOM_SEED,
)
from src.utils.helpers import get_logger, save_joblib

logger = get_logger(__name__)


def rf_feature_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    top_k: Optional[int] = None,
    save_path: Optional[Path] = None,
) -> Tuple[List[str], pd.DataFrame]:
    """Rank features by Random Forest importance.

    Args:
        X: Feature DataFrame.
        y: Encoded target labels.
        top_k: If provided, return only the top-K feature names.
        save_path: If provided, save the importance DataFrame.

    Returns:
        (list of selected feature names sorted by importance descending,
         DataFrame with all features and their importance scores)
    """
    logger.info("Computing Random Forest feature importance...")

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X, y)

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance_df["cumulative"] = importance_df["importance"].cumsum()

    logger.info(f"  Top 10 features by importance:")
    for _, row in importance_df.head(10).iterrows():
        logger.info(f"    {row['feature']:<40s} {row['importance']:.4f}")

    if save_path:
        importance_df.to_csv(save_path, index=False)
        logger.info(f"  Saved importance → {save_path}")

    # Select top-K or all features above a threshold
    if top_k:
        selected = importance_df.head(top_k)["feature"].tolist()
    else:
        # Keep features explaining 95% of cumulative importance
        threshold_idx = (importance_df["cumulative"] >= 0.95).idxmax()
        selected = importance_df.loc[:threshold_idx, "feature"].tolist()

    logger.info(f"  Selected {len(selected)} / {len(X.columns)} features")
    return selected, importance_df


def apply_pca(
    X: pd.DataFrame,
    columns_to_reduce: List[str],
    variance_threshold: float = PCA_VARIANCE_THRESHOLD,
    fit: bool = True,
    pca_model: Optional[PCA] = None,
    prefix: str = "pca",
    save_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, List[str], Optional[PCA]]:
    """Apply PCA to a subset of columns, replacing them with principal components.

    Args:
        X: Feature DataFrame.
        columns_to_reduce: Which columns to apply PCA to.
        variance_threshold: Cumulative explained variance target.
        fit: If True, fit a new PCA model.
        pca_model: Pre-fitted PCA model (used when fit=False).
        prefix: Prefix for PCA column names.
        save_path: If provided and fitting, save the PCA model.

    Returns:
        (DataFrame with PCA components replacing original columns,
         list of new PCA column names,
         fitted PCA model or None)
    """
    logger.info(
        f"Applying PCA to {len(columns_to_reduce)} columns "
        f"(variance threshold={variance_threshold})..."
    )

    X_subset = X[columns_to_reduce].fillna(0)

    if fit:
        pca = PCA(n_components=variance_threshold, random_state=RANDOM_SEED)
        X_pca = pca.fit_transform(X_subset)
        logger.info(
            f"  PCA: {len(columns_to_reduce)} → {pca.n_components_} components "
            f"({pca.explained_variance_ratio_.sum():.1%} variance)"
        )
        if save_path:
            save_joblib(pca, save_path)
    elif pca_model is not None:
        pca = pca_model
        X_pca = pca.transform(X_subset)
    else:
        raise ValueError("Either fit=True or pca_model must be provided")

    n_components = pca.n_components_
    pca_cols = [f"{prefix}_{i}" for i in range(n_components)]

    # Drop original columns, add PCA components
    X_out = X.drop(columns=columns_to_reduce)
    pca_df = pd.DataFrame(X_pca, index=X_out.index, columns=pca_cols)
    X_out = pd.concat([X_out, pca_df], axis=1)

    logger.info(f"  Output features after PCA: {len(X_out.columns)}")
    return X_out, pca_cols, pca if fit else None


def select_features(
    X: pd.DataFrame,
    y: np.ndarray,
    nlp_feature_cols: List[str],
    top_k: Optional[int] = None,
    pca_on_nlp: bool = True,
    save_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, List[str], dict]:
    """Full feature selection pipeline.

    1. Apply PCA to high-dimensional NLP features (optional).
    2. Rank remaining features by Random Forest importance.
    3. Return reduced feature matrix and selection metadata.

    Returns:
        (reduced feature DataFrame, selected feature names, metadata dict)
    """
    logger.info("=" * 60)
    logger.info("Starting feature selection pipeline")
    logger.info("=" * 60)
    logger.info(f"  Input features: {len(X.columns)}")

    metadata = {}

    # Step 1: PCA on NLP TF-IDF features (high-dimensional)
    tfidf_cols = [c for c in nlp_feature_cols if c.startswith("nlp_tfidf_")]
    if pca_on_nlp and len(tfidf_cols) > 10:
        logger.info(f"\n--- PCA on NLP TF-IDF features ({len(tfidf_cols)} cols) ---")
        X, pca_cols, pca_model = apply_pca(
            X,
            columns_to_reduce=tfidf_cols,
            fit=True,
            prefix="nlp_pca",
            save_path=save_dir / "nlp_pca.joblib" if save_dir else None,
        )
        metadata["nlp_pca"] = {
            "pca_model": pca_model,
            "original_cols": tfidf_cols,
            "new_cols": pca_cols,
        }
        # Update nlp feature list
        nlp_feature_cols = [
            c for c in nlp_feature_cols if c not in tfidf_cols
        ] + pca_cols

    # Step 2: RF Feature Importance
    logger.info(f"\n--- RF Feature Importance ({len(X.columns)} features) ---")
    selected_cols, importance_df = rf_feature_importance(
        X, y,
        top_k=top_k,
        save_path=save_dir / "feature_importance.csv" if save_dir else None,
    )
    metadata["feature_importance"] = importance_df
    metadata["selected_features"] = selected_cols

    # Step 3: Return reduced feature matrix
    X_selected = X[selected_cols]
    logger.info(f"\nFeature selection complete: {len(X.columns)} → {len(selected_cols)} features")

    return X_selected, selected_cols, metadata