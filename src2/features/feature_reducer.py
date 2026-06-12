"""Feature Selection & Dimensionality Reduction — Phase 2.5 (EXECUTE).

This module provides all dimensionality-reduction and feature-selection
utilities consumed by the Phase 2.5 pipeline step.  It sits between raw
feature extraction (Phase 2) and model training (Phase 3), and its outputs
are NumPy arrays / scikit-learn transformers saved to ``FEATURES_DIR``.

Responsibilities
----------------
* **PCA reduction** – fit a ``sklearn.decomposition.PCA`` instance that
  retains at least ``PCA_VARIANCE_THRESHOLD`` of cumulative explained
  variance, then project the input matrix into that lower-dimensional space.
* **Tree-based feature importance** – train a ``RandomForestClassifier`` on
  a stratified sample and surface per-feature importances as a ranked
  ``pd.DataFrame``.  Multi-label targets are handled by either using the
  first label column or by averaging importances across all labels.
* **Top-k feature selection** – given an importance ranking, slice the input
  matrix to keep only the ``top_k`` most informative columns.
* **Late fusion** – column-wise concatenation of audio and text feature
  matrices with graceful handling of missing modalities.
* **Persistence helpers** – ``save_reducer`` / ``load_reducer`` wrap
  ``joblib`` so that fitted PCA transformers can be checkpointed and reloaded
  without re-fitting.

All artefacts (PCA transformer, importance tables) should be written into
``FEATURES_DIR`` by the calling pipeline script; this module only performs
computation and returns results — it does not write to disk on its own
(except through the explicit ``save_reducer`` helper).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from src2.config import FEATURES_DIR, PCA_VARIANCE_THRESHOLD, RANDOM_SEED
from src2.utils.io_utils import load_joblib, save_joblib

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("music_genre")

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------
# Maximum number of samples used to fit the RandomForest importance estimator.
# Training on the full dataset is expensive; this cap keeps it tractable.
_RF_SAMPLE_CAP: int = 10_000

# Number of jobs for the RandomForestClassifier.  -1 = use all CPU cores.
_RF_N_JOBS: int = -1


# ---------------------------------------------------------------------------
# 1. PCA Reduction
# ---------------------------------------------------------------------------

def fit_pca(
    X: np.ndarray,
    variance_threshold: float = PCA_VARIANCE_THRESHOLD,
) -> tuple[PCA, np.ndarray]:
    """Fit a PCA transformer that retains ``variance_threshold`` of variance.

    Parameters
    ----------
    X:
        2-D feature matrix of shape ``(n_samples, n_features)``.  Expected to
        be float-typed; integer arrays are cast automatically.
    variance_threshold:
        Fraction of cumulative explained variance to retain, in the range
        ``(0.0, 1.0]``.  Defaults to ``PCA_VARIANCE_THRESHOLD`` from config.

    Returns
    -------
    tuple[PCA, np.ndarray]
        A 2-tuple of:

        * ``pca`` – the fitted ``sklearn.decomposition.PCA`` instance, which
          can be used to ``transform`` unseen data.
        * ``X_reduced`` – the projected matrix of shape
          ``(n_samples, n_components)``.

    Raises
    ------
    ValueError
        If ``X`` is not a 2-D array or ``variance_threshold`` is outside
        ``(0, 1]``.
    """
    if X.ndim != 2:
        raise ValueError(
            f"fit_pca expects a 2-D array, got shape {X.shape}."
        )
    if not (0.0 < variance_threshold <= 1.0):
        raise ValueError(
            f"variance_threshold must be in (0, 1], got {variance_threshold}."
        )

    n_samples, n_input_features = X.shape
    logger.info(
        "Fitting PCA on matrix of shape (%d, %d) "
        "targeting %.1f%% explained variance ...",
        n_samples,
        n_input_features,
        variance_threshold * 100,
    )

    try:
        pca = PCA(n_components=variance_threshold, random_state=RANDOM_SEED)
        X_reduced: np.ndarray = pca.fit_transform(X.astype(float))
    except Exception as exc:
        logger.error("PCA fitting failed: %s", exc, exc_info=True)
        raise

    n_components = pca.n_components_
    explained = pca.explained_variance_ratio_.sum()
    logger.info(
        "PCA selected %d components out of %d input features "
        "(cumulative explained variance: %.4f).",
        n_components,
        n_input_features,
        explained,
    )
    return pca, X_reduced


# ---------------------------------------------------------------------------
# 2. Tree-based Feature Importance
# ---------------------------------------------------------------------------

def compute_feature_importance(
    X: np.ndarray,
    Y: np.ndarray,
    feature_names: list[str],
    n_estimators: int = 100,
) -> pd.DataFrame:
    """Train a RandomForest and return per-feature importances, ranked desc.

    For multi-label targets (``Y.ndim == 2``), the function handles label
    structure in two ways:

    * **Single column** – used as-is.
    * **Multiple columns** – a separate forest is fitted per label column and
      the importances are averaged across all labels.  This is more expensive
      but gives a holistic importance ranking that reflects the full label set.

    To keep runtime bounded, training is performed on a stratified random
    subsample of at most ``_RF_SAMPLE_CAP`` rows.

    Parameters
    ----------
    X:
        2-D feature matrix of shape ``(n_samples, n_features)``.
    Y:
        Target array of shape ``(n_samples,)`` (single-label) or
        ``(n_samples, n_labels)`` (multi-label).
    feature_names:
        List of feature names of length ``n_features``.  Must align with the
        columns of ``X``.
    n_estimators:
        Number of trees in each RandomForestClassifier.  Defaults to 100.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``["feature", "importance"]`` sorted by
        ``importance`` in descending order (index reset).

    Raises
    ------
    ValueError
        If ``len(feature_names) != X.shape[1]``.
    """
    if len(feature_names) != X.shape[1]:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) must equal "
            f"X.shape[1] ({X.shape[1]})."
        )

    n_samples = X.shape[0]
    logger.info(
        "Computing tree-based feature importance on %d samples, "
        "%d features, n_estimators=%d.",
        n_samples,
        X.shape[1],
        n_estimators,
    )

    # ------------------------------------------------------------------ #
    # Subsample if the dataset is very large
    # ------------------------------------------------------------------ #
    rng = np.random.default_rng(RANDOM_SEED)
    if n_samples > _RF_SAMPLE_CAP:
        idx = rng.choice(n_samples, size=_RF_SAMPLE_CAP, replace=False)
        X_sub = X[idx]
        Y_sub = Y[idx]
        logger.info(
            "Dataset capped to %d samples for RandomForest fitting.",
            _RF_SAMPLE_CAP,
        )
    else:
        X_sub = X
        Y_sub = Y

    # ------------------------------------------------------------------ #
    # Determine label columns
    # ------------------------------------------------------------------ #
    if Y_sub.ndim == 1 or (Y_sub.ndim == 2 and Y_sub.shape[1] == 1):
        label_columns: list[np.ndarray] = [Y_sub.ravel()]
    else:
        label_columns = [Y_sub[:, col] for col in range(Y_sub.shape[1])]

    logger.info(
        "Fitting RandomForest across %d label column(s) ...",
        len(label_columns),
    )

    accumulated_importances = np.zeros(X_sub.shape[1], dtype=float)
    fitted_count = 0

    for col_idx, y_col in enumerate(label_columns):
        # Skip columns that have only a single unique class (RF will fail).
        unique_classes = np.unique(y_col)
        if unique_classes.size < 2:
            logger.warning(
                "Label column %d has only one unique class; skipping.",
                col_idx,
            )
            continue

        try:
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=_RF_N_JOBS,
            )
            rf.fit(X_sub, y_col)
            accumulated_importances += rf.feature_importances_
            fitted_count += 1
        except Exception as exc:
            logger.error(
                "RandomForest failed for label column %d: %s",
                col_idx,
                exc,
                exc_info=True,
            )

    if fitted_count == 0:
        logger.warning(
            "No label column yielded a valid RandomForest fit. "
            "Returning uniform importances."
        )
        mean_importances = np.ones(X_sub.shape[1]) / X_sub.shape[1]
    else:
        mean_importances = accumulated_importances / fitted_count

    importances_df = (
        pd.DataFrame(
            {"feature": feature_names, "importance": mean_importances}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    logger.info(
        "Top-5 features by importance:\n%s",
        importances_df.head(5).to_string(index=False),
    )
    return importances_df


# ---------------------------------------------------------------------------
# 3. Feature Selection by Importance
# ---------------------------------------------------------------------------

def select_top_features(
    X: np.ndarray,
    importances_df: pd.DataFrame,
    top_k: int,
) -> tuple[np.ndarray, list[str]]:
    """Slice ``X`` to the ``top_k`` most important features.

    Parameters
    ----------
    X:
        2-D feature matrix of shape ``(n_samples, n_features)``.
    importances_df:
        DataFrame returned by :func:`compute_feature_importance`, with columns
        ``["feature", "importance"]``, sorted descending by importance.
    top_k:
        Number of top features to retain.  Clamped to
        ``min(top_k, n_features)`` if the requested value exceeds the
        available number of features.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        A 2-tuple of:

        * ``X_reduced`` – array of shape ``(n_samples, top_k)``.
        * ``selected_names`` – ordered list of the selected feature names.

    Raises
    ------
    ValueError
        If ``top_k < 1`` or ``importances_df`` does not have a ``"feature"``
        column.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}.")
    if "feature" not in importances_df.columns:
        raise ValueError(
            "importances_df must contain a 'feature' column."
        )

    n_features = X.shape[1]
    effective_k = min(top_k, n_features)
    if effective_k < top_k:
        logger.warning(
            "top_k=%d exceeds available features (%d); clamping to %d.",
            top_k,
            n_features,
            effective_k,
        )

    # The DataFrame is already sorted descending; take the first top_k names.
    selected_names: list[str] = (
        importances_df["feature"].iloc[:effective_k].tolist()
    )

    # Build a position index for fast column slicing.
    # We map feature name -> column index using the full feature list embedded
    # in importances_df (which covers all n_features columns).
    all_feature_names: list[str] = importances_df["feature"].tolist()
    name_to_idx: dict[str, int] = {
        name: idx for idx, name in enumerate(all_feature_names)
    }

    try:
        col_indices: list[int] = [name_to_idx[name] for name in selected_names]
    except KeyError as exc:
        logger.error(
            "Feature name not found in importances_df: %s", exc, exc_info=True
        )
        raise

    X_reduced = X[:, col_indices]
    logger.info(
        "Selected top-%d features out of %d; reduced matrix shape: %s.",
        effective_k,
        n_features,
        X_reduced.shape,
    )
    return X_reduced, selected_names


# ---------------------------------------------------------------------------
# 4. Late Fusion
# ---------------------------------------------------------------------------

def fuse_features(
    audio_features: Optional[np.ndarray],
    text_features: Optional[np.ndarray],
) -> np.ndarray:
    """Concatenate audio and text feature matrices column-wise (late fusion).

    Either modality may be ``None`` — in that case the non-null modality is
    returned as-is.  If both are ``None`` a ``ValueError`` is raised.

    Parameters
    ----------
    audio_features:
        2-D array of shape ``(n_samples, n_audio_features)`` or ``None``.
    text_features:
        2-D array of shape ``(n_samples, n_text_features)`` or ``None``.

    Returns
    -------
    np.ndarray
        Fused matrix of shape
        ``(n_samples, n_audio_features + n_text_features)`` when both
        modalities are present, or the single non-null matrix otherwise.

    Raises
    ------
    ValueError
        If both ``audio_features`` and ``text_features`` are ``None``.
    RuntimeError
        If the two matrices have a mismatched number of rows.
    """
    if audio_features is None and text_features is None:
        raise ValueError(
            "Both audio_features and text_features are None; "
            "at least one modality must be provided."
        )

    if audio_features is None:
        logger.warning(
            "audio_features is None; returning text_features only "
            "(shape: %s).",
            text_features.shape,  # type: ignore[union-attr]
        )
        return text_features  # type: ignore[return-value]

    if text_features is None:
        logger.warning(
            "text_features is None; returning audio_features only "
            "(shape: %s).",
            audio_features.shape,
        )
        return audio_features

    if audio_features.shape[0] != text_features.shape[0]:
        raise RuntimeError(
            f"Row count mismatch: audio_features has {audio_features.shape[0]} "
            f"rows, text_features has {text_features.shape[0]} rows."
        )

    try:
        fused: np.ndarray = np.hstack([audio_features, text_features])
    except Exception as exc:
        logger.error(
            "Feature fusion (hstack) failed: %s", exc, exc_info=True
        )
        raise

    logger.info(
        "Late fusion complete — audio %s + text %s -> fused %s.",
        audio_features.shape,
        text_features.shape,
        fused.shape,
    )
    return fused


# ---------------------------------------------------------------------------
# 5. Save / Load Reducer
# ---------------------------------------------------------------------------

def save_reducer(pca: PCA, path: Path) -> None:
    """Persist a fitted PCA transformer to disk via joblib.

    The parent directory is created automatically if it does not exist.

    Parameters
    ----------
    pca:
        A fitted ``sklearn.decomposition.PCA`` instance.
    path:
        Destination file path
        (e.g. ``FEATURES_DIR / "pca_reducer.joblib"``).
    """
    try:
        path = Path(path)
        save_joblib(pca, path)
        logger.info(
            "PCA reducer saved to '%s' (n_components=%d).",
            path,
            pca.n_components_,
        )
    except Exception as exc:
        logger.error(
            "Failed to save PCA reducer to '%s': %s",
            path,
            exc,
            exc_info=True,
        )
        raise


def load_reducer(path: Path) -> PCA:
    """Load a joblib-serialised PCA transformer from disk.

    Parameters
    ----------
    path:
        Path to the ``.joblib`` file previously written by
        :func:`save_reducer`.

    Returns
    -------
    PCA
        The deserialised, fitted ``sklearn.decomposition.PCA`` instance.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No reducer artefact found at '{path}'. "
            "Run the Phase 2.5 pipeline to generate it."
        )
    try:
        pca: PCA = load_joblib(path)
        logger.info(
            "PCA reducer loaded from '%s' (n_components=%s).",
            path,
            getattr(pca, "n_components_", "unknown"),
        )
        return pca
    except Exception as exc:
        logger.error(
            "Failed to load PCA reducer from '%s': %s",
            path,
            exc,
            exc_info=True,
        )
        raise
