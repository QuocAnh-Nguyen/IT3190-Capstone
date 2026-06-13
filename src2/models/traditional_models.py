"""Traditional ML baseline models — Multi-Class (Single-Label) Classification.

Phase 3.2 — Traditional ML Models (multi-class, post-improve_plan).

This module builds and evaluates classical multi-class classifiers:

1. **RandomForestClassifier**
   Ensemble of decision trees with ``class_weight='balanced'``.

2. **LogisticRegression**
   Multi-class LR with ``class_weight='balanced'`` and saga solver.

3. **XGBClassifier**
   Gradient-boosted trees using ``sample_weight`` for class imbalance.

4. **SVC**
   RBF-kernel SVM with ``class_weight='balanced'``.

5. **LGBMClassifier** (optional, depends on lightgbm availability)
   Gradient-boosted alternative to XGBoost, often faster.

All models are trained with K-Fold cross-validation on the training split,
then re-fitted on the full training set before final evaluation on the held-out
test split.  Trained models are persisted to ``MODELS_DIR`` via joblib.

Typical usage
-------------
>>> from src2.models.traditional_models import run_all_traditional_models
>>> results = run_all_traditional_models(
...     X_train, y_train, X_test, y_test, label_names
... )
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.base import clone

from src2.config import MODELS_DIR, N_CV_FOLDS, RANDOM_SEED
from src2.models.evaluation_multiclass import compute_metrics_multiclass, cross_validate_multiclass
from src2.utils.io_utils import save_joblib

logger = logging.getLogger("music_genre")

# ---------------------------------------------------------------------------
# Builder functions — each returns an *unfitted* model
# ---------------------------------------------------------------------------


def build_random_forest(
    n_estimators: int = 300,
    random_state: int = RANDOM_SEED,
) -> RandomForestClassifier:
    """Build a multi-class Random Forest with balanced class weights.

    Parameters
    ----------
    n_estimators : int
        Number of trees.  Increased from 200 to 300 for better convergence.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    RandomForestClassifier
        Unfitted classifier.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    logger.debug("Built RandomForest: n_estimators=%d, class_weight=balanced", n_estimators)
    return model


def build_logistic_regression(
    random_state: int = RANDOM_SEED,
) -> LogisticRegression:
    """Build a multi-class Logistic Regression with balanced class weights.

    Uses ``solver='saga'`` which supports large-scale data, elastic-net
    penalty, and multi-class problems natively.

    Parameters
    ----------
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    LogisticRegression
        Unfitted classifier.
    """
    model = LogisticRegression(
        max_iter=2000,
        C=1.0,
        solver="saga",
        penalty="l2",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    logger.debug("Built LogisticRegression: solver=saga, class_weight=balanced")
    return model


def build_xgboost(
    n_estimators: int = 300,
    max_depth: int = 8,
    learning_rate: float = 0.1,
    random_state: int = RANDOM_SEED,
) -> XGBClassifier:
    """Build a multi-class XGBoost classifier.

    XGBoost requires integer-encoded labels.  The ``train_and_evaluate_multiclass``
    function handles label encoding internally when ``use_sample_weight=True``.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.
    max_depth : int
        Maximum tree depth.  Increased from default 6 to 8 per improve_plan.
    learning_rate : float
        Step-size shrinkage.  Slightly increased from typical 0.05 to 0.1.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    XGBClassifier
        Unfitted classifier with ``objective='multi:softmax'``.
        Uses ``enable_categorical=False`` (we'll encode labels numerically).
    """
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        tree_method="hist",
        objective="multi:softmax",
        eval_metric="mlogloss",
        random_state=random_state,
        verbosity=0,
        n_jobs=-1,
    )
    logger.debug(
        "Built XGBoost: n_estimators=%d, max_depth=%d, lr=%.2f",
        n_estimators, max_depth, learning_rate,
    )
    return model


def build_svm(
    random_state: int = RANDOM_SEED,
) -> SVC:
    """Build an RBF-kernel SVM with balanced class weights.

    Note: SVM scales quadratically with samples.  For datasets >10K samples,
    training may be slow.  Use ``max_iter`` to bound convergence time.

    Returns
    -------
    SVC
        Unfitted classifier with ``probability=True`` for soft outputs.
    """
    model = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,  # needed for ROC curves, calibration, etc.
        random_state=random_state,
        max_iter=2000,
    )
    logger.debug("Built SVC: kernel=rbf, class_weight=balanced, probability=True")
    return model


def build_lightgbm(
    n_estimators: int = 300,
    max_depth: int = 8,
    learning_rate: float = 0.1,
    random_state: int = RANDOM_SEED,
):
    """Build a multi-class LightGBM classifier.

    LightGBM is often faster than XGBoost and can yield comparable accuracy.
    Falls back gracefully if ``lightgbm`` is not installed.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.
    max_depth : int
        Maximum tree depth.
    learning_rate : float
        Step-size shrinkage.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    LGBMClassifier or None
        Unfitted classifier, or None if lightgbm is not installed.
    """
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            class_weight="balanced",
            objective="multiclass",
            random_state=random_state,
            verbose=-1,
            n_jobs=-1,
        )
        logger.debug(
            "Built LightGBM: n_estimators=%d, max_depth=%d, lr=%.2f",
            n_estimators, max_depth, learning_rate,
        )
        return model
    except ImportError:
        logger.warning("LightGBM not installed — skipping LightGBM model.")
        return None


# ---------------------------------------------------------------------------
# SMOTE oversampling for minority classes
# ---------------------------------------------------------------------------


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k_neighbors: int = 5,
    target_min_samples: int = 500,
    cap_ratio: float = 10.0,
    random_state: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE oversampling to the training set for minority classes.

    Only classes with fewer than *target_min_samples* samples are oversampled.
    Each is capped at *cap_ratio* × its original size to prevent excessive
    synthetic dominance.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix of shape ``(n_samples, n_features)``.
    y_train : np.ndarray
        Training labels (string or int), shape ``(n_samples,)``.
    k_neighbors : int
        Number of nearest neighbors for SMOTE.
    target_min_samples : int
        Oversample each minority class up to this many total samples.
    cap_ratio : float
        Maximum multiplier on a class's original sample count.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    X_resampled : np.ndarray
        Oversampled feature matrix.
    y_resampled : np.ndarray
        Oversampled labels.
    """
    from collections import Counter
    from imblearn.over_sampling import SMOTE

    # Count per class
    counts = Counter(y_train)
    logger.info("Class distribution before SMOTE:")
    for label, cnt in sorted(counts.items(), key=lambda kv: -kv[1]):
        logger.info("  %-30s  %6d", label, cnt)

    # Build sampling strategy: only oversample minority classes
    sampling_strategy: dict = {}
    for label, cnt in counts.items():
        if cnt < target_min_samples:
            target = min(target_min_samples, int(cnt * cap_ratio))
            # SMOTE needs at least k_neighbors+1 samples
            if cnt >= k_neighbors + 1 and target > cnt:
                sampling_strategy[label] = target
            else:
                logger.info(
                    "Skipping SMOTE for '%s': only %d samples (need >%d for k=%d).",
                    label, cnt, k_neighbors, k_neighbors,
                )

    if not sampling_strategy:
        logger.info("No minority classes qualify for SMOTE oversampling.")
        return X_train, y_train

    sampling_info = {
        k: f"{counts.get(k, 0)} → {v}" for k, v in sampling_strategy.items()
    }
    logger.info("SMOTE sampling strategy: %s", sampling_info)

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state,
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)

    counts_after = Counter(y_res)
    logger.info("Class distribution after SMOTE:")
    for label, cnt in sorted(counts_after.items(), key=lambda kv: -kv[1]):
        logger.info("  %-30s  %6d  (+%d)", label, cnt,
                     cnt - counts.get(label, 0))

    logger.info("SMOTE: %d → %d samples", len(X_train), len(X_res))
    return X_res, y_res


# ---------------------------------------------------------------------------
# Training + evaluation (multi-class)
# ---------------------------------------------------------------------------


def train_and_evaluate_multiclass(
    model: Any,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
    output_dir: Path,
    n_folds: int = N_CV_FOLDS,
    use_sample_weight: bool = False,
) -> dict[str, Any]:
    """Train *model* with K-Fold CV, fit on full training set, evaluate on test.

    Workflow
    --------
    1. Run K-Fold cross-validation on training split (multi-class metrics).
    2. Fit the model on the full training set.
    3. Generate predictions on the test set and compute final test metrics.
    4. Persist the trained model.
    5. Return a dict with keys ``'cv_metrics'`` and ``'summary'``, etc.

    Parameters
    ----------
    model : Any
        Unfitted scikit-learn compatible estimator.
    model_name : str
        Short identifier (e.g. ``'random_forest'``).
    X_train : np.ndarray
        Training feature matrix, shape ``(n_train, n_features)``.
    y_train : np.ndarray
        Training labels (string or int), shape ``(n_train,)``.
    X_test : np.ndarray
        Test feature matrix, shape ``(n_test, n_features)``.
    y_test : np.ndarray
        Test labels, shape ``(n_test,)``.
    label_names : list[str]
        Ordered class names.
    output_dir : Path
        Directory for saving serialised models.
    n_folds : int
        Number of CV folds.
    use_sample_weight : bool
        If True, compute and pass ``sample_weight`` to ``model.fit()``
        (used for XGBoost which doesn't natively support ``class_weight``).

    Returns
    -------
    dict
        Dictionary with ``'cv_metrics'``, ``'summary'``, ``'per_class'``,
        ``'confusion_matrix'``, and ``'classification_report'``.
    """
    logger.info("=" * 60)
    logger.info("Training model: %s", model_name)
    logger.info("  X_train shape: %s  y_train shape: %s", X_train.shape, y_train.shape)
    logger.info("  X_test  shape: %s  y_test  shape: %s", X_test.shape, y_test.shape)
    logger.info("  CV folds: %d", n_folds)

    # ------------------------------------------------------------------
    # Step 1 – K-Fold cross-validation (SMOTE applied INSIDE each fold)
    # ------------------------------------------------------------------
    logger.info("[%s] Starting %d-fold cross-validation ...", model_name, n_folds)
    cv_start = time.perf_counter()
    try:
        cv_metrics = cross_validate_multiclass(
            model_fn=lambda: clone(model),
            X=X_train,
            y=y_train,
            n_folds=n_folds,
            random_state=RANDOM_SEED,
            label_names=label_names,
            smote_config={
                "k_neighbors": 3,  # reduced from 5 for tiny classes in folds
                "target_min_samples": 400,
                "cap_ratio": 5.0,
            },
        )
    except Exception as exc:
        logger.error("[%s] Cross-validation failed: %s", model_name, exc, exc_info=True)
        cv_metrics = {}
    cv_elapsed = time.perf_counter() - cv_start

    if cv_metrics:
        mean = cv_metrics.get("mean", {})
        std = cv_metrics.get("std", {})
        logger.info(
            "[%s] CV complete in %.1fs — F1-macro: %.4f +/- %.4f  |  Accuracy: %.4f +/- %.4f",
            model_name,
            cv_elapsed,
            mean.get("macro_f1", float("nan")),
            std.get("macro_f1", float("nan")),
            mean.get("accuracy", float("nan")),
            std.get("accuracy", float("nan")),
        )

    # ------------------------------------------------------------------
    # Step 2 – Label encoding for XGBoost (requires integer labels)
    # ------------------------------------------------------------------
    from sklearn.preprocessing import LabelEncoder

    le = None
    y_train_encoded = y_train
    y_test_encoded = y_test
    if use_sample_weight:
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)
        logger.info("[%s] Encoded string labels → integers (0..%d).",
                     model_name, len(le.classes_) - 1)

    fit_kwargs: dict[str, Any] = {}
    if use_sample_weight:
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weight = compute_sample_weight(
            class_weight="balanced", y=y_train_encoded,
        )
        fit_kwargs["sample_weight"] = sample_weight
        logger.info("[%s] Computed balanced sample_weight array.", model_name)

    # ------------------------------------------------------------------
    # Step 3 – Fit on full training set
    # ------------------------------------------------------------------
    logger.info("[%s] Fitting on full training set ...", model_name)
    fit_start = time.perf_counter()
    try:
        model.fit(X_train, y_train_encoded, **fit_kwargs)
    except Exception as exc:
        logger.error("[%s] Model fitting failed: %s", model_name, exc, exc_info=True)
        return {"cv_metrics": cv_metrics, "summary": {}}
    fit_elapsed = time.perf_counter() - fit_start
    logger.info("[%s] Training complete in %.1fs.", model_name, fit_elapsed)

    # ------------------------------------------------------------------
    # Step 4 – Evaluate on test set
    # ------------------------------------------------------------------
    logger.info("[%s] Evaluating on test set ...", model_name)
    try:
        y_pred_encoded = model.predict(X_test)
        # Decode XGBoost integer predictions back to string labels
        if le is not None:
            y_pred = le.inverse_transform(y_pred_encoded)
        else:
            y_pred = y_pred_encoded
    except Exception as exc:
        logger.error("[%s] Prediction failed: %s", model_name, exc, exc_info=True)
        return {"cv_metrics": cv_metrics, "summary": {}}

    test_metrics = compute_metrics_multiclass(
        y_true=y_test,
        y_pred=y_pred,
        label_names=label_names,
    )

    if test_metrics:
        summary = test_metrics.get("summary", {})
        logger.info(
            "[%s] Test metrics — Accuracy: %.4f  |  F1-macro: %.4f  |  F1-weighted: %.4f",
            model_name,
            summary.get("accuracy", float("nan")),
            summary.get("macro_f1", float("nan")),
            summary.get("weighted_f1", float("nan")),
        )
        # Per-label breakdown
        per_label: dict[str, float] = test_metrics.get("per_label_f1", {})
        if per_label:
            logger.info("[%s] Per-label F1 scores:", model_name)
            for lbl, score in sorted(per_label.items(), key=lambda kv: -kv[1]):
                logger.info("    %-30s %.4f", lbl, score)

    # ------------------------------------------------------------------
    # Step 5 – Persist trained model
    # ------------------------------------------------------------------
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{model_name}.joblib"
    try:
        save_joblib(model, model_path)
        logger.info("[%s] Model saved to: %s", model_name, model_path)
    except Exception as exc:
        logger.error(
            "[%s] Failed to save model to %s: %s", model_name, model_path, exc,
        )

    test_metrics["cv_metrics"] = cv_metrics
    return test_metrics


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_all_traditional_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
    output_dir: Path | None = None,
    apply_smote_flag: bool = True,
    smote_k_neighbors: int = 5,
    smote_target_min: int = 500,
) -> dict[str, dict[str, Any]]:
    """Build, train, and evaluate all traditional multi-class models.

    Iterates over:
    * ``'random_forest'``  — RandomForestClassifier (balanced)
    * ``'logistic_regression'`` — LogisticRegression (balanced, saga)
    * ``'xgboost'`` — XGBClassifier (sample_weight balanced)
    * ``'svm'`` — SVC (rbf, balanced)
    * ``'lightgbm'`` — LGBMClassifier (balanced, optional)

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels (string).
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        Test labels (string).
    label_names : list[str]
        Ordered class names.
    output_dir : Path or None
        Directory for saving serialised models.
    apply_smote_flag : bool
        If True, apply SMOTE oversampling to X_train/y_train.
    smote_k_neighbors : int
        k_neighbors for SMOTE.
    smote_target_min : int
        Target minimum samples for minority classes.

    Returns
    -------
    dict
        Mapping ``model_name -> metrics_dict``.
    """
    if output_dir is None:
        output_dir = MODELS_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Apply SMOTE if enabled (only for final full-training fit; CV
    # handles SMOTE internally per-fold to prevent leakage)
    # We use a lighter SMOTE here than in CV to reduce overfitting
    # ------------------------------------------------------------------
    X_train_smoted, y_train_smoted = X_train, y_train
    if apply_smote_flag:
        logger.info("Applying SMOTE oversampling for final training fit ...")
        try:
            X_train_smoted, y_train_smoted = apply_smote(
                X_train, y_train,
                k_neighbors=smote_k_neighbors,
                target_min_samples=smote_target_min,
                random_state=RANDOM_SEED,
            )
        except Exception as exc:
            logger.error("SMOTE failed: %s — continuing without.", exc)

    # ------------------------------------------------------------------
    # Model registry
    # ------------------------------------------------------------------
    model_registry: list[tuple[str, Any, bool]] = [
        ("random_forest",       build_random_forest(),                False),
        ("logistic_regression", build_logistic_regression(),           False),
        ("xgboost",             build_xgboost(),                       True),  # needs sample_weight
        ("svm",                 build_svm(),                           False),
    ]

    # Optionally add LightGBM
    lgbm_model = build_lightgbm()
    if lgbm_model is not None:
        model_registry.append(("lightgbm", lgbm_model, False))

    all_results: dict[str, dict[str, Any]] = {}
    overall_start = time.perf_counter()

    for model_name, model, use_sw in model_registry:
        logger.info("Processing model: %s", model_name)
        try:
            result = train_and_evaluate_multiclass(
                model=model,
                model_name=model_name,
                X_train=X_train_smoted,   # SMOTEd data for final full fit
                y_train=y_train_smoted,
                X_test=X_test,
                y_test=y_test,
                label_names=label_names,
                output_dir=output_dir,
                n_folds=N_CV_FOLDS,
                use_sample_weight=use_sw,
            )
        except Exception as exc:
            logger.error(
                "Unexpected error while running model '%s': %s",
                model_name, exc, exc_info=True,
            )
            result = {"cv_metrics": {}, "summary": {}}

        all_results[model_name] = result
        logger.info("Finished model: %s", model_name)

    overall_elapsed = time.perf_counter() - overall_start

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("TRADITIONAL MODELS SUMMARY  (total: %.1fs)", overall_elapsed)
    logger.info("=" * 60)
    header = (
        f"{'Model':<25} {'Acc':>8} {'F1-Macro':>10} {'F1-Weight':>10}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for name, res in all_results.items():
        s = res.get("summary", {})
        logger.info(
            "%-25s %8.4f %10.4f %10.4f",
            name,
            s.get("accuracy", float("nan")),
            s.get("macro_f1", float("nan")),
            s.get("weighted_f1", float("nan")),
        )
    logger.info("=" * 60)

    return all_results