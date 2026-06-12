"""Traditional ML baseline models for Multi-Modal Music Genre Classification.

Phase 3.2 — Traditional ML Baseline Models (EXECUTE).

This module builds and evaluates three classical multi-label classifiers:

1. **OneVsRestClassifier(RandomForestClassifier)**
   A per-label forest ensemble; robust to class imbalance via ``class_weight``.

2. **ClassifierChain(LogisticRegression)**
   Exploits inter-label correlations by treating each label's prediction as an
   additional feature for the next classifier in the chain.

3. **OneVsRestClassifier(XGBClassifier)**
   Gradient-boosted trees wrapped per-label; fast histogram-based training.

All models are trained with K-Fold cross-validation on the training split and
then re-fitted on the full training set before final evaluation on the held-out
test split.  Trained models are persisted to ``MODELS_DIR`` via joblib.

Typical usage
-------------
>>> from src2.models.traditional_models import run_all_traditional_models
>>> results = run_all_traditional_models(
...     X_train, Y_train, X_test, Y_test, label_names, output_dir
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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from xgboost import XGBClassifier

from src2.config import MODELS_DIR, N_CV_FOLDS, RANDOM_SEED
from src2.models.evaluation import compute_metrics, cross_validate_multilabel
from src2.utils.io_utils import save_joblib

logger = logging.getLogger("music_genre")

# ---------------------------------------------------------------------------
# Builder functions — each returns an *unfitted* model object
# ---------------------------------------------------------------------------


def build_ovr_rf(
    n_estimators: int = 200,
    random_state: int = 42,
    class_weight: str = "balanced",
) -> OneVsRestClassifier:
    """Build an unfitted OneVsRest Random Forest classifier.

    Parameters
    ----------
    n_estimators:
        Number of trees in each per-label forest.
    random_state:
        Seed for reproducibility.
    class_weight:
        Strategy for handling class imbalance inside each tree.  ``'balanced'``
        inversely weights classes by their frequency.

    Returns
    -------
    OneVsRestClassifier
        Unfitted OneVsRest wrapper around a RandomForestClassifier.
    """
    base = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1,
    )
    model = OneVsRestClassifier(estimator=base, n_jobs=-1)
    logger.debug(
        "Built OVR-RF: n_estimators=%d, class_weight=%s, random_state=%d",
        n_estimators,
        class_weight,
        random_state,
    )
    return model


def build_classifier_chain(
    base_estimator: Any | None = None,
    random_state: int = 42,
) -> ClassifierChain:
    """Build an unfitted ClassifierChain with Logistic Regression as default base.

    Parameters
    ----------
    base_estimator:
        Any scikit-learn compatible estimator.  When ``None``, defaults to
        ``LogisticRegression(max_iter=1000, C=1.0, solver='saga', n_jobs=-1)``.
    random_state:
        Seed used to determine the label ordering in the chain.

    Returns
    -------
    ClassifierChain
        Unfitted ClassifierChain wrapping *base_estimator*.
    """
    if base_estimator is None:
        base_estimator = LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="saga",
            n_jobs=-1,
        )
    model = ClassifierChain(
        base_estimator=base_estimator,
        order="random",
        random_state=random_state,
    )
    logger.debug(
        "Built ClassifierChain: base=%s, random_state=%d",
        type(base_estimator).__name__,
        random_state,
    )
    return model


def build_ovr_xgb(
    n_estimators: int = 200,
    random_state: int = 42,
) -> OneVsRestClassifier:
    """Build an unfitted OneVsRest XGBoost classifier.

    Uses histogram-based tree construction (``tree_method='hist'``) for faster
    training on dense feature matrices, and ``eval_metric='logloss'`` as the
    internal evaluation criterion.

    Parameters
    ----------
    n_estimators:
        Number of boosting rounds per per-label XGBoost estimator.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    OneVsRestClassifier
        Unfitted OneVsRest wrapper around an XGBClassifier.
    """
    base = XGBClassifier(
        n_estimators=n_estimators,
        tree_method="hist",
        eval_metric="logloss",
        random_state=random_state,
        verbosity=0,
        use_label_encoder=False,
        n_jobs=-1,
    )
    model = OneVsRestClassifier(estimator=base, n_jobs=1)
    logger.debug(
        "Built OVR-XGB: n_estimators=%d, tree_method=hist, random_state=%d",
        n_estimators,
        random_state,
    )
    return model


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------


def train_and_evaluate(
    model: Any,
    model_name: str,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    label_names: list[str],
    output_dir: Path,
    n_folds: int = 5,
) -> dict[str, Any]:
    """Train *model* with K-Fold CV, fit on full training set, evaluate on test.

    Workflow
    --------
    1. Run ``cross_validate_multilabel`` on the training split to obtain CV
       metrics (mean ± std across folds).
    2. Fit the model on the full training set.
    3. Generate predictions on the test set and compute final test metrics via
       ``compute_metrics``.
    4. Persist the trained model to ``output_dir / f"{model_name}.joblib"``.
    5. Return a dict with keys ``'cv_metrics'`` and ``'test_metrics'``.

    Parameters
    ----------
    model:
        An unfitted scikit-learn–compatible estimator.
    model_name:
        Short, filesystem-safe identifier (e.g. ``'ovr_rf'``).
    X_train:
        Training feature matrix, shape ``(n_train, n_features)``.
    Y_train:
        Binary label matrix for training, shape ``(n_train, n_labels)``.
    X_test:
        Test feature matrix, shape ``(n_test, n_features)``.
    Y_test:
        Binary label matrix for test, shape ``(n_test, n_labels)``.
    label_names:
        Ordered list of genre label strings (length ``n_labels``).
    output_dir:
        Directory where the serialised model file will be written.
    n_folds:
        Number of folds for K-Fold cross-validation.

    Returns
    -------
    dict
        ``{'cv_metrics': dict, 'test_metrics': dict}`` where each inner dict
        contains F1-macro, F1-micro, subset accuracy, Hamming loss, etc.
    """
    logger.info("=" * 60)
    logger.info("Training model: %s", model_name)
    logger.info("  X_train shape : %s  Y_train shape : %s", X_train.shape, Y_train.shape)
    logger.info("  X_test  shape : %s  Y_test  shape : %s", X_test.shape, Y_test.shape)
    logger.info("  CV folds      : %d", n_folds)

    # ------------------------------------------------------------------
    # Step 1 – K-Fold cross-validation on the training set
    # ------------------------------------------------------------------
    logger.info("[%s] Starting %d-fold cross-validation ...", model_name, n_folds)
    cv_start = time.perf_counter()
    try:
        cv_metrics = cross_validate_multilabel(
            model=model,
            X=X_train,
            Y=Y_train,
            n_folds=n_folds,
            random_state=RANDOM_SEED,
        )
    except Exception as exc:
        logger.error("[%s] Cross-validation failed: %s", model_name, exc, exc_info=True)
        cv_metrics = {}
    cv_elapsed = time.perf_counter() - cv_start

    if cv_metrics:
        logger.info(
            "[%s] CV complete in %.1fs — F1-macro: %.4f +/- %.4f  |  F1-micro: %.4f +/- %.4f",
            model_name,
            cv_elapsed,
            cv_metrics.get("mean_f1_macro", float("nan")),
            cv_metrics.get("std_f1_macro", float("nan")),
            cv_metrics.get("mean_f1_micro", float("nan")),
            cv_metrics.get("std_f1_micro", float("nan")),
        )
    else:
        logger.warning("[%s] CV metrics could not be collected.", model_name)

    # ------------------------------------------------------------------
    # Step 2 – Fit on full training set
    # ------------------------------------------------------------------
    logger.info("[%s] Fitting on full training set ...", model_name)
    fit_start = time.perf_counter()
    try:
        model.fit(X_train, Y_train)
    except Exception as exc:
        logger.error("[%s] Model fitting failed: %s", model_name, exc, exc_info=True)
        return {"cv_metrics": cv_metrics, "test_metrics": {}}
    fit_elapsed = time.perf_counter() - fit_start
    logger.info("[%s] Training complete in %.1fs.", model_name, fit_elapsed)

    # ------------------------------------------------------------------
    # Step 3 – Evaluate on test set
    # ------------------------------------------------------------------
    logger.info("[%s] Evaluating on test set ...", model_name)
    try:
        Y_pred = model.predict(X_test)
    except Exception as exc:
        logger.error("[%s] Prediction failed: %s", model_name, exc, exc_info=True)
        return {"cv_metrics": cv_metrics, "test_metrics": {}}

    try:
        test_metrics = compute_metrics(
            Y_true=Y_test,
            Y_pred=Y_pred,
            label_names=label_names,
        )
    except Exception as exc:
        logger.error("[%s] Metric computation failed: %s", model_name, exc, exc_info=True)
        test_metrics = {}

    if test_metrics:
        logger.info(
            "[%s] Test metrics — F1-macro: %.4f  |  F1-micro: %.4f  |  "
            "Subset acc: %.4f  |  Hamming loss: %.4f",
            model_name,
            test_metrics.get("f1_macro", float("nan")),
            test_metrics.get("f1_micro", float("nan")),
            test_metrics.get("subset_accuracy", float("nan")),
            test_metrics.get("hamming_loss", float("nan")),
        )
        # Per-label breakdown
        per_label: dict[str, float] = test_metrics.get("per_label_f1", {})
        if per_label:
            logger.info("[%s] Per-label F1 scores:", model_name)
            for lbl, score in sorted(per_label.items(), key=lambda kv: -kv[1]):
                logger.info("    %-30s %.4f", lbl, score)

    # ------------------------------------------------------------------
    # Step 4 – Persist trained model
    # ------------------------------------------------------------------
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{model_name}.joblib"
    try:
        save_joblib(model, model_path)
        logger.info("[%s] Model saved to: %s", model_name, model_path)
    except Exception as exc:
        logger.error(
            "[%s] Failed to save model to %s: %s", model_name, model_path, exc, exc_info=True
        )

    return {
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_all_traditional_models(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    label_names: list[str],
    output_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Build, train, and evaluate all three traditional multi-label models.

    Iterates over:

    * ``'ovr_rf'``    — OneVsRest Random Forest
    * ``'chain_lr'``  — Classifier Chain (Logistic Regression)
    * ``'ovr_xgb'``   — OneVsRest XGBoost

    Each model is trained and evaluated via :func:`train_and_evaluate`.

    Parameters
    ----------
    X_train:
        Training feature matrix, shape ``(n_train, n_features)``.
    Y_train:
        Binary label matrix for training, shape ``(n_train, n_labels)``.
    X_test:
        Test feature matrix, shape ``(n_test, n_features)``.
    Y_test:
        Binary label matrix for test, shape ``(n_test, n_labels)``.
    label_names:
        Ordered list of genre label strings (length ``n_labels``).
    output_dir:
        Directory for saving serialised models.  Defaults to ``MODELS_DIR``
        from :mod:`src2.config`.

    Returns
    -------
    dict
        Mapping ``model_name -> {'cv_metrics': ..., 'test_metrics': ...}``.
    """
    if output_dir is None:
        output_dir = MODELS_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model registry: (name, unfitted model instance)
    model_registry: list[tuple[str, Any]] = [
        ("ovr_rf",   build_ovr_rf(n_estimators=200, random_state=RANDOM_SEED)),
        ("chain_lr", build_classifier_chain(random_state=RANDOM_SEED)),
        ("ovr_xgb",  build_ovr_xgb(n_estimators=200, random_state=RANDOM_SEED)),
    ]

    all_results: dict[str, dict[str, Any]] = {}
    overall_start = time.perf_counter()

    for model_name, model in model_registry:
        logger.info("Processing model: %s", model_name)
        try:
            result = train_and_evaluate(
                model=model,
                model_name=model_name,
                X_train=X_train,
                Y_train=Y_train,
                X_test=X_test,
                Y_test=Y_test,
                label_names=label_names,
                output_dir=output_dir,
                n_folds=N_CV_FOLDS,
            )
        except Exception as exc:
            logger.error(
                "Unexpected error while running model '%s': %s",
                model_name,
                exc,
                exc_info=True,
            )
            result = {"cv_metrics": {}, "test_metrics": {}}

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
        f"{'Model':<15} {'F1-Macro':>10} {'F1-Micro':>10} "
        f"{'SubsetAcc':>11} {'HammingL':>10}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for name, res in all_results.items():
        tm = res.get("test_metrics", {})
        logger.info(
            "%-15s %10.4f %10.4f %11.4f %10.4f",
            name,
            tm.get("f1_macro", float("nan")),
            tm.get("f1_micro", float("nan")),
            tm.get("subset_accuracy", float("nan")),
            tm.get("hamming_loss", float("nan")),
        )
    logger.info("=" * 60)

    return all_results
