"""Model building: training, hyperparameter tuning, and evaluation.

Implements the PRD's dual-stream hybrid architecture:
  Stream 1: Numerical features (acoustic + graph + interaction)
  Stream 2: Text-derived features (NLP + sentiment + TF-IDF PCA)
  Fusion: Concatenate both streams → predict genre.

Models (in progression):
  1. Logistic Regression (baseline)
  2. Random Forest
  3. XGBoost
  4. MLP Neural Network (optional/stretch)

Evaluation:
  - Stratified K-Fold Cross-Validation
  - Accuracy, Precision, Recall, F1 (Macro/Weighted)
  - Confusion Matrix
  - Classification Report per genre
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import (
    CV_FOLDS,
    MLP_HIDDEN_LAYERS,
    RANDOM_SEED,
    RF_N_ESTIMATORS,
    XGB_N_ESTIMATORS,
)
from src.utils.helpers import get_logger, save_joblib, save_json

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _get_scoring_metrics() -> Dict[str, str]:
    """Return scoring dict for cross_validate."""
    return {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
    }


def _evaluate_model(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    model_name: str,
) -> dict:
    """Fit and evaluate a single model on a train/test split.

    Returns dict of metrics.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "n_samples": len(X_test),
    }

    return results


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_models(
    X: np.ndarray,
    y: np.ndarray,
    label_encoder: LabelEncoder,
    n_splits: int = CV_FOLDS,
) -> pd.DataFrame:
    """Run stratified K-fold CV on all models and return comparison table.

    Args:
        X: Feature matrix (n_samples × n_features).
        y: Encoded labels (0..n_classes-1).
        label_encoder: Fitted LabelEncoder.
        n_splits: Number of CV folds.

    Returns:
        DataFrame with mean ± std metrics per model.
    """
    logger.info("=" * 60)
    logger.info(f"STRATIFIED {n_splits}-FOLD CROSS-VALIDATION")
    logger.info("=" * 60)
    logger.info(f"  Samples: {len(X):,}  Features: {X.shape[1]}  Classes: {label_encoder.classes_.shape[0]}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    scoring = _get_scoring_metrics()

    # Define models to evaluate (each wrapped with SMOTE in a pipeline
    # so oversampling happens per-fold, preventing data leakage)
    models = {
        "Logistic Regression": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_SEED, k_neighbors=5)),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )),
        ]),
        "Random Forest": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_SEED, k_neighbors=5)),
            ("clf", RandomForestClassifier(
                n_estimators=RF_N_ESTIMATORS,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )),
        ]),
        "XGBoost": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_SEED, k_neighbors=5)),
            ("clf", XGBClassifier(
                n_estimators=XGB_N_ESTIMATORS,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective="multi:softmax",
                eval_metric="mlogloss",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )),
        ]),
        "MLP": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_SEED, k_neighbors=5)),
            ("clf", MLPClassifier(
                hidden_layer_sizes=MLP_HIDDEN_LAYERS,
                max_iter=500,
                early_stopping=True,
                random_state=RANDOM_SEED,
            )),
        ]),
    }

    all_fold_results = []
    cv_summary = []

    for name, model in models.items():
        logger.info(f"\n--- {name} ---")

        # Run CV
        cv_results = cross_validate(
            model,
            X, y,
            cv=skf,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1,
        )

        # Collect fold-level results
        for fold in range(n_splits):
            fold_result = {"model": name, "fold": fold}
            for metric in scoring:
                key = f"test_{metric}"
                fold_result[metric] = cv_results[key][fold]
            all_fold_results.append(fold_result)

        # Compute mean ± std
        summary = {"model": name}
        metric_lines = []
        for metric in scoring:
            key = f"test_{metric}"
            mean_val = cv_results[key].mean()
            std_val = cv_results[key].std()
            summary[metric] = f"{mean_val:.4f} ± {std_val:.4f}"
            metric_lines.append(f"    {metric:<20s} = {mean_val:.4f} ± {std_val:.4f}")

        logger.info("\n".join(metric_lines))
        cv_summary.append(summary)

    summary_df = pd.DataFrame(cv_summary)
    fold_df = pd.DataFrame(all_fold_results)

    # Print summary table
    logger.info("\n" + "-" * 60)
    logger.info("CV SUMMARY (mean ± std)")
    logger.info("-" * 60)
    for _, row in summary_df.iterrows():
        logger.info(f"\n  {row['model']}:")
        for metric in scoring:
            logger.info(f"    {metric:<20s} {row[metric]}")

    return summary_df, fold_df


# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

def train_best_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    best_model_name: str,
) -> Tuple[object, dict, np.ndarray, np.ndarray]:
    """Train the best model on the full training set and evaluate on test set.

    Returns:
        (trained model, metrics dict, y_true, y_pred)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL EVALUATION — {best_model_name}")
    logger.info(f"{'='*60}")

    # Re-instantiate the best model
    if best_model_name == "Logistic Regression":
        model = LogisticRegression(
            max_iter=2000, class_weight="balanced",
            random_state=RANDOM_SEED, n_jobs=-1,
        )
    elif best_model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, class_weight="balanced",
            random_state=RANDOM_SEED, n_jobs=-1,
        )
    elif best_model_name == "XGBoost":
        model = XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS, learning_rate=0.1, max_depth=6,
            objective="multi:softmax", eval_metric="mlogloss",
            random_state=RANDOM_SEED, n_jobs=-1,
        )
    elif best_model_name == "MLP":
        model = MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN_LAYERS, max_iter=500,
            early_stopping=True, random_state=RANDOM_SEED,
        )
    else:
        raise ValueError(f"Unknown model: {best_model_name}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    logger.info(f"  Test Accuracy:   {metrics['accuracy']:.4f}")
    logger.info(f"  Test F1 (Macro):  {metrics['f1_macro']:.4f}")
    logger.info(f"  Test F1 (Weighted): {metrics['f1_weighted']:.4f}")

    # Classification report
    genre_names = label_encoder.classes_
    report = classification_report(
        y_test, y_pred, target_names=genre_names, zero_division=0
    )
    logger.info(f"\nClassification Report:\n{report}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    return model, metrics, cm, y_pred, y_proba


# ---------------------------------------------------------------------------
# Training wrapper for Phase 3 pipeline
# ---------------------------------------------------------------------------

def train_models(
    df: pd.DataFrame,
    target_col: str = "main_genre",
    encoded_col: str = "genre_encoded",
    test_size: float = 0.2,
    save_dir: Optional[Path] = None,
) -> dict:
    """Full model training pipeline.

    Args:
        df: Feature matrix DataFrame with target column.
        target_col: Name of the genre label column.
        encoded_col: Name of the pre-encoded label column.
        test_size: Fraction for hold-out test set.
        save_dir: Directory to save trained models and results.

    Returns:
        Dict with all results: models, metrics, predictions, etc.
    """
    logger.info("\n" + "=" * 60)
    logger.info("MODEL TRAINING PIPELINE")
    logger.info("=" * 60)

    # Separate features and target
    feature_cols = [c for c in df.columns if c not in (target_col, encoded_col)]
    X_all = df[feature_cols].values
    y_raw = df[target_col]
    y_encoded = df[encoded_col].values

    # Label encoder
    label_enc = LabelEncoder()
    label_enc.fit(y_raw)
    n_classes = len(label_enc.classes_)
    logger.info(f"  Classes: {n_classes}")

    # Scale features
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)

    # Train/Test split (stratified)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_encoded,
        test_size=test_size,
        stratify=y_encoded,
        random_state=RANDOM_SEED,
    )
    logger.info(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

    # ---- Phase 3a: Cross-Validation ----
    cv_summary, cv_folds = cross_validate_models(
        X_train, y_train, label_enc,
        n_splits=CV_FOLDS,
    )

    # Pick best model by F1 macro
    best_idx = cv_summary["f1_macro"].str.extract(r"([\d.]+)")[0].astype(float).idxmax()
    best_model_name = cv_summary.loc[best_idx, "model"]
    logger.info(f"\n  Best model: {best_model_name}")

    # ---- Phase 3b: Final evaluation ----
    best_model, metrics, cm, y_pred, y_proba = train_best_model(
        X_train, X_test, y_train, y_test, label_enc, best_model_name,
    )

    # ---- Phase 3c: Train best model on ALL data (for deployment) ----
    logger.info(f"\nTraining final {best_model_name} on all data for deployment...")
    if best_model_name == "Logistic Regression":
        deploy_model = LogisticRegression(
            max_iter=2000, class_weight="balanced",
            random_state=RANDOM_SEED, n_jobs=-1,
        )
    elif best_model_name == "Random Forest":
        deploy_model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, class_weight="balanced",
            random_state=RANDOM_SEED, n_jobs=-1,
        )
    elif best_model_name == "XGBoost":
        deploy_model = XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS, learning_rate=0.1, max_depth=6,
            objective="multi:softmax", eval_metric="mlogloss",
            random_state=RANDOM_SEED, n_jobs=-1,
        )
    else:
        deploy_model = MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN_LAYERS, max_iter=500,
            early_stopping=True, random_state=RANDOM_SEED,
        )
    deploy_model.fit(X_all, y_encoded)
    deploy_acc = deploy_model.score(X_all, y_encoded)
    logger.info(f"  Training accuracy (full data): {deploy_acc:.4f}")

    # ---- Save ----
    if save_dir:
        save_dir = Path(save_dir)
        save_joblib(deploy_model, save_dir / "best_model.joblib")
        save_joblib(scaler, save_dir / "feature_scaler.joblib")
        save_joblib(label_enc, save_dir / "label_encoder.joblib")
        save_joblib(feature_cols, save_dir / "feature_columns.joblib")
        cv_summary.to_csv(save_dir / "cv_summary.csv", index=False)
        np.save(save_dir / "confusion_matrix.npy", cm)
        save_json(
            {"best_model": best_model_name, "metrics": metrics, "n_classes": n_classes},
            save_dir / "final_metrics.json",
        )
        logger.info(f"\nAll models and artifacts saved → {save_dir}")

    return {
        "best_model_name": best_model_name,
        "best_model": best_model,
        "deploy_model": deploy_model,
        "scaler": scaler,
        "label_encoder": label_enc,
        "feature_cols": feature_cols,
        "cv_summary": cv_summary,
        "cv_folds": cv_folds,
        "metrics": metrics,
        "confusion_matrix": cm,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }