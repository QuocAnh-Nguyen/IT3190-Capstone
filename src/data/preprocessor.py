"""Data preprocessing: imputation, scaling, SMOTE, and train/val/test splitting."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.config import (
    ACOUSTIC_FEATURE_COLS,
    CV_FOLDS,
    RANDOM_SEED,
    TARGET_COL,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.utils.helpers import get_logger, load_joblib, save_joblib

logger = get_logger(__name__)


def impute_acoustic_features(
    df: pd.DataFrame,
    save_imputer_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, object]:
    """Impute missing acoustic features using IterativeImputer (Bayesian Ridge).

    This implements regression-based imputation: each feature with missing
    values is regressed on the other features in an iterative fashion.

    Args:
        df: DataFrame with acoustic feature columns.
        save_imputer_path: If provided, serialize the fitted imputer.

    Returns:
        (df with imputed values, fitted imputer)
    """
    logger.info("Imputing missing acoustic features...")

    # Check which columns have missing values
    missing = df[ACOUSTIC_FEATURE_COLS].isnull().sum()
    missing_pct = (missing / len(df)) * 100
    logger.info(f"  Missing value rates:\n{missing_pct[missing_pct > 0].to_string()}")

    if missing.sum() == 0:
        logger.info("  No missing values found — skipping imputation")
        return df, None

    # IterativeImputer uses BayesianRidge by default
    imputer = IterativeImputer(
        max_iter=10,
        random_state=RANDOM_SEED,
        verbose=0,
    )

    imputed_values = imputer.fit_transform(df[ACOUSTIC_FEATURE_COLS])
    df = df.copy()
    df[ACOUSTIC_FEATURE_COLS] = imputed_values

    if save_imputer_path:
        logger.info(f"  Saving imputer to {save_imputer_path}")
        save_joblib(imputer, save_imputer_path)

    logger.info("  Imputation complete")
    return df, imputer


def encode_labels(
    y: pd.Series,
    save_encoder_path: Optional[Path] = None,
) -> Tuple[np.ndarray, LabelEncoder]:
    """Label-encode the target genre column.

    Returns:
        (encoded labels as int array, fitted LabelEncoder)
    """
    logger.info(f"Encoding target labels: {y.nunique()} unique classes")
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(y)

    if save_encoder_path:
        save_joblib(encoder, save_encoder_path)

    logger.info(f"  Classes: {list(encoder.classes_)}")
    return encoded, encoder


def split_data(
    X: pd.DataFrame,
    y: np.ndarray,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val/test split.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    logger.info(
        f"Splitting data: {train_ratio:.0%} train / {val_ratio:.0%} val / {test_ratio:.0%} test"
    )

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_ratio,
        stratify=y,
        random_state=random_state,
    )

    # Second split: train vs val (adjust val_ratio relative to temp)
    val_ratio_adj = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio_adj,
        stratify=y_temp,
        random_state=random_state,
    )

    logger.info(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Apply SMOTE oversampling to the training set only.

    For classes with fewer than SMOTE's k_neighbors requirement,
    we use a fallback: set k_neighbors = min(samples_in_smallest_class - 1, 5).
    """
    logger.info("Applying SMOTE oversampling to training set...")

    # Determine safe k_neighbors
    class_counts = pd.Series(y_train).value_counts()
    min_count = class_counts.min()
    k_neighbors = min(5, min_count - 1) if min_count > 1 else 1

    if k_neighbors < 1:
        logger.warning("  Some classes have only 1 sample — SMOTE skipped")
        return X_train, y_train

    logger.info(f"  Using k_neighbors={k_neighbors} (min class size={min_count})")

    smote = SMOTE(
        k_neighbors=k_neighbors,
        random_state=random_state,
        n_jobs=-1,
    )
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    new_dist = pd.Series(y_resampled).value_counts()
    logger.info(f"  Before SMOTE: {len(X_train):,} samples")
    logger.info(f"  After SMOTE:  {len(X_resampled):,} samples")
    logger.info(f"  New class distribution (top 5):\n{new_dist.head().to_string()}")

    return pd.DataFrame(X_resampled, columns=X_train.columns), y_resampled


def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    numeric_cols: list[str],
    save_scaler_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Standardize numeric features: fit on train, transform all sets."""
    logger.info(f"Scaling {len(numeric_cols)} numeric features...")

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val_scaled[numeric_cols] = scaler.transform(X_val[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    if save_scaler_path:
        save_joblib(scaler, save_scaler_path)

    logger.info("  Scaling complete")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def preprocess_pipeline(
    df: pd.DataFrame,
    numeric_feature_cols: list[str],
    save_dir: Path,
    apply_smote_flag: bool = True,
) -> dict:
    """Run the full preprocessing pipeline and return processed splits.

    Args:
        df: Merged DataFrame with all features + target column.
        numeric_feature_cols: Which numeric columns to use as features.
        save_dir: Directory for serialized preprocessing artifacts.
        apply_smote_flag: Whether to apply SMOTE.

    Returns:
        Dict with keys: X_train, X_val, X_test, y_train, y_val, y_test,
                        scaler, imputer, label_encoder
    """
    logger.info("=" * 60)
    logger.info("Starting preprocessing pipeline")
    logger.info("=" * 60)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Separate features and target
    feature_cols = [c for c in numeric_feature_cols if c in df.columns and c != TARGET_COL]
    X = df[feature_cols].copy()
    y_raw = df[TARGET_COL]

    # 2. Impute missing acoustic features
    acoustic_cols_in_X = [c for c in ACOUSTIC_FEATURE_COLS if c in X.columns]
    if acoustic_cols_in_X:
        X_imputed, imputer = impute_acoustic_features(
            X, save_imputer_path=save_dir / "imputer.joblib"
        )
    else:
        X_imputed = X
        imputer = None

    # 3. Encode labels
    y, label_encoder = encode_labels(
        y_raw, save_encoder_path=save_dir / "label_encoder.joblib"
    )

    # 4. Train/val/test split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_imputed, y)

    # 5. SMOTE (training only)
    if apply_smote_flag:
        X_train, y_train = apply_smote(X_train, y_train)

    # 6. Scale numeric features
    X_train_s, X_val_s, X_test_s, scaler = scale_features(
        X_train, X_val, X_test,
        numeric_cols=feature_cols,
        save_scaler_path=save_dir / "scaler.joblib",
    )

    logger.info("Preprocessing pipeline complete")
    return {
        "X_train": X_train_s,
        "X_val": X_val_s,
        "X_test": X_test_s,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler,
        "imputer": imputer,
        "label_encoder": label_encoder,
        "feature_cols": feature_cols,
    }