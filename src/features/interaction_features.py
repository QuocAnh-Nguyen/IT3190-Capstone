"""Interaction feature generation via PolynomialFeatures.

Creates pairwise cross-product features from the acoustic feature set
(e.g., energy × danceability, acousticness × valence) to help linear
models capture non-linear decision boundaries between genres.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from src.config import ACOUSTIC_FEATURE_COLS, RANDOM_SEED
from src.utils.helpers import get_logger, save_joblib

logger = get_logger(__name__)


def generate_interaction_features(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    degree: int = 2,
    interaction_only: bool = True,
    include_bias: bool = False,
    fit: bool = True,
    poly_transformer: Optional[PolynomialFeatures] = None,
    save_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, List[str], Optional[PolynomialFeatures]]:
    """Generate pairwise interaction features (cross-products only).

    Args:
        df: Input DataFrame with numeric columns.
        numeric_cols: Which columns to generate interactions from.
            Defaults to ACOUSTIC_FEATURE_COLS.
        degree: Polynomial degree (2 = pairwise only).
        interaction_only: If True, only cross-terms (no squares).
        include_bias: If True, include a bias (constant) column.
        fit: If True, fit a new PolynomialFeatures transformer.
        poly_transformer: If provided, use this pre-fitted transformer.
        save_path: If provided and fitting, save the transformer.

    Returns:
        (df with interaction columns appended, list of interaction column names,
         fitted PolynomialFeatures or None)
    """
    logger.info("Generating interaction features...")

    if numeric_cols is None:
        numeric_cols = [c for c in ACOUSTIC_FEATURE_COLS if c in df.columns]

    logger.info(f"  Input features: {len(numeric_cols)}")

    X = df[numeric_cols].fillna(0).values

    if fit:
        poly = PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias,
        )
        X_poly = poly.fit_transform(X)
        if save_path:
            save_joblib(poly, save_path)
            logger.info(f"  Saved PolynomialFeatures → {save_path}")
    elif poly_transformer is not None:
        poly = poly_transformer
        X_poly = poly.transform(X)
    else:
        raise ValueError("Either fit=True or poly_transformer must be provided")

    # Generate column names
    feature_names = poly.get_feature_names_out(numeric_cols)

    # Skip the bias column and the original features (columns 0 to len(numeric_cols))
    # We only want the new interaction terms
    interaction_names = list(feature_names[len(numeric_cols) + (1 if include_bias else 0):])
    interaction_values = X_poly[:, len(numeric_cols) + (1 if include_bias else 0):]

    # Create interaction DataFrame
    interaction_df = pd.DataFrame(
        interaction_values,
        index=df.index,
        columns=[f"interact_{name.replace(' ', '_')}" for name in interaction_names],
    )

    logger.info(f"  Generated {len(interaction_df.columns)} interaction features")

    # Concatenate
    df_out = pd.concat([df, interaction_df], axis=1)

    return df_out, list(interaction_df.columns), poly if fit else None