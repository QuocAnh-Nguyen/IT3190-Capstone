#!/usr/bin/env python3
"""Phase 2: Feature Engineering — extract all feature blocks and build
the final feature matrix ready for model training.

Usage: python scripts/run_phase2.py

Feature blocks extracted:
  1. Basic numeric features (acoustic + song metadata)
  2. Graph/network features (artist collaboration network)
  3. NLP features (lyrics: lexical, sentiment, TF-IDF)
  4. Interaction features (acoustic cross-products)
  5. K-Means cluster features
  6. Feature selection (RF importance + PCA)
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

import pandas as pd
import numpy as np

from src.config import (
    ACOUSTIC_FEATURE_COLS,
    DATA_PROCESSED_DIR,
    FEATURE_MATRIX_PARQUET,
    MODELS_DIR,
    RANDOM_SEED,
    TARGET_COL,
)
from src.features.graph_features import extract_graph_features
from src.features.nlp_features import extract_nlp_features
from src.features.interaction_features import generate_interaction_features
from src.features.cluster_features import extract_cluster_features
from src.features.feature_selection import select_features
from src.utils.helpers import get_logger, load_joblib, save_joblib

logger = get_logger("phase2")

# Ensure models dir exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load Phase 1 data
# ---------------------------------------------------------------------------
logger.info("Loading merged data from Phase 1...")
df = load_joblib(DATA_PROCESSED_DIR / "merged_data.pkl")
logger.info(f"  {len(df):,} rows, {len(df.columns)} columns")

# ---------------------------------------------------------------------------
# 1. Basic numeric features
# ---------------------------------------------------------------------------
logger.info("\n" + "=" * 60)
logger.info("1. BASIC NUMERIC FEATURES")
logger.info("=" * 60)

# Acoustic features (13 columns) + song metadata
basic_numeric_cols = [c for c in ACOUSTIC_FEATURE_COLS if c in df.columns]
extra_cols = ["popularity", "explicit", "num_artists", "artist_popularity", "followers"]
for c in extra_cols:
    if c in df.columns:
        basic_numeric_cols.append(c)

# Convert explicit to numeric if needed
if "explicit" in df.columns and df["explicit"].dtype == object:
    df["explicit"] = df["explicit"].map({"True": 1, "False": 0, True: 1, False: 0}).fillna(0)

logger.info(f"Basic numeric features: {len(basic_numeric_cols)}")
logger.info(f"  {basic_numeric_cols}")

# ---------------------------------------------------------------------------
# 2. Graph/Network Features
# ---------------------------------------------------------------------------
logger.info("\n" + "=" * 60)
logger.info("2. GRAPH/NETWORK FEATURES")
logger.info("=" * 60)

df, artist_graph, artist_metrics = extract_graph_features(
    df,
    build_graph=True,
    save_dir=MODELS_DIR,
)

graph_cols = [c for c in df.columns if c.startswith("graph_")]
logger.info(f"Graph feature columns: {graph_cols}")

# ---------------------------------------------------------------------------
# 3. NLP Features (Lyrics)
# ---------------------------------------------------------------------------
logger.info("\n" + "=" * 60)
logger.info("3. NLP FEATURES")
logger.info("=" * 60)

df, tfidf_vectorizer, svd_reducer = extract_nlp_features(
    df,
    lyrics_col="lyrics",
    fit_vectorizer=True,
    save_dir=MODELS_DIR,
)

nlp_cols = [c for c in df.columns if c.startswith("nlp_")]
logger.info(f"NLP feature columns: {len(nlp_cols)}")

# ---------------------------------------------------------------------------
# 4. Interaction Features
# ---------------------------------------------------------------------------
logger.info("\n" + "=" * 60)
logger.info("4. INTERACTION FEATURES")
logger.info("=" * 60)

# Use only the 13 acoustic features for interactions
df, interact_cols, poly_transformer = generate_interaction_features(
    df,
    numeric_cols=basic_numeric_cols[:13],  # acoustic only
    fit=True,
    save_path=MODELS_DIR / "poly_features.joblib",
)
logger.info(f"Interaction feature columns: {len(interact_cols)}")

# ---------------------------------------------------------------------------
# 5. K-Means Cluster Features
# ---------------------------------------------------------------------------
logger.info("\n" + "=" * 60)
logger.info("5. K-MEANS CLUSTER FEATURES")
logger.info("=" * 60)

df, cluster_cols, kmeans_model, cluster_scaler = extract_cluster_features(
    df,
    numeric_cols=basic_numeric_cols[:13],
    fit=True,
    save_dir=MODELS_DIR,
)
logger.info(f"Cluster feature columns: {len(cluster_cols)}")

# ---------------------------------------------------------------------------
# 6. Build feature matrix & feature selection
# ---------------------------------------------------------------------------
logger.info("\n" + "=" * 60)
logger.info("6. FEATURE SELECTION")
logger.info("=" * 60)

# Collect all feature columns
all_feature_cols = (
    basic_numeric_cols
    + graph_cols
    + nlp_cols
    + interact_cols
    + cluster_cols
)
logger.info(f"Total feature columns before selection: {len(all_feature_cols)}")

# Separate features and target
X = df[all_feature_cols].copy()
y_raw = df[TARGET_COL]

# Encode labels for feature selection
from sklearn.preprocessing import LabelEncoder
label_enc = LabelEncoder()
y = label_enc.fit_transform(y_raw)

# Run feature selection
X_selected, selected_cols, selection_metadata = select_features(
    X, y,
    nlp_feature_cols=nlp_cols,
    top_k=200,  # keep top 200 features
    pca_on_nlp=True,
    save_dir=MODELS_DIR,
)

# Persist final feature matrix with target
final_df = X_selected.copy()
final_df[TARGET_COL] = y_raw.values
final_df["genre_encoded"] = y

FEATURE_MATRIX_PATH = DATA_PROCESSED_DIR / "feature_matrix.pkl"
save_joblib(final_df, FEATURE_MATRIX_PATH)
# Save selected feature names as JSON
from src.utils.helpers import save_json
save_json(selected_cols, MODELS_DIR / "selected_features.json")

# Also save the full df (with all features) for the pipeline
FULL_FEATURE_PATH = DATA_PROCESSED_DIR / "full_features.pkl"
save_joblib(df, FULL_FEATURE_PATH)

logger.info("\n" + "=" * 60)
logger.info("PHASE 2 — FEATURE ENGINEERING COMPLETE")
logger.info("=" * 60)
logger.info(f"Final feature matrix: {final_df.shape[0]:,} rows × {final_df.shape[1]:,} cols")
logger.info(f"Saved → {FEATURE_MATRIX_PATH}")
logger.info(f"Full features → {FULL_FEATURE_PATH}")

# Print summary to stdout
print("\n" + "=" * 60)
print("PHASE 2 — FEATURE ENGINEERING COMPLETE")
print("=" * 60)
print(f"Samples:           {final_df.shape[0]:,}")
print(f"Selected features: {len(selected_cols)}")
print(f"Features by block:")
print(f"  Basic numeric:   {len(basic_numeric_cols)}")
print(f"  Graph/network:   {len(graph_cols)}")
print(f"  NLP (all):       {len(nlp_cols)}")
print(f"  Interactions:    {len(interact_cols)}")
print(f"  Clusters:        {len(cluster_cols)}")
print(f"  Total (before selection): {len(all_feature_cols)}")
print(f"  Selected:        {len(selected_cols)}")
print(f"Saved → {FEATURE_MATRIX_PATH}")
print("Done.")