"""Centralized configuration for the multi-modal music genre classification project."""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root & directory structure
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
DOCS_DIR = ROOT_DIR / "docs"

# Raw data subdirectories
METADATA_DIR = DATA_RAW_DIR / "musicoset_metadata"
POPULARITY_DIR = DATA_RAW_DIR / "musicoset_popularity"
SONGFEATURES_DIR = DATA_RAW_DIR / "musicoset_songfeatures"
ADDITIONAL_DIR = DATA_RAW_DIR / "additional"

# ---------------------------------------------------------------------------
# Raw file paths
# ---------------------------------------------------------------------------
# Metadata CSVs
ARTISTS_CSV = METADATA_DIR / "artists.csv"
SONGS_CSV = METADATA_DIR / "songs.csv"
ALBUMS_CSV = METADATA_DIR / "albums.csv"
TRACKS_CSV = METADATA_DIR / "tracks.csv"
RELEASES_CSV = METADATA_DIR / "releases.csv"

# Song features CSVs
ACOUSTIC_FEATURES_CSV = SONGFEATURES_DIR / "acoustic_features.csv"
LYRICS_CSV = SONGFEATURES_DIR / "lyrics.csv"

# Popularity CSVs
SONG_CHART_CSV = POPULARITY_DIR / "song_chart.csv"
SONG_POP_CSV = POPULARITY_DIR / "song_pop.csv"
ARTIST_CHART_CSV = POPULARITY_DIR / "artist_chart.csv"
ARTIST_POP_CSV = POPULARITY_DIR / "artist_pop.csv"
ALBUM_CHART_CSV = POPULARITY_DIR / "album_chart.csv"
ALBUM_POP_CSV = POPULARITY_DIR / "album_pop.csv"

# Additional datasets
HITS_CSV = ADDITIONAL_DIR / "hits_dataset.csv"
NONHITS_CSV = ADDITIONAL_DIR / "nonhits_dataset.csv"

# ---------------------------------------------------------------------------
# Processed artifact paths
# ---------------------------------------------------------------------------
MERGED_DATA_PARQUET = DATA_PROCESSED_DIR / "merged_data.parquet"
FEATURE_MATRIX_PARQUET = DATA_PROCESSED_DIR / "feature_matrix.parquet"
TRAIN_INDICES_PARQUET = DATA_PROCESSED_DIR / "train_indices.parquet"
VAL_INDICES_PARQUET = DATA_PROCESSED_DIR / "val_indices.parquet"
TEST_INDICES_PARQUET = DATA_PROCESSED_DIR / "test_indices.parquet"

# Serialized preprocessing artifacts
SCALER_PATH = MODELS_DIR / "scaler.joblib"
IMPUTER_PATH = MODELS_DIR / "imputer.joblib"
PCA_PATH = MODELS_DIR / "pca.joblib"
FEATURE_SELECTOR_PATH = MODELS_DIR / "feature_selector.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"

# Serialized models
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_model.joblib"
RF_MODEL_PATH = MODELS_DIR / "rf_model.joblib"
XGBOOST_MODEL_PATH = MODELS_DIR / "xgb_model.joblib"
MLP_MODEL_PATH = MODELS_DIR / "mlp_model.joblib"

# ---------------------------------------------------------------------------
# Acoustic feature columns (from Spotify)
# ---------------------------------------------------------------------------
ACOUSTIC_FEATURE_COLS = [
    "duration_ms",
    "key",
    "mode",
    "time_signature",
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "valence",
    "tempo",
]

# Non-acoustic numeric columns from song metadata
SONG_NUMERIC_COLS = [
    "popularity",
    "explicit",
    "num_artists",
]

# ---------------------------------------------------------------------------
# Target variable
# ---------------------------------------------------------------------------
TARGET_COL = "main_genre"
MIN_SAMPLES_PER_GENRE = 50  # consolidate genres below this into "Other"

# ---------------------------------------------------------------------------
# Train/Val/Test split ratios
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ---------------------------------------------------------------------------
# K-Means clustering
# ---------------------------------------------------------------------------
KMEANS_K_RANGE = range(2, 20)
KMEANS_RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# NLP / TF-IDF
# ---------------------------------------------------------------------------
TFIDF_MAX_FEATURES = 500
TFIDF_NGRAM_RANGE = (1, 2)

# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
PCA_VARIANCE_THRESHOLD = 0.95

# ---------------------------------------------------------------------------
# Random seed for reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Model hyperparameter defaults
# ---------------------------------------------------------------------------
RF_DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 2,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

XGB_DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "eval_metric": "mlogloss",
}

MLP_DEFAULT_PARAMS = {
    "hidden_layer_sizes": (256, 128, 64),
    "activation": "relu",
    "alpha": 0.001,
    "batch_size": 32,
    "learning_rate_init": 0.001,
    "max_iter": 500,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 20,
    "random_state": RANDOM_SEED,
}

# ---------------------------------------------------------------------------
# Derived constants for convenience imports
# ---------------------------------------------------------------------------
RF_N_ESTIMATORS = RF_DEFAULT_PARAMS["n_estimators"]
XGB_N_ESTIMATORS = XGB_DEFAULT_PARAMS["n_estimators"]
MLP_HIDDEN_LAYERS = MLP_DEFAULT_PARAMS["hidden_layer_sizes"]

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Ensure directories exist
# ---------------------------------------------------------------------------
for _d in (DATA_PROCESSED_DIR, MODELS_DIR, NOTEBOOKS_DIR):
    _d.mkdir(parents=True, exist_ok=True)