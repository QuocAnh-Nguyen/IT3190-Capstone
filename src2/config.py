"""Centralised configuration for the Multi-Modal Music Genre Classification project.

All paths, constants, and hyperparameter defaults live here so downstream
modules can import a single source of truth.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (derived relative to this file)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Raw data paths
# ---------------------------------------------------------------------------
DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
SONGS_CSV: Path = DATA_RAW / "musicoset_metadata" / "songs.csv"
ARTISTS_CSV: Path = DATA_RAW / "musicoset_metadata" / "artists.csv"
LYRICS_CSV: Path = DATA_RAW / "musicoset_songfeatures" / "lyrics.csv"

# ---------------------------------------------------------------------------
# Processed / intermediate data paths
# ---------------------------------------------------------------------------
DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"
AUDIO_METADATA_CSV: Path = DATA_PROCESSED / "audio_metadata.csv"

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
AUDIO_PREVIEWS_DIR: Path = PROJECT_ROOT / "audio_previews"

# ---------------------------------------------------------------------------
# Output directories (artefacts produced during the pipeline)
# ---------------------------------------------------------------------------
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
MODELS_DIR: Path = OUTPUT_DIR / "models"
REPORTS_DIR: Path = OUTPUT_DIR / "reports"
FEATURES_DIR: Path = OUTPUT_DIR / "features"
FIGURES_DIR: Path = OUTPUT_DIR / "figures"

# ---------------------------------------------------------------------------
# Processed datasets (saved by Phase 1, consumed by later phases)
# ---------------------------------------------------------------------------
CLEANED_DATASET_CSV: Path = DATA_PROCESSED / "cleaned_dataset.csv"
LABEL_MATRIX_NPY: Path = DATA_PROCESSED / "label_matrix.npy"
LABEL_NAMES_TXT: Path = DATA_PROCESSED / "label_names.txt"
SONG_IDS_TXT: Path = DATA_PROCESSED / "song_ids.txt"
MLB_PKL: Path = DATA_PROCESSED / "multilabel_binarizer.pkl"
DATA_REPORT_TXT: Path = REPORTS_DIR / "phase1_data_report.txt"

# ---------------------------------------------------------------------------
# Random seed for reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Genre filtering thresholds
# ---------------------------------------------------------------------------
MIN_GENRE_COUNT: int = 50  # genres with fewer tracks are dropped
MAX_GENRE_COUNT: int = 1000  # cap for MLSMOTE (avoid memory blowup)

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.2
VAL_SIZE: float = 0.1  # taken from the training portion

# ---------------------------------------------------------------------------
# Audio processing (Phase 2)
# ---------------------------------------------------------------------------
AUDIO_SAMPLE_RATE: int = 22050
AUDIO_DURATION_SEC: float = 30.0  # preview clip length
N_MFCC: int = 20
N_MELS: int = 128
N_CHROMA: int = 12

# ---------------------------------------------------------------------------
# Text processing (Phase 2)
# ---------------------------------------------------------------------------
TFIDF_MAX_FEATURES: int = 5000
TFIDF_NGRAM_RANGE: tuple[int, int] = (1, 2)

# ---------------------------------------------------------------------------
# PCA (Phase 2)
# ---------------------------------------------------------------------------
PCA_VARIANCE_THRESHOLD: float = 0.95

# ---------------------------------------------------------------------------
# Model training (Phase 3)
# ---------------------------------------------------------------------------
N_CV_FOLDS: int = 5
BATCH_SIZE: int = 64
MLP_HIDDEN_DIMS: tuple[int, ...] = (512, 256, 128)
MLP_DROPOUT: float = 0.3
MLP_LR: float = 1e-3
MLP_EPOCHS: int = 100
MLP_EARLY_STOPPING_PATIENCE: int = 10


def ensure_dirs() -> None:
    """Create all output directories if they don't already exist."""
    for d in (OUTPUT_DIR, MODELS_DIR, REPORTS_DIR, FEATURES_DIR, FIGURES_DIR,
              DATA_PROCESSED):
        d.mkdir(parents=True, exist_ok=True)