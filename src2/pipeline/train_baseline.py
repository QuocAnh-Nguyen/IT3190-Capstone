"""Phase 3.4 — Improved Training Pipeline (Multi-Class Single-Label).

Post-improve_plan rewrite.  This pipeline:
1. Loads & cleans data
2. Encodes labels with "stage & screen" excluded
3. Converts multi-label → single-label ("rarest first" strategy)
4. Extracts audio features (traditional),
5. Extracts text features with reduced TF-IDF + TruncatedSVD,
6. Fuses features, applies PCA,
7. Optionally applies SMOTE to minority classes,
8. Trains 4–5 multi-class traditional ML models,
9. Generates evaluation metrics and comparison reports.

Run as::
    python -m src2.pipeline.train_baseline
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src2 import config
from src2.data import data_cleaner, data_loader, label_encoder
from src2.data.label_converter import convert_to_single_label, compute_label_counts, consolidate_genres
from src2.features import audio_traditional, feature_reducer, text_traditional
from src2.models import traditional_models
from src2.models.evaluation_multiclass import (
    compute_metrics_multiclass,
    compare_models_multiclass,
    save_metrics_multiclass,
    plot_confusion_matrix,
    plot_per_class_f1,
)
from src2.utils.io_utils import load_pickle, save_pickle, save_npy
from src2.utils.logging_utils import log_section, setup_logger

logger = logging.getLogger("music_genre")

# ---------------------------------------------------------------------------
# Cached artefact paths (post-improve_plan)
# ---------------------------------------------------------------------------
_AUDIO_FEAT_PKL: Path = config.FEATURES_DIR / "audio_features_traditional_v2.pkl"
_TEXT_FEAT_PKL: Path = config.FEATURES_DIR / "text_features_traditional_v2.pkl"
_CLEANED_WITH_LABEL_CSV: Path = config.DATA_PROCESSED / "cleaned_single_label.csv"


def _elapsed(start: float) -> str:
    secs = int(time.time() - start)
    mins, secs = divmod(secs, 60)
    return f"{mins}m {secs}s" if mins else f"{secs}s"


def _align_ids(
    ids_a: list[str], arr_a: np.ndarray,
    ids_b: list[str], arr_b: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Inner-join two feature arrays on shared song IDs."""
    set_b = {sid: i for i, sid in enumerate(ids_b)}
    shared_ids: list[str] = []
    rows_a: list[int] = []
    rows_b: list[int] = []
    for i, sid in enumerate(ids_a):
        if sid in set_b:
            shared_ids.append(sid)
            rows_a.append(i)
            rows_b.append(set_b[sid])
    logger.info(
        "ID alignment: %d ∩ %d = %d shared",
        len(ids_a), len(ids_b), len(shared_ids),
    )
    return shared_ids, arr_a[rows_a], arr_b[rows_b]


# ---------------------------------------------------------------------------
# Phase 1: Data loading + cleaning
# ---------------------------------------------------------------------------

def phase1_load_and_clean() -> pd.DataFrame:
    """Load raw data, merge, clean — with disk cache."""
    log_section(logger, "Phase 1 — Data Loading & Cleaning")
    t0 = time.time()

    cache_path = config.CLEANED_DATASET_CSV
    if cache_path.exists():
        logger.info("Cache hit — loading from %s", cache_path)
        df = pd.read_csv(cache_path, dtype={"song_id": str})
        logger.info("Loaded %d rows x %d cols [%s]", *df.shape, _elapsed(t0))
        return df

    logger.info("Cache miss — running load_and_merge() ...")
    raw_df = data_loader.load_and_merge()
    cleaned, avail_report, genre_freq = data_cleaner.clean_dataset(raw_df)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(cache_path, index=False)
    logger.info("Saved cleaned dataset -> %s", cache_path)
    logger.info("Phase 1 complete: %d rows x %d cols [%s]", *cleaned.shape, _elapsed(t0))
    return cleaned


# ---------------------------------------------------------------------------
# Phase 1.4: Label encoding (with exclusion) + single-label conversion
# ---------------------------------------------------------------------------

def phase1_4_label_encoding(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Encode labels with exclusion, then convert to single-label.

    Returns
    -------
    df_with_label : pd.DataFrame
        The input df with a ``primary_genre`` column added.
    label_names : list[str]
        Sorted list of unique single-label genres.
    """
    log_section(logger, "Phase 1.4 — Label Encoding & Single-Label Conversion")
    t0 = time.time()

    excluded = config.EXCLUDED_GENRES
    strategy = config.SINGLE_LABEL_STRATEGY

    logger.info("Excluding genres: %s", excluded)
    logger.info("Conversion strategy: %s", strategy)

    # Compute multi-label counts before exclusion (for "rarest" strategy)
    label_counts = compute_label_counts(df)

    logger.info("Genre counts before exclusion:")
    for genre, cnt in label_counts.head(10).items():
        logger.info("  %-30s  %6d", genre, cnt)

    # Convert to single-label (this also handles exclusion)
    df_single = convert_to_single_label(
        df,
        label_counts=label_counts,
        strategy=strategy,
        exclude_genres=excluded,
        min_count=config.MIN_GENRE_COUNT,
    )

    label_names = sorted(df_single["primary_genre"].unique())
    logger.info("After exclusion + conversion: %d classes: %s",
                len(label_names), label_names)

    # ------------------------------------------------------------------
    # Genre consolidation (Step 3C of improve_plan)
    # Merge tail classes: blues+jazz → "Blues & Jazz",
    #                     latin+reggae → "Latin & Caribbean"
    # ------------------------------------------------------------------
    consolidation = config.GENRE_CONSOLIDATION
    if consolidation:
        log_section(logger, "Genre Consolidation (Step 3C)")
        df_single = consolidate_genres(df_single, consolidation)
        label_names = sorted(df_single["primary_genre"].unique())
        logger.info("After consolidation: %d classes: %s",
                    len(label_names), label_names)

    # Save processed dataset with primary_genre
    _CLEANED_WITH_LABEL_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_single.to_csv(_CLEANED_WITH_LABEL_CSV, index=False)
    logger.info("Saved cleaned single-label dataset -> %s", _CLEANED_WITH_LABEL_CSV)

    logger.info(
        "Phase 1.4 complete: %d tracks, %d classes [%s]",
        len(df_single), len(label_names), _elapsed(t0),
    )
    return df_single, label_names


# ---------------------------------------------------------------------------
# Phase 2.1: Audio features
# ---------------------------------------------------------------------------

def phase2_1_audio_features(
    df: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Extract traditional audio features with cache."""
    log_section(logger, "Phase 2.1 — Audio Feature Extraction (Traditional)")
    t0 = time.time()

    if _AUDIO_FEAT_PKL.exists():
        logger.info("Cache hit — loading from %s", _AUDIO_FEAT_PKL)
        payload = load_pickle(_AUDIO_FEAT_PKL)
        logger.info("Loaded audio features: %s [%s]",
                     payload["features"].shape, _elapsed(t0))
        return payload["features"], payload["song_ids"]

    audio_df = df[df["has_audio"].astype(bool)][["song_id", "mp3_path"]].copy()
    song_ids_for_audio: list[str] = audio_df["song_id"].tolist()
    mp3_paths_for_audio: list[str] = audio_df["mp3_path"].tolist()

    logger.info("Extracting audio features for %d songs ...", len(song_ids_for_audio))
    extracted_df = audio_traditional.extract_all_features(
        song_ids=song_ids_for_audio,
        mp3_paths=mp3_paths_for_audio,
    )

    audio_arr = extracted_df.values
    audio_ids = extracted_df.index.astype(str).tolist()

    _AUDIO_FEAT_PKL.parent.mkdir(parents=True, exist_ok=True)
    save_pickle({"features": audio_arr, "song_ids": audio_ids}, _AUDIO_FEAT_PKL)
    logger.info("Saved audio features -> %s", _AUDIO_FEAT_PKL)

    logger.info("Phase 2.1 complete: %s, %d songs [%s]",
                 audio_arr.shape, len(audio_ids), _elapsed(t0))
    return audio_arr, audio_ids


# ---------------------------------------------------------------------------
# Phase 2.2: Text features
# ---------------------------------------------------------------------------

def phase2_2_text_features(
    df: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Extract text features (TF-IDF + SVD + lexical) with cache."""
    log_section(logger, "Phase 2.2 — Text Feature Extraction (Traditional, improved)")
    t0 = time.time()

    if _TEXT_FEAT_PKL.exists():
        logger.info("Cache hit — loading from %s", _TEXT_FEAT_PKL)
        payload = load_pickle(_TEXT_FEAT_PKL)
        logger.info("Loaded text features: %s [%s]",
                     payload["features"].shape, _elapsed(t0))
        return payload["features"], payload["song_ids"]

    logger.info("Extracting text features for %d songs ...", len(df))
    text_arr, text_ids = text_traditional.extract_text_features(df)

    _TEXT_FEAT_PKL.parent.mkdir(parents=True, exist_ok=True)
    save_pickle({"features": text_arr, "song_ids": text_ids}, _TEXT_FEAT_PKL)
    logger.info("Saved text features -> %s", _TEXT_FEAT_PKL)

    logger.info("Phase 2.2 complete: %s, %d songs [%s]",
                 text_arr.shape, len(text_ids), _elapsed(t0))
    return text_arr, text_ids


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    run_start = time.time()
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Logging
    log_file = config.REPORTS_DIR / f"train_baseline_v2_{run_ts}.log"
    setup_logger("music_genre", log_file=log_file)

    log_section(logger, f"Improved Multi-Class Pipeline [{run_ts}]")
    logger.info("Log file -> %s", log_file)
    logger.info("Project  -> %s", config.PROJECT_ROOT)

    config.ensure_dirs()

    # ------------------------------------------------------------------
    # Phase 1: Load data
    # ------------------------------------------------------------------
    try:
        df = phase1_load_and_clean()
    except Exception as exc:
        logger.critical("Phase 1 FAILED: %s — aborting.", exc, exc_info=True)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 1.4: Label encoding + single-label conversion
    # ------------------------------------------------------------------
    try:
        df, label_names = phase1_4_label_encoding(df)
    except Exception as exc:
        logger.critical("Phase 1.4 FAILED: %s — aborting.", exc, exc_info=True)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 2.1: Audio features
    # ------------------------------------------------------------------
    try:
        audio_arr, audio_ids = phase2_1_audio_features(df)
    except Exception as exc:
        logger.critical("Phase 2.1 FAILED: %s — aborting.", exc, exc_info=True)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 2.2: Text features
    # ------------------------------------------------------------------
    try:
        text_arr, text_ids = phase2_2_text_features(df)
    except Exception as exc:
        logger.critical("Phase 2.2 FAILED: %s — aborting.", exc, exc_info=True)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Feature fusion (inner join on song_id)
    # ------------------------------------------------------------------
    log_section(logger, "Feature Fusion — Audio + Text")
    shared_ids, aligned_audio, aligned_text = _align_ids(
        audio_ids, audio_arr, text_ids, text_arr,
    )

    if len(shared_ids) == 0:
        logger.critical("No shared IDs between audio and text — aborting.")
        sys.exit(1)

    fused_X_raw = feature_reducer.fuse_features(aligned_audio, aligned_text)
    logger.info("Fused (raw) feature matrix: %s", fused_X_raw.shape)

    # ------------------------------------------------------------------
    # Align labels to fused feature IDs
    # ------------------------------------------------------------------
    df_indexed = df.set_index("song_id")
    y_all = np.array([df_indexed.loc[sid, "primary_genre"]
                      if sid in df_indexed.index else None
                      for sid in shared_ids])

    # Drop rows where label is missing
    valid_mask = np.array([y is not None for y in y_all])
    missing_labels = (~valid_mask).sum()
    if missing_labels > 0:
        logger.warning("%d songs have features but no primary_genre — dropping.",
                       missing_labels)
    fused_X_raw = fused_X_raw[valid_mask]
    y_all = y_all[valid_mask]
    shared_ids_valid = [sid for sid, m in zip(shared_ids, valid_mask) if m]

    logger.info("After label alignment: X=%s, y=%d", fused_X_raw.shape, len(y_all))

    # ------------------------------------------------------------------
    # Train / test split (BEFORE PCA — prevents data leakage)
    # ------------------------------------------------------------------
    log_section(logger, "Train / Test Split")
    indices = np.arange(len(fused_X_raw))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y_all,
    )

    X_train_raw = fused_X_raw[train_idx]
    X_test_raw = fused_X_raw[test_idx]
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]

    logger.info(
        "Split: train=%d, test=%d  |  test_size=%.2f  stratify=True",
        len(X_train_raw), len(X_test_raw), config.TEST_SIZE,
    )

    # ------------------------------------------------------------------
    # PCA: fit on train, transform test
    # ------------------------------------------------------------------
    log_section(logger, "PCA Dimensionality Reduction")
    t0 = time.time()

    pca_model, X_train_pca = feature_reducer.fit_pca(
        X_train_raw,
        variance_threshold=config.PCA_VARIANCE_THRESHOLD,
        n_components=config.PCA_N_COMPONENTS,
    )
    X_test_pca = pca_model.transform(X_test_raw)

    logger.info(
        "PCA complete: train=%s, test=%s [%s]",
        X_train_pca.shape, X_test_pca.shape, _elapsed(t0),
    )

    # Save fitted PCA
    feature_reducer.save_reducer(
        pca_model, config.FEATURES_DIR / "pca_reducer_v2.joblib",
    )

    # ------------------------------------------------------------------
    # Train traditional models
    # ------------------------------------------------------------------
    log_section(logger, "Training Traditional Multi-Class Models")
    t0 = time.time()

    all_results: dict[str, dict[str, Any]] = {}

    try:
        trad_results = traditional_models.run_all_traditional_models(
            X_train=X_train_pca,
            y_train=y_train,
            X_test=X_test_pca,
            y_test=y_test,
            label_names=label_names,
            output_dir=config.MODELS_DIR,
            apply_smote_flag=config.SMOTE_ENABLED,
            smote_k_neighbors=config.SMOTE_K_NEIGHBORS,
            smote_target_min=config.SMOTE_TARGET_MIN_SAMPLES,
        )
        all_results.update(trad_results)
    except Exception as exc:
        logger.error("Traditional models training FAILED: %s", exc, exc_info=True)

    logger.info("Model training complete [%s]", _elapsed(t0))

    # ------------------------------------------------------------------
    # Save per-model metrics and visualisations
    # ------------------------------------------------------------------
    log_section(logger, "Saving Per-Model Metrics & Visualisations")

    for model_name, metrics in all_results.items():
        if not metrics.get("summary"):
            logger.warning("No metrics for '%s' — skipping.", model_name)
            continue

        save_metrics_multiclass(metrics, model_name, output_dir=config.REPORTS_DIR)

        # Per-class F1 chart
        per_class_df = metrics.get("per_class")
        if per_class_df is not None:
            plot_per_class_f1(
                per_class_df,
                output_path=config.FIGURES_DIR / f"{model_name}_per_class_f1.png",
                title=f"Per-Class F1 — {model_name}",
            )

        # Confusion matrix
        cm = metrics.get("confusion_matrix")
        if cm is not None:
            plot_confusion_matrix(
                cm, label_names,
                output_path=config.FIGURES_DIR / f"{model_name}_confusion_matrix.png",
                title=f"Confusion Matrix — {model_name}",
            )

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------
    log_section(logger, "Model Comparison")

    if all_results:
        comparison_df = compare_models_multiclass(all_results, output_dir=config.REPORTS_DIR)
        logger.info("Final comparison:\n%s", comparison_df.to_string())

        # Log the key result
        best_model = comparison_df.index[0]
        best_f1 = comparison_df.loc[best_model, "macro_f1"]
        logger.info("")
        logger.info("=" * 60)
        logger.info("  BEST MODEL: %s  |  F1-Macro: %.4f  |  Accuracy: %.4f",
                     best_model, best_f1,
                     comparison_df.loc[best_model, "accuracy"])
        logger.info("=" * 60)

        if best_f1 >= 0.60:
            logger.info("✓ TARGET ACHIEVED: F1-macro ≥ 60%%!")
        else:
            logger.info("✗ Below 60%% target — consider genre consolidation (Step 3C).")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    log_section(logger, f"Pipeline Finished [total: {_elapsed(run_start)}]")
    logger.info("Models trained: %d", len(all_results))
    logger.info("Reports dir: %s", config.REPORTS_DIR)
    logger.info("Figures dir: %s", config.FIGURES_DIR)


if __name__ == "__main__":
    main()