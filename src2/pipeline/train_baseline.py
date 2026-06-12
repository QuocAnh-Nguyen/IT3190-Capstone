"""Phase 3.4 — Training Pipeline Orchestration (EXECUTE).

This is the **main entry-point** for the full traditional-feature baseline
pipeline.  Run it as::

    python -m src2.pipeline.train_baseline

Pipeline stages executed in order
----------------------------------
1.   Data loading      – load + merge raw CSVs, clean, cache to disk.
1.4  Label encoding    – multi-label binarisation, cache artefacts.
2.1  Audio features    – traditional hand-crafted audio descriptors.
2.2  Text features     – TF-IDF / NLP features from lyrics.
2.5  Feature fusion    – inner-join alignment, concatenation, PCA reduction.
     Train/test split  – stratification-free split with a fixed random seed.
3.2  Traditional models – SVM, RF, LR, XGBoost, etc. (one-vs-rest wrappers).
3.3  MLP model         – PyTorch multi-label MLP with early stopping.
     Results comparison – unified evaluation table written to REPORTS_DIR.

Every stage is cache-aware: if the expected output artefact already exists on
disk the stage is skipped and the cached version is loaded, making reruns fast
during experimentation.  Each stage is also individually wrapped in a
try/except block so that a failure in one stage does not abort the whole run
(a warning is logged and subsequent stages that can proceed will do so).

Milestone: Phase 3 — Baseline System Training.
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src2 import config
from src2.data import data_cleaner, data_loader, label_encoder
from src2.features import audio_traditional, feature_reducer, text_traditional
from src2.models import evaluation, neural_models, traditional_models
from src2.utils.io_utils import (
    load_npy,
    load_pickle,
    save_npy,
    save_pickle,
)
from src2.utils.logging_utils import log_section, setup_logger

# ---------------------------------------------------------------------------
# Module-level logger (handlers are attached in main())
# ---------------------------------------------------------------------------
logger = logging.getLogger("music_genre")

# ---------------------------------------------------------------------------
# Artefact paths that are not already declared in config
# ---------------------------------------------------------------------------
_AUDIO_FEAT_PKL: Path = config.FEATURES_DIR / "audio_features_traditional.pkl"
_TEXT_FEAT_PKL: Path = config.FEATURES_DIR / "text_features_traditional.pkl"
_FUSED_FEAT_NPY: Path = config.FEATURES_DIR / "fused_features_pca.npy"
_FUSED_IDS_PKL: Path = config.FEATURES_DIR / "fused_song_ids.pkl"
_RESULTS_JSON: Path = config.REPORTS_DIR / "baseline_results_comparison.json"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _elapsed(start: float) -> str:
    """Return a human-readable elapsed-time string since *start*.

    Parameters
    ----------
    start : float
        Value returned by ``time.time()`` at phase start.

    Returns
    -------
    str
        Formatted string such as ``"2m 14s"``.
    """
    secs = int(time.time() - start)
    mins, secs = divmod(secs, 60)
    return f"{mins}m {secs}s" if mins else f"{secs}s"


def _align_ids(
    ids_a: list[str],
    arr_a: np.ndarray,
    ids_b: list[str],
    arr_b: np.ndarray,
) -> Tuple[list[str], np.ndarray, np.ndarray]:
    """Inner-join two feature arrays on their shared song IDs.

    Parameters
    ----------
    ids_a : list[str]
        Ordered song IDs corresponding to rows in *arr_a*.
    arr_a : np.ndarray
        Feature matrix of shape ``(n_a, d_a)``.
    ids_b : list[str]
        Ordered song IDs corresponding to rows in *arr_b*.
    arr_b : np.ndarray
        Feature matrix of shape ``(n_b, d_b)``.

    Returns
    -------
    shared_ids : list[str]
        Intersection of IDs, preserving the order from *ids_a*.
    aligned_a : np.ndarray
        Rows of *arr_a* whose ID is in both sets.
    aligned_b : np.ndarray
        Corresponding rows of *arr_b* in the same order.
    """
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
        "ID alignment: %d intersect %d = %d shared songs",
        len(ids_a),
        len(ids_b),
        len(shared_ids),
    )
    return shared_ids, arr_a[rows_a], arr_b[rows_b]


def _align_labels(
    label_song_ids: list[str],
    Y: np.ndarray,
    target_ids: list[str],
) -> np.ndarray:
    """Return the rows of *Y* that correspond to *target_ids*.

    Parameters
    ----------
    label_song_ids : list[str]
        The full ordered list of IDs that *Y* was built from.
    Y : np.ndarray
        Label matrix of shape ``(n_songs, n_classes)``.
    target_ids : list[str]
        The ordered subset of IDs we want rows for.

    Returns
    -------
    np.ndarray
        Sub-matrix of *Y* of shape ``(len(target_ids), n_classes)``.
    """
    id_to_row = {sid: i for i, sid in enumerate(label_song_ids)}
    indices = [id_to_row[sid] for sid in target_ids if sid in id_to_row]
    missing = [sid for sid in target_ids if sid not in id_to_row]
    if missing:
        logger.warning(
            "%d song IDs in feature set have no label row — they will be dropped.",
            len(missing),
        )
    return Y[indices]


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------


def phase1_data_loading() -> pd.DataFrame:
    """Phase 1: Load and clean the dataset, with disk-cache support.

    Returns
    -------
    pd.DataFrame
        The cleaned dataset (columns include song_id, genres, lyrics,
        mp3_path, has_lyrics, has_audio, …).

    Raises
    ------
    RuntimeError
        If loading fails and no cached version is available.
    """
    log_section(logger, "Phase 1 — Data Loading & Cleaning")
    t0 = time.time()

    cache_path: Path = config.DATA_PROCESSED / "cleaned_dataset.csv"

    if cache_path.exists():
        logger.info("Cache hit — loading cleaned_dataset.csv from %s", cache_path)
        try:
            df = pd.read_csv(cache_path, dtype={"song_id": str})
            logger.info(
                "Loaded cached dataset: %d rows x %d cols  [%s]",
                *df.shape,
                _elapsed(t0),
            )
            return df
        except Exception as exc:
            logger.warning(
                "Failed to load cache (%s) — regenerating from raw sources.", exc
            )

    # Cache miss — run full load + clean
    logger.info("Cache miss — running load_and_merge() ...")
    raw_df = data_loader.load_and_merge()

    logger.info("Running clean_dataset() ...")
    cleaned, avail_report, genre_freq = data_cleaner.clean_dataset(raw_df)

    # Persist cleaned dataset
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(cache_path, index=False)
        logger.info("Saved cleaned dataset -> %s", cache_path)
    except Exception as exc:
        logger.warning("Could not save cleaned dataset cache: %s", exc)

    logger.info(
        "Phase 1 complete: %d rows x %d cols  [%s]",
        *cleaned.shape,
        _elapsed(t0),
    )
    return cleaned


def phase1_4_label_encoding(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, list[str], list[str]]:
    """Phase 1.4: Encode multi-label genre targets, with cache support.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset containing ``song_id`` and ``genres`` columns.

    Returns
    -------
    Y : np.ndarray
        Binary label matrix of shape ``(n_songs, n_classes)``.
    label_names : list[str]
        Ordered list of genre class names.
    song_ids : list[str]
        Ordered list of song IDs corresponding to rows of *Y*.
    """
    log_section(logger, "Phase 1.4 — Label Encoding")
    t0 = time.time()

    labels_exist = (
        config.LABEL_MATRIX_NPY.exists()
        and config.MLB_PKL.exists()
        and config.LABEL_NAMES_TXT.exists()
        and config.SONG_IDS_TXT.exists()
    )

    if labels_exist:
        logger.info(
            "Cache hit — loading label artefacts from %s", config.DATA_PROCESSED
        )
        try:
            Y = load_npy(config.LABEL_MATRIX_NPY)
            with open(config.LABEL_NAMES_TXT, "r", encoding="utf-8") as fh:
                label_names = [ln.strip() for ln in fh if ln.strip()]
            with open(config.SONG_IDS_TXT, "r", encoding="utf-8") as fh:
                song_ids = [ln.strip() for ln in fh if ln.strip()]
            logger.info(
                "Loaded labels: Y=%s, %d classes, %d songs  [%s]",
                Y.shape,
                len(label_names),
                len(song_ids),
                _elapsed(t0),
            )
            return Y, label_names, song_ids
        except Exception as exc:
            logger.warning(
                "Failed to load cached labels (%s) — re-encoding ...", exc
            )

    # Cache miss — encode
    logger.info("Cache miss — running label_encoder.encode_labels() ...")
    Y, label_names, song_ids, mlb = label_encoder.encode_labels(df)

    # Persist artefacts
    try:
        config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        save_npy(Y, config.LABEL_MATRIX_NPY)
        save_pickle(mlb, config.MLB_PKL)
        with open(config.LABEL_NAMES_TXT, "w", encoding="utf-8") as fh:
            fh.write("\n".join(label_names))
        with open(config.SONG_IDS_TXT, "w", encoding="utf-8") as fh:
            fh.write("\n".join(song_ids))
        logger.info("Saved label artefacts -> %s", config.DATA_PROCESSED)
    except Exception as exc:
        logger.warning("Could not save label artefacts: %s", exc)

    logger.info(
        "Phase 1.4 complete: Y=%s, %d classes  [%s]",
        Y.shape,
        len(label_names),
        _elapsed(t0),
    )
    return Y, label_names, song_ids


def phase2_1_audio_features(
    song_ids: list[str],
    mp3_paths: list[str],
) -> Tuple[np.ndarray, list[str]]:
    """Phase 2.1: Extract traditional audio features, with cache support.

    Parameters
    ----------
    song_ids : list[str]
        Ordered list of song IDs matching ``mp3_paths``.
    mp3_paths : list[str]
        Ordered list of MP3 file paths (relative or absolute).

    Returns
    -------
    audio_arr : np.ndarray
        Feature matrix of shape ``(n_songs, n_audio_features)``.
    audio_ids : list[str]
        Ordered song IDs corresponding to rows of *audio_arr*.
    """
    log_section(logger, "Phase 2.1 — Audio Feature Extraction (Traditional)")
    t0 = time.time()

    if _AUDIO_FEAT_PKL.exists():
        logger.info("Cache hit — loading audio features from %s", _AUDIO_FEAT_PKL)
        try:
            payload = load_pickle(_AUDIO_FEAT_PKL)
            audio_arr: np.ndarray = payload["features"]
            audio_ids: list[str] = payload["song_ids"]
            logger.info(
                "Loaded audio features: %s, %d songs  [%s]",
                audio_arr.shape,
                len(audio_ids),
                _elapsed(t0),
            )
            return audio_arr, audio_ids
        except Exception as exc:
            logger.warning(
                "Failed to load cached audio features (%s) — re-extracting ...", exc
            )

    logger.info(
        "Cache miss — extracting audio features for %d songs ...", len(song_ids)
    )
    audio_arr, audio_ids = audio_traditional.extract_all_features(
        song_ids=song_ids,
        mp3_paths=mp3_paths,
    )

    try:
        _AUDIO_FEAT_PKL.parent.mkdir(parents=True, exist_ok=True)
        save_pickle({"features": audio_arr, "song_ids": audio_ids}, _AUDIO_FEAT_PKL)
        logger.info("Saved audio features -> %s", _AUDIO_FEAT_PKL)
    except Exception as exc:
        logger.warning("Could not save audio features: %s", exc)

    logger.info(
        "Phase 2.1 complete: %s, %d songs  [%s]",
        audio_arr.shape,
        len(audio_ids),
        _elapsed(t0),
    )
    return audio_arr, audio_ids


def phase2_2_text_features(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, list[str]]:
    """Phase 2.2: Extract traditional text (TF-IDF) features, with cache support.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset containing ``song_id`` and ``lyrics`` columns.

    Returns
    -------
    text_arr : np.ndarray
        Feature matrix of shape ``(n_songs, n_text_features)``.
    text_ids : list[str]
        Ordered song IDs corresponding to rows of *text_arr*.
    """
    log_section(logger, "Phase 2.2 — Text Feature Extraction (Traditional)")
    t0 = time.time()

    if _TEXT_FEAT_PKL.exists():
        logger.info("Cache hit — loading text features from %s", _TEXT_FEAT_PKL)
        try:
            payload = load_pickle(_TEXT_FEAT_PKL)
            text_arr: np.ndarray = payload["features"]
            text_ids: list[str] = payload["song_ids"]
            logger.info(
                "Loaded text features: %s, %d songs  [%s]",
                text_arr.shape,
                len(text_ids),
                _elapsed(t0),
            )
            return text_arr, text_ids
        except Exception as exc:
            logger.warning(
                "Failed to load cached text features (%s) — re-extracting ...", exc
            )

    logger.info(
        "Cache miss — extracting text features for %d songs ...", len(df)
    )
    text_arr, text_ids = text_traditional.extract_text_features(df)

    try:
        _TEXT_FEAT_PKL.parent.mkdir(parents=True, exist_ok=True)
        save_pickle({"features": text_arr, "song_ids": text_ids}, _TEXT_FEAT_PKL)
        logger.info("Saved text features -> %s", _TEXT_FEAT_PKL)
    except Exception as exc:
        logger.warning("Could not save text features: %s", exc)

    logger.info(
        "Phase 2.2 complete: %s, %d songs  [%s]",
        text_arr.shape,
        len(text_ids),
        _elapsed(t0),
    )
    return text_arr, text_ids


def phase2_5_feature_fusion(
    audio_arr: np.ndarray,
    audio_ids: list[str],
    text_arr: np.ndarray,
    text_ids: list[str],
) -> Tuple[np.ndarray, list[str]]:
    """Phase 2.5: Align, fuse, and PCA-reduce audio + text features.

    Performs an inner-join on song IDs, then concatenates both feature
    matrices and applies PCA with variance threshold
    ``config.PCA_VARIANCE_THRESHOLD``.

    Parameters
    ----------
    audio_arr : np.ndarray
        Audio feature matrix ``(n_audio, d_audio)``.
    audio_ids : list[str]
        Song IDs for rows of *audio_arr*.
    text_arr : np.ndarray
        Text feature matrix ``(n_text, d_text)``.
    text_ids : list[str]
        Song IDs for rows of *text_arr*.

    Returns
    -------
    X_pca : np.ndarray
        PCA-reduced fused feature matrix.
    shared_ids : list[str]
        Song IDs (inner join) corresponding to rows of *X_pca*.
    """
    log_section(logger, "Phase 2.5 — Feature Alignment, Fusion & PCA")
    t0 = time.time()

    if _FUSED_FEAT_NPY.exists() and _FUSED_IDS_PKL.exists():
        logger.info(
            "Cache hit — loading fused PCA features from %s", _FUSED_FEAT_NPY
        )
        try:
            X_pca = load_npy(_FUSED_FEAT_NPY)
            shared_ids: list[str] = load_pickle(_FUSED_IDS_PKL)
            logger.info(
                "Loaded fused features: %s, %d songs  [%s]",
                X_pca.shape,
                len(shared_ids),
                _elapsed(t0),
            )
            return X_pca, shared_ids
        except Exception as exc:
            logger.warning(
                "Failed to load cached fused features (%s) — re-computing ...", exc
            )

    # Align on shared song IDs
    shared_ids, aligned_audio, aligned_text = _align_ids(
        audio_ids, audio_arr, text_ids, text_arr
    )

    if len(shared_ids) == 0:
        raise RuntimeError(
            "No shared song IDs between audio and text feature sets — "
            "cannot proceed with feature fusion."
        )

    logger.info(
        "Fusing features: audio=%s + text=%s -> concatenated dim=%d",
        aligned_audio.shape,
        aligned_text.shape,
        aligned_audio.shape[1] + aligned_text.shape[1],
    )

    # Concatenate via feature_reducer
    fused_X = feature_reducer.fuse_features(aligned_audio, aligned_text)
    logger.info("Fused feature matrix: %s", fused_X.shape)

    # PCA reduction
    logger.info(
        "Fitting PCA (variance threshold=%.2f) ...", config.PCA_VARIANCE_THRESHOLD
    )
    X_pca = feature_reducer.fit_pca(fused_X)
    logger.info("PCA-reduced feature matrix: %s", X_pca.shape)

    # Persist
    try:
        _FUSED_FEAT_NPY.parent.mkdir(parents=True, exist_ok=True)
        save_npy(X_pca, _FUSED_FEAT_NPY)
        save_pickle(shared_ids, _FUSED_IDS_PKL)
        logger.info(
            "Saved fused PCA features -> %s  |  IDs -> %s",
            _FUSED_FEAT_NPY,
            _FUSED_IDS_PKL,
        )
    except Exception as exc:
        logger.warning("Could not save fused features: %s", exc)

    logger.info(
        "Phase 2.5 complete: X_pca=%s, %d songs  [%s]",
        X_pca.shape,
        len(shared_ids),
        _elapsed(t0),
    )
    return X_pca, shared_ids


def phase_train_test_split(
    X: np.ndarray,
    Y: np.ndarray,
    label_song_ids: list[str],
    feature_song_ids: list[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align labels to feature IDs then perform a stratification-free split.

    Multi-label stratification is avoided due to the complexity of finding a
    perfect stratifier for high-cardinality multi-label targets.  A random
    split with a fixed seed ensures full reproducibility.

    Parameters
    ----------
    X : np.ndarray
        Full feature matrix, rows aligned to *feature_song_ids*.
    Y : np.ndarray
        Full label matrix, rows aligned to *label_song_ids*.
    label_song_ids : list[str]
        Song IDs in the order rows of *Y* were built.
    feature_song_ids : list[str]
        Song IDs in the order rows of *X* were built.

    Returns
    -------
    X_train, X_test, Y_train, Y_test : np.ndarray
        Train and test splits.
    """
    log_section(logger, "Train / Test Split")

    # Align label rows to feature ID order
    Y_aligned = _align_labels(label_song_ids, Y, feature_song_ids)

    logger.info(
        "Split inputs: X=%s, Y=%s  |  TEST_SIZE=%.2f, SEED=%d",
        X.shape,
        Y_aligned.shape,
        config.TEST_SIZE,
        config.RANDOM_SEED,
    )

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y_aligned,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
    )

    logger.info(
        "Split complete: train=%d, test=%d samples",
        len(X_train),
        len(X_test),
    )
    return X_train, X_test, Y_train, Y_test


def phase3_2_traditional_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    label_names: list[str],
) -> Dict[str, Any]:
    """Phase 3.2: Train and evaluate traditional ML models.

    Delegates to ``traditional_models.run_all_traditional_models()``.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        PCA-reduced fused feature matrices for train and test sets.
    Y_train, Y_test : np.ndarray
        Binary label matrices for train and test sets.
    label_names : list[str]
        Ordered class names (used for per-class reporting).

    Returns
    -------
    dict
        Mapping ``{model_name: metrics_dict}`` for all trained models.
    """
    log_section(logger, "Phase 3.2 — Traditional ML Models")
    t0 = time.time()

    results: Dict[str, Any] = traditional_models.run_all_traditional_models(
        X_train=X_train,
        X_test=X_test,
        Y_train=Y_train,
        Y_test=Y_test,
        label_names=label_names,
    )

    logger.info(
        "Phase 3.2 complete: trained %d models  [%s]",
        len(results),
        _elapsed(t0),
    )
    for name, metrics in results.items():
        f1_micro = metrics.get("f1_micro", float("nan"))
        f1_macro = metrics.get("f1_macro", float("nan"))
        logger.info(
            "  %-30s  f1_micro=%.4f  f1_macro=%.4f",
            name,
            f1_micro,
            f1_macro,
        )
    return results


def _split_modalities(
    audio: np.ndarray,
    text: np.ndarray,
    Y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split audio, text, and label arrays consistently using index-based split.

    Parameters
    ----------
    audio : np.ndarray
        Audio feature matrix ``(n, d_a)``.
    text : np.ndarray
        Text feature matrix ``(n, d_t)``.
    Y : np.ndarray
        Label matrix ``(n, n_classes)``.

    Returns
    -------
    audio_train, audio_test, text_train, text_test, Y_train, Y_test : np.ndarray
        Corresponding train and test splits for each array.
    """
    indices = np.arange(len(audio))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
    )
    return (
        audio[train_idx],
        audio[test_idx],
        text[train_idx],
        text[test_idx],
        Y[train_idx],
        Y[test_idx],
    )


def phase3_3_mlp_model(
    audio_arr: np.ndarray,
    audio_ids: list[str],
    text_arr: np.ndarray,
    text_ids: list[str],
    Y: np.ndarray,
    label_song_ids: list[str],
    label_names: list[str],
) -> Dict[str, Any]:
    """Phase 3.3: Train and evaluate the MLP neural model.

    Audio and text feature arrays are re-aligned independently to a shared
    inner-join ID set before being passed to the MLP pipeline, which handles
    modality-aware concatenation internally.

    Parameters
    ----------
    audio_arr : np.ndarray
        Full audio feature matrix.
    audio_ids : list[str]
        Song IDs for *audio_arr* rows.
    text_arr : np.ndarray
        Full text feature matrix.
    text_ids : list[str]
        Song IDs for *text_arr* rows.
    Y : np.ndarray
        Full label matrix.
    label_song_ids : list[str]
        Song IDs for rows of *Y*.
    label_names : list[str]
        Ordered class names.

    Returns
    -------
    dict
        Metrics dictionary produced by the MLP pipeline.
    """
    log_section(logger, "Phase 3.3 — MLP Neural Model")
    t0 = time.time()

    # Align audio & text to a shared ID set (inner join)
    shared_ids, aligned_audio, aligned_text = _align_ids(
        audio_ids, audio_arr, text_ids, text_arr
    )

    if len(shared_ids) == 0:
        raise RuntimeError(
            "No shared song IDs between audio and text feature sets for MLP — "
            "cannot train MLP model."
        )

    # Align labels to the shared ID order
    Y_aligned = _align_labels(label_song_ids, Y, shared_ids)

    # Consistent train/test split across all three arrays
    (
        audio_train, audio_test,
        text_train,  text_test,
        Y_train,     Y_test,
    ) = _split_modalities(aligned_audio, aligned_text, Y_aligned)

    logger.info(
        "MLP split: train=%d, test=%d  |  audio_d=%d, text_d=%d, classes=%d",
        len(audio_train),
        len(audio_test),
        aligned_audio.shape[1],
        aligned_text.shape[1],
        Y_aligned.shape[1],
    )

    mlp_results: Dict[str, Any] = neural_models.run_mlp_pipeline(
        audio_train=audio_train,
        audio_test=audio_test,
        text_train=text_train,
        text_test=text_test,
        Y_train=Y_train,
        Y_test=Y_test,
        label_names=label_names,
    )

    f1_micro = mlp_results.get("f1_micro", float("nan"))
    f1_macro = mlp_results.get("f1_macro", float("nan"))
    logger.info(
        "Phase 3.3 complete: MLP  f1_micro=%.4f  f1_macro=%.4f  [%s]",
        f1_micro,
        f1_macro,
        _elapsed(t0),
    )
    return mlp_results


def phase_results_comparison(
    all_results: Dict[str, Dict[str, Any]],
    label_names: list[str],
) -> None:
    """Compare all model results and write a consolidated report.

    Calls ``evaluation.compare_models()`` to persist a structured JSON/CSV
    report, then logs a sorted summary table to the pipeline log.

    Parameters
    ----------
    all_results : dict
        Mapping ``{model_name: metrics_dict}`` for every trained model.
    label_names : list[str]
        Ordered class names (passed through to comparison utilities).
    """
    log_section(logger, "Results Comparison & Final Summary")

    try:
        evaluation.compare_models(
            results=all_results,
            label_names=label_names,
            output_path=_RESULTS_JSON,
        )
        logger.info("Comparison report saved -> %s", _RESULTS_JSON)
    except Exception as exc:
        logger.error("compare_models() failed: %s", exc)
        logger.debug(traceback.format_exc())

    # Always log a tidy console summary regardless of report-write success
    separator = "=" * 70
    logger.info(separator)
    logger.info("  FINAL MODEL COMPARISON SUMMARY")
    logger.info(separator)
    header = (
        f"{'Model':<35}  {'F1-micro':>10}  {'F1-macro':>10}  {'Hamming':>10}"
    )
    logger.info(header)
    logger.info("-" * len(header))

    sorted_results = sorted(
        all_results.items(),
        key=lambda kv: kv[1].get("f1_micro", 0.0),
        reverse=True,
    )
    for name, metrics in sorted_results:
        f1_micro  = metrics.get("f1_micro",    float("nan"))
        f1_macro  = metrics.get("f1_macro",    float("nan"))
        hamming   = metrics.get("hamming_loss", float("nan"))
        logger.info(
            "%-35s  %10.4f  %10.4f  %10.4f",
            name,
            f1_micro,
            f1_macro,
            hamming,
        )
    logger.info(separator)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full baseline training pipeline end-to-end.

    Each major phase is wrapped in an individual try/except so that a failure
    in any single stage is logged and the pipeline continues with subsequent
    stages where possible.  Downstream stages that depend on the failed
    stage's output will catch their own missing-variable errors and skip
    gracefully with an informative warning.
    """
    run_start = time.time()
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ------------------------------------------------------------------
    # Logging setup — must happen before ensure_dirs so the log file path
    # is valid.  REPORTS_DIR is created by ensure_dirs below, but
    # setup_logger also calls parent.mkdir(parents=True, exist_ok=True).
    # ------------------------------------------------------------------
    log_file = config.REPORTS_DIR / f"train_baseline_{run_ts}.log"
    setup_logger("music_genre", log_file=log_file)

    log_section(
        logger,
        f"Multi-Modal Music Genre Classification — Baseline Pipeline  [{run_ts}]",
    )
    logger.info("Log file  -> %s", log_file)
    logger.info("Project   -> %s", config.PROJECT_ROOT)

    # ------------------------------------------------------------------
    # Ensure all output directories exist
    # ------------------------------------------------------------------
    try:
        config.ensure_dirs()
        logger.info("Output directories verified.")
    except Exception as exc:
        logger.critical("ensure_dirs() failed: %s — aborting.", exc)
        sys.exit(1)

    # Mutable accumulators — remain None until the corresponding phase
    # succeeds, so downstream guards can safely check `is not None`.
    df: Optional[pd.DataFrame] = None
    Y: Optional[np.ndarray] = None
    label_names: Optional[list[str]] = None
    label_song_ids: Optional[list[str]] = None
    audio_arr: Optional[np.ndarray] = None
    audio_ids: Optional[list[str]] = None
    text_arr: Optional[np.ndarray] = None
    text_ids: Optional[list[str]] = None
    X_pca: Optional[np.ndarray] = None
    fused_ids: Optional[list[str]] = None
    X_train: Optional[np.ndarray] = None
    X_test: Optional[np.ndarray] = None
    Y_train: Optional[np.ndarray] = None
    Y_test: Optional[np.ndarray] = None
    all_results: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Phase 1 — Data loading & cleaning
    # ------------------------------------------------------------------
    try:
        df = phase1_data_loading()
    except Exception as exc:
        logger.error("Phase 1 FAILED: %s", exc)
        logger.debug(traceback.format_exc())
        logger.critical("Cannot continue without a dataset — aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 1.4 — Label encoding
    # ------------------------------------------------------------------
    try:
        Y, label_names, label_song_ids = phase1_4_label_encoding(df)
    except Exception as exc:
        logger.error("Phase 1.4 FAILED: %s", exc)
        logger.debug(traceback.format_exc())
        logger.critical("Cannot continue without label matrix — aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 2.1 — Audio feature extraction
    # ------------------------------------------------------------------
    try:
        audio_df = df[df["has_audio"].astype(bool)][["song_id", "mp3_path"]].copy()
        song_ids_for_audio: list[str] = audio_df["song_id"].tolist()
        mp3_paths_for_audio: list[str] = audio_df["mp3_path"].tolist()

        audio_arr, audio_ids = phase2_1_audio_features(
            song_ids=song_ids_for_audio,
            mp3_paths=mp3_paths_for_audio,
        )
    except Exception as exc:
        logger.error("Phase 2.1 FAILED: %s", exc)
        logger.debug(traceback.format_exc())
        logger.warning(
            "Audio features unavailable — MLP and fusion stages will be skipped."
        )

    # ------------------------------------------------------------------
    # Phase 2.2 — Text feature extraction
    # ------------------------------------------------------------------
    try:
        text_arr, text_ids = phase2_2_text_features(df)
    except Exception as exc:
        logger.error("Phase 2.2 FAILED: %s", exc)
        logger.debug(traceback.format_exc())
        logger.warning(
            "Text features unavailable — MLP and fusion stages will be skipped."
        )

    # ------------------------------------------------------------------
    # Phase 2.5 — Feature alignment, fusion & PCA
    # ------------------------------------------------------------------
    if audio_arr is not None and text_arr is not None:
        try:
            X_pca, fused_ids = phase2_5_feature_fusion(
                audio_arr,
                audio_ids,  # type: ignore[arg-type]
                text_arr,
                text_ids,   # type: ignore[arg-type]
            )
        except Exception as exc:
            logger.error("Phase 2.5 FAILED: %s", exc)
            logger.debug(traceback.format_exc())
            logger.warning(
                "Fused features unavailable — traditional model training skipped."
            )
    else:
        logger.warning(
            "Skipping Phase 2.5 — one or both feature modalities are unavailable."
        )

    # ------------------------------------------------------------------
    # Train / test split
    # ------------------------------------------------------------------
    if X_pca is not None and fused_ids is not None:
        try:
            X_train, X_test, Y_train, Y_test = phase_train_test_split(
                X_pca,
                Y,                  # type: ignore[arg-type]
                label_song_ids,     # type: ignore[arg-type]
                fused_ids,
            )
        except Exception as exc:
            logger.error("Train/test split FAILED: %s", exc)
            logger.debug(traceback.format_exc())
            logger.warning(
                "Skipping downstream model training — split failed."
            )
    else:
        logger.warning(
            "Skipping train/test split — fused feature matrix not available."
        )

    # ------------------------------------------------------------------
    # Phase 3.2 — Traditional ML models
    # ------------------------------------------------------------------
    if X_train is not None and Y_train is not None:
        try:
            trad_results = phase3_2_traditional_models(
                X_train,
                X_test,      # type: ignore[arg-type]
                Y_train,
                Y_test,      # type: ignore[arg-type]
                label_names, # type: ignore[arg-type]
            )
            all_results.update(trad_results)
        except Exception as exc:
            logger.error("Phase 3.2 FAILED: %s", exc)
            logger.debug(traceback.format_exc())
    else:
        logger.warning(
            "Skipping Phase 3.2 — train/test splits not available."
        )

    # ------------------------------------------------------------------
    # Phase 3.3 — MLP neural model
    # ------------------------------------------------------------------
    if (
        audio_arr is not None
        and audio_ids is not None
        and text_arr is not None
        and text_ids is not None
        and Y is not None
    ):
        try:
            mlp_result = phase3_3_mlp_model(
                audio_arr,
                audio_ids,
                text_arr,
                text_ids,
                Y,
                label_song_ids,  # type: ignore[arg-type]
                label_names,     # type: ignore[arg-type]
            )
            all_results["MLP"] = mlp_result
        except Exception as exc:
            logger.error("Phase 3.3 FAILED: %s", exc)
            logger.debug(traceback.format_exc())
    else:
        logger.warning(
            "Skipping Phase 3.3 — audio or text features (or labels) unavailable."
        )

    # ------------------------------------------------------------------
    # Results comparison
    # ------------------------------------------------------------------
    if all_results and label_names is not None:
        try:
            phase_results_comparison(all_results, label_names)
        except Exception as exc:
            logger.error("Results comparison FAILED: %s", exc)
            logger.debug(traceback.format_exc())
    else:
        logger.warning(
            "No model results to compare (all training phases may have failed)."
        )

    # ------------------------------------------------------------------
    # Pipeline complete
    # ------------------------------------------------------------------
    log_section(
        logger,
        f"Pipeline finished  [total elapsed: {_elapsed(run_start)}]",
    )
    logger.info(
        "Models trained: %d  |  Results report: %s",
        len(all_results),
        _RESULTS_JSON if _RESULTS_JSON.exists() else "not written",
    )


# ---------------------------------------------------------------------------
# Entry-point guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
