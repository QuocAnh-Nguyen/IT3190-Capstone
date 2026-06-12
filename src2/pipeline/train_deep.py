"""Deep Learning Training Pipeline — Phase 3.4 (Multi-Modal Music Genre Classification).

WARNING: DEFERRED EXECUTION
================================================================================================
This pipeline uses pre-trained deep learning models (BERT, AST) and requires:
  - GPU with CUDA support
  - transformers, torchaudio libraries installed
  - Significant compute time (hours)
This script is fully implemented but should NOT be run during the baseline phase.
================================================================================================

Overview
--------
This module mirrors ``train_baseline.py`` (Phase 3.3) but replaces hand-crafted
traditional features with dense embeddings produced by large pre-trained models:

* **Audio embeddings** — Audio Spectrogram Transformer (AST / wav2vec2) extracted
  via :mod:`src2.features.audio_deeplearning`.
* **Text embeddings** — BERT / sentence-transformers extracted via
  :mod:`src2.features.text_deeplearning`.

Because DL embeddings are already compact dense vectors, no PCA dimensionality
reduction step is needed before fusion.  The same late-fusion concatenation
strategy as the baseline is applied, then four classifier families are evaluated:

1. One-vs-Rest Random Forest (OVR-RF)
2. Classifier Chains with XGBoost base
3. XGBoost multi-output
4. MLP deep classifier (via :mod:`src2.models.neural_models`)

Results are persisted under ``outputs/reports/`` and ``outputs/models/`` so they
can be compared with baseline results in a later evaluation phase.

Execution guard
---------------
A ``DEFERRED`` sentinel at the bottom of the module prevents accidental execution
during the baseline phase.  Remove or bypass the guard only when running on a
GPU-equipped environment with the required libraries installed.
"""

# DEFERRED: not executed during baseline phase

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src2 import config
from src2.data import data_loader, data_cleaner, label_encoder
from src2.features import audio_deeplearning, text_deeplearning, feature_reducer  # noqa: F401
from src2.models import traditional_models, neural_models, evaluation
from src2.utils.logging_utils import setup_logger, log_section
from src2.utils.io_utils import (
    save_pickle,
    load_pickle,
    save_npy,
    load_npy,
    save_json,
    save_text_list,
    load_text_list,
)

# ---------------------------------------------------------------------------
# Module-level logger — all child modules share the same "music_genre" root.
# ---------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger("music_genre")

# ---------------------------------------------------------------------------
# Artefact paths specific to this pipeline (DL variants)
# ---------------------------------------------------------------------------
_DL_AUDIO_EMB_PATH: Path = config.FEATURES_DIR / "dl_audio_embeddings.npy"
_DL_TEXT_EMB_PATH: Path = config.FEATURES_DIR / "dl_text_embeddings.npy"
_DL_FUSED_EMB_PATH: Path = config.FEATURES_DIR / "dl_fused_embeddings.npy"
_DL_SONG_IDS_PATH: Path = config.FEATURES_DIR / "dl_song_ids.txt"

_DL_RESULTS_JSON: Path = config.REPORTS_DIR / "phase34_dl_results.json"
_DL_REPORT_TXT: Path = config.REPORTS_DIR / "phase34_dl_report.txt"
_DL_MODELS_DIR: Path = config.MODELS_DIR / "dl_pipeline"


# ===========================================================================
# Section 1 — Directory and logging setup
# ===========================================================================

def setup_environment() -> None:
    """Initialise output directories and configure file-based logging.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    config.ensure_dirs()
    _DL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    log_file = config.REPORTS_DIR / "train_deep.log"
    setup_logger(name="music_genre", log_file=log_file, level=logging.INFO)

    log_section(logger, "Phase 3.4 — Deep Learning Training Pipeline")
    logger.info("Output root  : %s", config.OUTPUT_DIR)
    logger.info("DL models dir: %s", _DL_MODELS_DIR)
    logger.warning(
        "DEFERRED PIPELINE — ensure CUDA GPU, transformers, and torchaudio "
        "are available before running this script."
    )


# ===========================================================================
# Section 2 — Load Phase 1 artefacts (cleaned dataset + label matrix)
# ===========================================================================

def load_phase1_artifacts() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Load the cleaned dataset and label artefacts produced by Phase 1.

    Reuses the exact same files as the baseline pipeline so both pipelines
    operate on an identical data split.

    Parameters
    ----------
    None

    Returns
    -------
    df : pd.DataFrame
        Cleaned dataset with all metadata and lyrics columns.
    label_matrix : np.ndarray of shape (n_samples, n_classes)
        Binary multi-label indicator matrix.
    label_names : list[str]
        Ordered genre label names corresponding to columns of *label_matrix*.

    Raises
    ------
    FileNotFoundError
        If any of the required Phase 1 artefact files are missing.
    """
    log_section(logger, "Loading Phase 1 Artefacts")

    required = [
        config.CLEANED_DATASET_CSV,
        config.LABEL_MATRIX_NPY,
        config.LABEL_NAMES_TXT,
        config.SONG_IDS_TXT,
        config.MLB_PKL,
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(
                f"Phase 1 artefact not found: {p}. "
                "Run the Phase 1 data pipeline (data_pipeline.py) first."
            )

    logger.info("Reading cleaned dataset from: %s", config.CLEANED_DATASET_CSV)
    df: pd.DataFrame = data_loader.load_cleaned_dataset(config.CLEANED_DATASET_CSV)
    logger.info("Cleaned dataset shape: %s", df.shape)

    logger.info("Loading label matrix from: %s", config.LABEL_MATRIX_NPY)
    label_matrix: np.ndarray = load_npy(config.LABEL_MATRIX_NPY)
    logger.info("Label matrix shape: %s", label_matrix.shape)

    logger.info("Loading label names from: %s", config.LABEL_NAMES_TXT)
    label_names: list[str] = load_text_list(config.LABEL_NAMES_TXT)
    logger.info("Number of genre labels: %d", len(label_names))

    return df, label_matrix, label_names


# ===========================================================================
# Section 3 — Audio DL embeddings  [DEFERRED]
# ===========================================================================

def extract_audio_dl_embeddings(df: pd.DataFrame) -> np.ndarray:
    """Extract audio embeddings via the Audio Spectrogram Transformer (AST).

    .. note::
        DEFERRED — requires GPU + torchaudio + transformers.
        Embeddings are cached to disk; subsequent runs load from cache.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset.  Must contain a column that maps rows to audio file
        paths (resolved via :attr:`config.AUDIO_PREVIEWS_DIR`).

    Returns
    -------
    audio_embeddings : np.ndarray of shape (n_samples, audio_embed_dim)
        Dense audio embeddings for every track in *df*.
    """
    # DEFERRED: not executed during baseline phase

    log_section(logger, "Audio DL Embeddings (AST / wav2vec2)")

    if _DL_AUDIO_EMB_PATH.exists():
        logger.info(
            "Cached audio DL embeddings found at %s — loading.", _DL_AUDIO_EMB_PATH
        )
        audio_embeddings: np.ndarray = load_npy(_DL_AUDIO_EMB_PATH)
        logger.info("Audio DL embeddings shape (from cache): %s", audio_embeddings.shape)
        return audio_embeddings

    logger.info(
        "No cache found.  Extracting audio DL embeddings for %d tracks …", len(df)
    )
    logger.warning(
        "This step may take several hours on a single GPU.  "
        "Ensure sufficient VRAM (>= 8 GB recommended)."
    )

    try:
        audio_embeddings = audio_deeplearning.extract_all_embeddings(df)
    except Exception as exc:
        logger.error(
            "Audio DL embedding extraction failed: %s", exc, exc_info=True
        )
        raise

    logger.info("Audio DL embeddings shape: %s", audio_embeddings.shape)
    save_npy(audio_embeddings, _DL_AUDIO_EMB_PATH)
    logger.info("Audio DL embeddings cached to: %s", _DL_AUDIO_EMB_PATH)

    return audio_embeddings


# ===========================================================================
# Section 4 — Text DL embeddings  [DEFERRED]
# ===========================================================================

def extract_text_dl_embeddings(df: pd.DataFrame) -> np.ndarray:
    """Extract text embeddings from lyrics using BERT / sentence-transformers.

    .. note::
        DEFERRED — requires GPU + transformers library.
        Embeddings are cached to disk; subsequent runs load from cache.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset.  Must contain a ``lyrics`` (or equivalent) column
        consumed by :func:`text_deeplearning.extract_all_embeddings`.

    Returns
    -------
    text_embeddings : np.ndarray of shape (n_samples, text_embed_dim)
        Dense BERT-style sentence embeddings for every track in *df*.
    """
    # DEFERRED: not executed during baseline phase

    log_section(logger, "Text DL Embeddings (BERT / sentence-transformers)")

    if _DL_TEXT_EMB_PATH.exists():
        logger.info(
            "Cached text DL embeddings found at %s — loading.", _DL_TEXT_EMB_PATH
        )
        text_embeddings: np.ndarray = load_npy(_DL_TEXT_EMB_PATH)
        logger.info("Text DL embeddings shape (from cache): %s", text_embeddings.shape)
        return text_embeddings

    logger.info(
        "No cache found.  Extracting text DL embeddings for %d tracks …", len(df)
    )
    logger.warning(
        "This step encodes lyrics with a large pre-trained language model.  "
        "Expect ~15-60 minutes depending on dataset size and hardware."
    )

    try:
        text_embeddings = text_deeplearning.extract_all_embeddings(df)
    except Exception as exc:
        logger.error(
            "Text DL embedding extraction failed: %s", exc, exc_info=True
        )
        raise

    logger.info("Text DL embeddings shape: %s", text_embeddings.shape)
    save_npy(text_embeddings, _DL_TEXT_EMB_PATH)
    logger.info("Text DL embeddings cached to: %s", _DL_TEXT_EMB_PATH)

    return text_embeddings


# ===========================================================================
# Section 5 — Late fusion of DL embeddings
# ===========================================================================

def fuse_dl_embeddings(
    audio_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
) -> np.ndarray:
    """Concatenate audio and text DL embeddings into a unified feature matrix.

    Unlike the baseline pipeline, **no PCA** is applied here because DL
    embeddings are already compact, information-dense vectors.  Simple
    concatenation is sufficient and avoids discarding representational nuances
    learned by the pre-trained backbones.

    Parameters
    ----------
    audio_embeddings : np.ndarray of shape (n_samples, d_audio)
        Dense audio embeddings from AST / wav2vec2.
    text_embeddings : np.ndarray of shape (n_samples, d_text)
        Dense BERT / sentence-transformer embeddings.

    Returns
    -------
    fused : np.ndarray of shape (n_samples, d_audio + d_text)
        Horizontally concatenated embedding matrix.

    Raises
    ------
    ValueError
        If *audio_embeddings* and *text_embeddings* have mismatched first
        dimensions (i.e., different numbers of samples).
    """
    log_section(logger, "Late Fusion of DL Embeddings (Concatenation — No PCA)")

    n_audio, n_text = audio_embeddings.shape[0], text_embeddings.shape[0]
    if n_audio != n_text:
        raise ValueError(
            f"Sample count mismatch between audio ({n_audio}) and text "
            f"({n_text}) embeddings.  Verify that both were extracted from "
            "the same aligned dataset."
        )

    logger.info(
        "Audio embedding dim: %d | Text embedding dim: %d",
        audio_embeddings.shape[1],
        text_embeddings.shape[1],
    )

    fused: np.ndarray = np.concatenate([audio_embeddings, text_embeddings], axis=1)
    logger.info("Fused DL embedding matrix shape: %s", fused.shape)

    save_npy(fused, _DL_FUSED_EMB_PATH)
    logger.info("Fused DL embeddings saved to: %s", _DL_FUSED_EMB_PATH)

    return fused


# ===========================================================================
# Section 6 — Train / test split
# ===========================================================================

def split_data(
    X: np.ndarray,
    y: np.ndarray,
    song_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Stratified multi-label train/test split on fused DL features.

    Uses the same ``TEST_SIZE`` and ``RANDOM_SEED`` constants as the baseline
    pipeline to ensure comparable evaluation conditions.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Fused DL feature matrix.
    y : np.ndarray of shape (n_samples, n_classes)
        Multi-label indicator matrix.
    song_ids : list[str]
        Song identifiers aligned with *X* and *y*.

    Returns
    -------
    X_train : np.ndarray
    X_test  : np.ndarray
    y_train : np.ndarray
    y_test  : np.ndarray
    ids_train : list[str]
    ids_test  : list[str]
    """
    log_section(logger, "Train / Test Split")

    from sklearn.model_selection import train_test_split

    logger.info(
        "Splitting data: test_size=%.2f, random_seed=%d",
        config.TEST_SIZE,
        config.RANDOM_SEED,
    )

    # Multi-label stratification via iterative splitting
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_SEED,
        )
        train_idx, test_idx = next(msss.split(X, y))
        logger.info("Using MultilabelStratifiedShuffleSplit.")
    except ImportError:
        logger.warning(
            "iterstrat not available — falling back to random shuffle split."
        )
        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_SEED,
        )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    ids_train = [song_ids[i] for i in train_idx]
    ids_test = [song_ids[i] for i in test_idx]

    logger.info("Training samples : %d", len(X_train))
    logger.info("Test samples     : %d", len(X_test))
    logger.info("Feature dimension: %d", X_train.shape[1])
    logger.info("Label dimension  : %d", y_train.shape[1])

    return X_train, X_test, y_train, y_test, ids_train, ids_test


# ===========================================================================
# Section 7 — Traditional classifiers on DL features
# ===========================================================================

def run_traditional_classifiers_on_dl(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
) -> dict[str, Any]:
    """Fit and evaluate traditional multi-label classifiers on DL embeddings.

    Three classifier families are assessed:

    1. **OVR-RF** — One-vs-Rest Random Forest
    2. **CC-XGB** — Classifier Chains with XGBoost as base estimator
    3. **XGBoost** — Natively multi-output XGBoost

    Parameters
    ----------
    X_train : np.ndarray
        Training features (fused DL embeddings).
    X_test : np.ndarray
        Test features.
    y_train : np.ndarray
        Training labels.
    y_test : np.ndarray
        Test labels.
    label_names : list[str]
        Genre label names for per-class reporting.

    Returns
    -------
    results : dict[str, Any]
        Nested dict mapping classifier name to evaluation metric dict.
    """
    # DEFERRED: not executed during baseline phase

    log_section(logger, "Traditional Classifiers on DL Embeddings")

    results: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 7a. One-vs-Rest Random Forest
    # ------------------------------------------------------------------
    logger.info("Training OVR Random Forest on DL embeddings ...")
    t0 = time.time()
    try:
        ovr_rf = traditional_models.build_ovr_random_forest(
            random_state=config.RANDOM_SEED
        )
        ovr_rf.fit(X_train, y_train)
        y_pred_ovr = ovr_rf.predict(X_test)
        metrics_ovr = evaluation.evaluate_multilabel(
            y_test, y_pred_ovr, label_names=label_names
        )
        results["ovr_rf_dl"] = metrics_ovr
        elapsed = time.time() - t0
        logger.info(
            "OVR-RF (DL) — micro-F1: %.4f | macro-F1: %.4f | elapsed: %.1fs",
            metrics_ovr.get("micro_f1", float("nan")),
            metrics_ovr.get("macro_f1", float("nan")),
            elapsed,
        )
        save_pickle(ovr_rf, _DL_MODELS_DIR / "ovr_rf_dl.pkl")
    except Exception as exc:
        logger.error("OVR-RF (DL) training failed: %s", exc, exc_info=True)
        results["ovr_rf_dl"] = {"error": str(exc)}

    # ------------------------------------------------------------------
    # 7b. Classifier Chains with XGBoost base
    # ------------------------------------------------------------------
    logger.info("Training Classifier Chains (XGBoost base) on DL embeddings ...")
    t0 = time.time()
    try:
        cc_xgb = traditional_models.build_classifier_chains_xgb(
            random_state=config.RANDOM_SEED
        )
        cc_xgb.fit(X_train, y_train)
        y_pred_cc = cc_xgb.predict(X_test)
        metrics_cc = evaluation.evaluate_multilabel(
            y_test, y_pred_cc, label_names=label_names
        )
        results["cc_xgb_dl"] = metrics_cc
        elapsed = time.time() - t0
        logger.info(
            "CC-XGB (DL) — micro-F1: %.4f | macro-F1: %.4f | elapsed: %.1fs",
            metrics_cc.get("micro_f1", float("nan")),
            metrics_cc.get("macro_f1", float("nan")),
            elapsed,
        )
        save_pickle(cc_xgb, _DL_MODELS_DIR / "cc_xgb_dl.pkl")
    except Exception as exc:
        logger.error("CC-XGB (DL) training failed: %s", exc, exc_info=True)
        results["cc_xgb_dl"] = {"error": str(exc)}

    # ------------------------------------------------------------------
    # 7c. Multi-output XGBoost
    # ------------------------------------------------------------------
    logger.info("Training Multi-output XGBoost on DL embeddings ...")
    t0 = time.time()
    try:
        mo_xgb = traditional_models.build_multioutput_xgb(
            random_state=config.RANDOM_SEED
        )
        mo_xgb.fit(X_train, y_train)
        y_pred_xgb = mo_xgb.predict(X_test)
        metrics_xgb = evaluation.evaluate_multilabel(
            y_test, y_pred_xgb, label_names=label_names
        )
        results["mo_xgb_dl"] = metrics_xgb
        elapsed = time.time() - t0
        logger.info(
            "MO-XGB (DL) — micro-F1: %.4f | macro-F1: %.4f | elapsed: %.1fs",
            metrics_xgb.get("micro_f1", float("nan")),
            metrics_xgb.get("macro_f1", float("nan")),
            elapsed,
        )
        save_pickle(mo_xgb, _DL_MODELS_DIR / "mo_xgb_dl.pkl")
    except Exception as exc:
        logger.error("MO-XGB (DL) training failed: %s", exc, exc_info=True)
        results["mo_xgb_dl"] = {"error": str(exc)}

    return results


# ===========================================================================
# Section 8 — MLP deep classifier on DL features
# ===========================================================================

def run_mlp_dl_pipeline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
) -> dict[str, Any]:
    """Train and evaluate the MLP deep classifier on DL embeddings.

    Delegates to :func:`neural_models.run_mlp_dl_pipeline` which wraps
    PyTorch model construction, training loop with early stopping, and
    inference.

    Parameters
    ----------
    X_train : np.ndarray
        Training DL feature matrix.
    X_test : np.ndarray
        Test DL feature matrix.
    y_train : np.ndarray
        Training label matrix.
    y_test : np.ndarray
        Test label matrix.
    label_names : list[str]
        Genre label names for per-class reporting.

    Returns
    -------
    mlp_results : dict[str, Any]
        Evaluation metrics for the MLP model keyed as ``"mlp_dl"``.
    """
    # DEFERRED: not executed during baseline phase

    log_section(logger, "MLP Deep Classifier on DL Embeddings")

    mlp_results: dict[str, Any] = {}

    logger.info(
        "MLP config — hidden_dims=%s, dropout=%.2f, lr=%.0e, epochs=%d, patience=%d",
        config.MLP_HIDDEN_DIMS,
        config.MLP_DROPOUT,
        config.MLP_LR,
        config.MLP_EPOCHS,
        config.MLP_EARLY_STOPPING_PATIENCE,
    )

    t0 = time.time()
    try:
        mlp_metrics, mlp_model = neural_models.run_mlp_dl_pipeline(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            label_names=label_names,
            hidden_dims=config.MLP_HIDDEN_DIMS,
            dropout=config.MLP_DROPOUT,
            lr=config.MLP_LR,
            epochs=config.MLP_EPOCHS,
            patience=config.MLP_EARLY_STOPPING_PATIENCE,
            batch_size=config.BATCH_SIZE,
            random_state=config.RANDOM_SEED,
            model_save_path=_DL_MODELS_DIR / "mlp_dl.pt",
        )
        elapsed = time.time() - t0
        mlp_results["mlp_dl"] = mlp_metrics
        logger.info(
            "MLP (DL) — micro-F1: %.4f | macro-F1: %.4f | elapsed: %.1fs",
            mlp_metrics.get("micro_f1", float("nan")),
            mlp_metrics.get("macro_f1", float("nan")),
            elapsed,
        )
    except Exception as exc:
        logger.error("MLP (DL) pipeline failed: %s", exc, exc_info=True)
        mlp_results["mlp_dl"] = {"error": str(exc)}

    return mlp_results


# ===========================================================================
# Section 9 — Aggregate, compare, and persist results
# ===========================================================================

def compare_and_save_results(
    all_results: dict[str, Any],
    label_names: list[str],
) -> None:
    """Compile all classifier results into a summary report and persist artefacts.

    Parameters
    ----------
    all_results : dict[str, Any]
        Mapping of classifier name to metric dict, collected across all
        training sections.
    label_names : list[str]
        Genre label names for the per-class section of the text report.

    Returns
    -------
    None
    """
    log_section(logger, "Results Summary — Phase 3.4 DL Pipeline")

    # ------------------------------------------------------------------
    # Build comparison table
    # ------------------------------------------------------------------
    rows: list[dict[str, Any]] = []
    for clf_name, metrics in all_results.items():
        if "error" in metrics:
            rows.append(
                {
                    "classifier": clf_name,
                    "micro_f1": float("nan"),
                    "macro_f1": float("nan"),
                    "hamming_loss": float("nan"),
                    "subset_accuracy": float("nan"),
                    "status": f"ERROR: {metrics['error']}",
                }
            )
        else:
            rows.append(
                {
                    "classifier": clf_name,
                    "micro_f1": metrics.get("micro_f1", float("nan")),
                    "macro_f1": metrics.get("macro_f1", float("nan")),
                    "hamming_loss": metrics.get("hamming_loss", float("nan")),
                    "subset_accuracy": metrics.get("subset_accuracy", float("nan")),
                    "status": "OK",
                }
            )

    summary_df = pd.DataFrame(rows).sort_values("micro_f1", ascending=False)

    # ------------------------------------------------------------------
    # Log table to console / log file
    # ------------------------------------------------------------------
    logger.info("\n%s", summary_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Persist JSON results
    # ------------------------------------------------------------------
    try:
        save_json(all_results, _DL_RESULTS_JSON)
        logger.info("Full results JSON saved to: %s", _DL_RESULTS_JSON)
    except Exception as exc:
        logger.error("Failed to save results JSON: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Persist human-readable text report
    # ------------------------------------------------------------------
    try:
        _write_text_report(summary_df, all_results, label_names)
        logger.info("Text report saved to: %s", _DL_REPORT_TXT)
    except Exception as exc:
        logger.error("Failed to write text report: %s", exc, exc_info=True)


def _write_text_report(
    summary_df: pd.DataFrame,
    all_results: dict[str, Any],
    label_names: list[str],
) -> None:
    """Write a human-readable plain-text report for Phase 3.4.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Classifier comparison table.
    all_results : dict[str, Any]
        Full nested results dict (includes per-class metrics when present).
    label_names : list[str]
        Genre label names.

    Returns
    -------
    None
    """
    _DL_REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
    sep = "=" * 80
    thin = "-" * 80

    with open(_DL_REPORT_TXT, "w", encoding="utf-8") as fh:
        fh.write(f"{sep}\n")
        fh.write("Phase 3.4 — Deep Learning Pipeline Results\n")
        fh.write(f"{sep}\n\n")

        fh.write("Classifier Comparison (sorted by micro-F1)\n")
        fh.write(f"{thin}\n")
        fh.write(summary_df.to_string(index=False))
        fh.write(f"\n\n{sep}\n\n")

        fh.write(f"Number of genre labels: {len(label_names)}\n")
        fh.write("Labels: " + ", ".join(label_names) + "\n\n")

        for clf_name, metrics in all_results.items():
            fh.write(f"{thin}\n")
            fh.write(f"Classifier: {clf_name}\n")
            fh.write(f"{thin}\n")
            if "error" in metrics:
                fh.write(f"  ERROR: {metrics['error']}\n\n")
                continue
            for key, val in metrics.items():
                if key == "per_class":
                    continue
                fh.write(f"  {key:30s}: {val}\n")
            # Per-class block (if present)
            per_class: dict[str, Any] = metrics.get("per_class", {})
            if per_class:
                fh.write("\n  Per-class F1 scores:\n")
                for genre, f1 in per_class.items():
                    fh.write(f"    {genre:30s}: {f1:.4f}\n")
            fh.write("\n")

        fh.write(f"{sep}\n")
        fh.write("END OF REPORT\n")
        fh.write(f"{sep}\n")


# ===========================================================================
# Section 10 — Main entry point  [DEFERRED]
# ===========================================================================

def main() -> None:  # DEFERRED: not executed during baseline phase
    """Orchestrate the full Phase 3.4 deep learning training pipeline.

    Execution sequence
    ------------------
    1. Environment setup (dirs, logging).
    2. Load Phase 1 artefacts (cleaned dataset, label matrix, label names).
    3. Extract audio DL embeddings via AST / wav2vec2.
    4. Extract text DL embeddings via BERT.
    5. Late-fuse embeddings (concatenation; no PCA).
    6. Stratified train/test split.
    7. Train and evaluate traditional classifiers on DL features.
    8. Train and evaluate MLP deep classifier.
    9. Aggregate, compare, and persist results.

    .. warning::
        DEFERRED — do NOT invoke this function during the baseline phase.
        A GPU with CUDA support and the ``transformers`` / ``torchaudio``
        libraries must be available.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # DEFERRED: not executed during baseline phase

    pipeline_start = time.time()

    # ------------------------------------------------------------------
    # 1. Environment
    # ------------------------------------------------------------------
    setup_environment()

    # ------------------------------------------------------------------
    # 2. Load Phase 1 artefacts
    # ------------------------------------------------------------------
    df, label_matrix, label_names = load_phase1_artifacts()
    song_ids: list[str] = load_text_list(config.SONG_IDS_TXT)

    if len(song_ids) != label_matrix.shape[0]:
        logger.warning(
            "Song ID count (%d) differs from label matrix rows (%d). "
            "Using integer indices as fallback.",
            len(song_ids),
            label_matrix.shape[0],
        )
        song_ids = [str(i) for i in range(label_matrix.shape[0])]

    # ------------------------------------------------------------------
    # 3. Audio DL embeddings  [DEFERRED]
    # ------------------------------------------------------------------
    audio_embeddings: np.ndarray = extract_audio_dl_embeddings(df)

    # ------------------------------------------------------------------
    # 4. Text DL embeddings  [DEFERRED]
    # ------------------------------------------------------------------
    text_embeddings: np.ndarray = extract_text_dl_embeddings(df)

    # ------------------------------------------------------------------
    # 5. Fuse embeddings (late fusion — no PCA)
    # ------------------------------------------------------------------
    fused_embeddings: np.ndarray = fuse_dl_embeddings(audio_embeddings, text_embeddings)

    # ------------------------------------------------------------------
    # 6. Train / test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test, ids_train, ids_test = split_data(
        fused_embeddings, label_matrix, song_ids
    )

    # Persist split song IDs for reproducibility
    save_text_list(ids_train, config.FEATURES_DIR / "dl_train_ids.txt")
    save_text_list(ids_test, config.FEATURES_DIR / "dl_test_ids.txt")
    logger.info(
        "Split IDs saved — train: %d, test: %d", len(ids_train), len(ids_test)
    )

    # ------------------------------------------------------------------
    # 7. Traditional classifiers on DL features  [DEFERRED]
    # ------------------------------------------------------------------
    trad_results: dict[str, Any] = run_traditional_classifiers_on_dl(
        X_train, X_test, y_train, y_test, label_names
    )

    # ------------------------------------------------------------------
    # 8. MLP deep classifier  [DEFERRED]
    # ------------------------------------------------------------------
    mlp_results: dict[str, Any] = run_mlp_dl_pipeline(
        X_train, X_test, y_train, y_test, label_names
    )

    # ------------------------------------------------------------------
    # 9. Aggregate and persist all results
    # ------------------------------------------------------------------
    all_results: dict[str, Any] = {**trad_results, **mlp_results}
    compare_and_save_results(all_results, label_names)

    elapsed_total = time.time() - pipeline_start
    log_section(logger, f"Phase 3.4 DL Pipeline Complete — {elapsed_total / 60:.1f} min")
    logger.info(
        "All DL pipeline artefacts saved under: %s", config.OUTPUT_DIR
    )


# ===========================================================================
# Guard — prevents accidental execution during baseline phase
# ===========================================================================

if __name__ == "__main__":
    # DEFERRED: not executed during baseline phase
    logger.warning(
        "train_deep.py is marked DEFERRED.  "
        "This pipeline requires CUDA GPU, transformers, and torchaudio.  "
        "Remove this guard only when running in a full DL environment."
    )
    _ALLOW_DL_EXECUTION: bool = False  # Flip to True only in DL environment

    if _ALLOW_DL_EXECUTION:
        main()
    else:
        logger.error(
            "Execution blocked by DEFERRED guard.  "
            "Set _ALLOW_DL_EXECUTION = True to run the DL pipeline."
        )
        sys.exit(0)
