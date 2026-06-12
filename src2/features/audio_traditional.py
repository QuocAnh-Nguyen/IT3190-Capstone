"""Traditional audio feature extraction for Multi-Modal Music Genre Classification.

Phase 2.1 — Traditional Audio Feature Extraction (EXECUTE)
===========================================================
This module extracts a fixed-length vector of ~209 hand-crafted audio features
from each 30-second MP3 preview using `librosa`.  The features are stored in a
flat, named dictionary so that every feature has a human-readable column name
in the resulting DataFrame.

Feature inventory
-----------------
| Feature group         | Config param | Statistic(s)   | # cols |
|-----------------------|--------------|----------------|--------|
| MFCCs                 | N_MFCC = 20  | mean + std     |  40    |
| Mel-Spectrogram       | N_MELS = 128 | mean per band  | 128    |
| Chroma                | N_CHROMA=12  | mean + std     |  24    |
| Spectral Contrast     | 7 bands      | mean per band  |   7    |
| Zero Crossing Rate    | —            | mean + std     |   2    |
| Spectral Centroid     | —            | mean + std     |   2    |
| Spectral Bandwidth    | —            | mean + std     |   2    |
| Spectral Rolloff      | —            | mean + std     |   2    |
| RMS Energy            | —            | mean + std     |   2    |
|                       |              | **Total**      | **209**|

Usage
-----
Typically called from the Phase 2.1 notebook or orchestration script:

    from src2.features.audio_traditional import extract_all_features
    from src2.config import AUDIO_PREVIEWS_DIR, FEATURES_DIR

    df_audio = extract_all_features(
        song_ids=song_ids,
        mp3_paths=mp3_paths,
        audio_dir=AUDIO_PREVIEWS_DIR,
        checkpoint_path=FEATURES_DIR / "audio_traditional_checkpoint.pkl",
    )
    df_audio.to_csv(FEATURES_DIR / "audio_traditional_features.csv")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from src2.config import (
    AUDIO_DURATION_SEC,
    AUDIO_PREVIEWS_DIR,
    AUDIO_SAMPLE_RATE,
    FEATURES_DIR,
    N_CHROMA,
    N_MELS,
    N_MFCC,
)
from src2.utils.io_utils import load_pickle, save_pickle

logger = logging.getLogger("music_genre")

# ---------------------------------------------------------------------------
# Single-file feature extraction
# ---------------------------------------------------------------------------


def extract_features_from_file(
    mp3_path: Path,
    sr: int = AUDIO_SAMPLE_RATE,
    duration: float = AUDIO_DURATION_SEC,
) -> dict[str, float]:
    """Extract all traditional audio features for a single MP3 file.

    Loads up to `duration` seconds of audio at sample rate `sr`, then
    computes all feature groups and returns them as a flat dict keyed by
    descriptive feature names.

    Parameters
    ----------
    mp3_path : Path
        Absolute or relative path to the MP3 preview file.
    sr : int, optional
        Target sample rate for loading. Defaults to ``AUDIO_SAMPLE_RATE``.
    duration : float, optional
        Maximum number of seconds to load. Defaults to ``AUDIO_DURATION_SEC``.

    Returns
    -------
    dict[str, float]
        Flat mapping of feature name -> scalar value.  The dict contains
        exactly 209 entries when all feature groups succeed.

    Raises
    ------
    Exception
        Re-raises any librosa / IO exception so the caller can decide how to
        handle it (e.g. skip the file and log the error).
    """
    features: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Load audio
    # ------------------------------------------------------------------
    y, _ = librosa.load(str(mp3_path), sr=sr, duration=duration, mono=True)

    # Pad with zeros if the clip is shorter than requested duration
    expected_samples = int(sr * duration)
    if len(y) < expected_samples:
        y = np.pad(y, (0, expected_samples - len(y)), mode="constant")

    # ------------------------------------------------------------------
    # Pre-compute shared transforms (avoid redundant FFTs)
    # ------------------------------------------------------------------
    stft = np.abs(librosa.stft(y))  # magnitude spectrogram

    # ------------------------------------------------------------------
    # 1. MFCCs  -> 40 features (mean + std per coefficient)
    # ------------------------------------------------------------------
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        features[f"mfcc_{i:02d}_mean"] = float(np.mean(mfccs[i]))
        features[f"mfcc_{i:02d}_std"] = float(np.std(mfccs[i]))

    # ------------------------------------------------------------------
    # 2. Mel-Spectrogram  -> 128 features (mean energy per mel band)
    # ------------------------------------------------------------------
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    # Convert to dB for more meaningful statistics
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    for i in range(N_MELS):
        features[f"mel_{i:03d}_mean"] = float(np.mean(mel_spec_db[i]))

    # ------------------------------------------------------------------
    # 3. Chroma Features  -> 24 features (mean + std per chroma bin)
    # ------------------------------------------------------------------
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr, n_chroma=N_CHROMA)
    for i in range(N_CHROMA):
        features[f"chroma_{i:02d}_mean"] = float(np.mean(chroma[i]))
        features[f"chroma_{i:02d}_std"] = float(np.std(chroma[i]))

    # ------------------------------------------------------------------
    # 4. Spectral Contrast  -> 7 features (mean per frequency band)
    # ------------------------------------------------------------------
    spec_contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
    n_bands = spec_contrast.shape[0]  # typically 7 (6 bands + 1 valley)
    for i in range(n_bands):
        features[f"spec_contrast_{i:02d}_mean"] = float(np.mean(spec_contrast[i]))

    # ------------------------------------------------------------------
    # 5. Zero Crossing Rate  -> 2 features
    # ------------------------------------------------------------------
    zcr = librosa.feature.zero_crossing_rate(y)
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    # ------------------------------------------------------------------
    # 6. Spectral Centroid  -> 2 features
    # ------------------------------------------------------------------
    centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)
    features["spectral_centroid_mean"] = float(np.mean(centroid))
    features["spectral_centroid_std"] = float(np.std(centroid))

    # ------------------------------------------------------------------
    # 7. Spectral Bandwidth  -> 2 features
    # ------------------------------------------------------------------
    bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sr)
    features["spectral_bandwidth_mean"] = float(np.mean(bandwidth))
    features["spectral_bandwidth_std"] = float(np.std(bandwidth))

    # ------------------------------------------------------------------
    # 8. Spectral Rolloff  -> 2 features
    # ------------------------------------------------------------------
    rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr)
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))
    features["spectral_rolloff_std"] = float(np.std(rolloff))

    # ------------------------------------------------------------------
    # 9. RMS Energy  -> 2 features
    # ------------------------------------------------------------------
    rms = librosa.feature.rms(y=y)
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))

    return features


# ---------------------------------------------------------------------------
# Batch extraction with checkpointing
# ---------------------------------------------------------------------------


def extract_all_features(
    song_ids: list[str],
    mp3_paths: list[Path | str],
    audio_dir: Path = AUDIO_PREVIEWS_DIR,
    checkpoint_path: Path = FEATURES_DIR / "audio_traditional_checkpoint.pkl",
    batch_size: int = 500,
    sr: int = AUDIO_SAMPLE_RATE,
    duration: float = AUDIO_DURATION_SEC,
) -> pd.DataFrame:
    """Batch-extract traditional audio features for all tracks with checkpointing.

    Iterates over every (song_id, mp3_path) pair, extracts features via
    :func:`extract_features_from_file`, and accumulates results.  Every
    ``batch_size`` successful extractions the partial results are persisted to
    ``checkpoint_path`` so that the job can be resumed after interruption
    without reprocessing already-completed files.

    Parameters
    ----------
    song_ids : list[str]
        Ordered list of song identifiers, one per track.
    mp3_paths : list[Path | str]
        Ordered list of MP3 file paths corresponding to ``song_ids``.  Each
        path may be absolute or relative to ``audio_dir``.
    audio_dir : Path, optional
        Base directory prepended to any relative ``mp3_paths``.
        Defaults to ``AUDIO_PREVIEWS_DIR``.
    checkpoint_path : Path, optional
        Location of the ``.pkl`` checkpoint file used for incremental saves
        and resume.  Defaults to ``FEATURES_DIR / audio_traditional_checkpoint.pkl``.
    batch_size : int, optional
        Number of files to process between consecutive checkpoint saves.
        Defaults to 500.
    sr : int, optional
        Sample rate passed through to :func:`extract_features_from_file`.
    duration : float, optional
        Clip duration (seconds) passed through to
        :func:`extract_features_from_file`.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``song_id`` with one column per extracted feature
        (~209 columns).  Rows for files that could not be processed are
        omitted; errors are logged at WARNING level.

    Notes
    -----
    * The checkpoint file stores a ``dict[str, dict[str, float]]`` mapping
      song_id -> feature_dict.
    * If ``checkpoint_path`` already exists on entry, completed song_ids are
      loaded and skipped, so only the remaining files are processed.
    * Files whose resolved path does not exist are skipped with a WARNING.
    """
    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if len(song_ids) != len(mp3_paths):
        raise ValueError(
            f"song_ids length ({len(song_ids)}) must equal "
            f"mp3_paths length ({len(mp3_paths)})."
        )

    # Ensure the checkpoint parent directory exists
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Resume from checkpoint if available
    # ------------------------------------------------------------------
    results: dict[str, dict[str, float]] = {}
    if checkpoint_path.exists():
        try:
            results = load_pickle(checkpoint_path)
            logger.info(
                "Resumed from checkpoint '%s' — %d songs already processed.",
                checkpoint_path,
                len(results),
            )
        except Exception as exc:
            logger.warning(
                "Failed to load checkpoint '%s' (%s). Starting from scratch.",
                checkpoint_path,
                exc,
            )
            results = {}

    already_done: set[str] = set(results.keys())
    remaining_pairs: list[tuple[str, Path | str]] = [
        (sid, mp)
        for sid, mp in zip(song_ids, mp3_paths)
        if sid not in already_done
    ]

    total = len(song_ids)
    skipped_count = len(already_done)
    error_count = 0
    processed_since_last_save = 0

    logger.info(
        "Starting audio feature extraction: %d total tracks, "
        "%d already done, %d remaining.",
        total,
        skipped_count,
        len(remaining_pairs),
    )

    # ------------------------------------------------------------------
    # Main extraction loop
    # ------------------------------------------------------------------
    with tqdm(
        remaining_pairs,
        total=len(remaining_pairs),
        desc="Extracting audio features",
        unit="track",
        dynamic_ncols=True,
    ) as pbar:
        for song_id, raw_path in pbar:
            # Resolve the file path
            mp3_path = Path(raw_path)
            if not mp3_path.is_absolute():
                mp3_path = audio_dir / mp3_path

            # Skip missing files
            if not mp3_path.exists():
                logger.warning(
                    "Audio file not found, skipping song_id=%s  path=%s",
                    song_id,
                    mp3_path,
                )
                error_count += 1
                pbar.set_postfix({"errors": error_count}, refresh=False)
                continue

            # Extract features — catch any librosa / IO error per file
            try:
                feat_dict = extract_features_from_file(mp3_path, sr=sr, duration=duration)
                results[song_id] = feat_dict
                processed_since_last_save += 1
            except Exception as exc:
                logger.warning(
                    "Feature extraction failed for song_id=%s  path=%s  error=%s",
                    song_id,
                    mp3_path,
                    exc,
                )
                error_count += 1
                pbar.set_postfix({"errors": error_count}, refresh=False)
                continue

            # ----------------------------------------------------------
            # Checkpoint save every `batch_size` successful extractions
            # ----------------------------------------------------------
            if processed_since_last_save >= batch_size:
                _save_checkpoint(results, checkpoint_path)
                processed_since_last_save = 0
                logger.debug(
                    "Checkpoint saved — %d total songs processed so far.",
                    len(results),
                )

    # ------------------------------------------------------------------
    # Final checkpoint save (flush any remainder)
    # ------------------------------------------------------------------
    if processed_since_last_save > 0:
        _save_checkpoint(results, checkpoint_path)
        logger.debug("Final checkpoint saved.")

    # ------------------------------------------------------------------
    # Build and return DataFrame
    # ------------------------------------------------------------------
    logger.info(
        "Extraction complete: %d successful, %d errors/skipped.",
        len(results),
        error_count,
    )

    if not results:
        logger.warning("No features were extracted. Returning empty DataFrame.")
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "song_id"
    df = df.sort_index()

    logger.info(
        "Audio feature DataFrame shape: %s  |  columns: %d",
        df.shape,
        df.shape[1],
    )
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _save_checkpoint(
    results: dict[str, dict[str, float]],
    path: Path,
) -> None:
    """Persist the current results dict to a pickle checkpoint file.

    Parameters
    ----------
    results : dict[str, dict[str, float]]
        Mapping of song_id -> feature dict accumulated so far.
    path : Path
        Destination path for the pickle file.  Parent directories are
        created automatically.
    """
    try:
        save_pickle(results, path)
    except Exception as exc:
        logger.error(
            "Checkpoint save failed at '%s': %s.  Progress will not be persisted.",
            path,
            exc,
        )


# ---------------------------------------------------------------------------
# Feature inventory helper (diagnostic use)
# ---------------------------------------------------------------------------


def get_feature_names() -> list[str]:
    """Return the ordered list of feature column names produced by this module.

    Useful for documentation, sanity-checking DataFrame columns, or building
    downstream feature selection masks without loading any audio.

    Returns
    -------
    list[str]
        List of ~209 feature name strings in the same order as they would
        appear in the DataFrame returned by :func:`extract_all_features`.
    """
    names: list[str] = []

    # MFCCs
    for i in range(N_MFCC):
        names.append(f"mfcc_{i:02d}_mean")
        names.append(f"mfcc_{i:02d}_std")

    # Mel-Spectrogram
    for i in range(N_MELS):
        names.append(f"mel_{i:03d}_mean")

    # Chroma
    for i in range(N_CHROMA):
        names.append(f"chroma_{i:02d}_mean")
        names.append(f"chroma_{i:02d}_std")

    # Spectral Contrast (7 bands by librosa default)
    _N_CONTRAST_BANDS = 7
    for i in range(_N_CONTRAST_BANDS):
        names.append(f"spec_contrast_{i:02d}_mean")

    # Scalar features
    names += [
        "zcr_mean",
        "zcr_std",
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "spectral_bandwidth_mean",
        "spectral_bandwidth_std",
        "spectral_rolloff_mean",
        "spectral_rolloff_std",
        "rms_mean",
        "rms_std",
    ]

    return names
