"""Phase 2.3 — Pre-trained Audio Embeddings via Audio Spectrogram Transformer (AST).

WARNING: DEFERRED EXECUTION
================================
This module is implemented but NOT executed during the baseline phase.
It requires a GPU-capable environment and the ``transformers`` and ``torchaudio``
libraries.  To activate: set USE_DEEP_AUDIO=True in the pipeline and install
dependencies:

    pip install transformers torchaudio

The default model used is:
    MIT/ast-finetuned-audioset-10-10-0.4593

which produces 768-dimensional [CLS] token embeddings from raw audio waveforms
and was fine-tuned on AudioSet for audio classification.

Overview
--------
* ``load_audio_encoder``   — download / cache and return (model, processor) pair.
* ``preprocess_audio``     — load an MP3 preview, resample, and normalise to mono.
* ``extract_embedding``    — single-file forward pass; returns 1-D numpy embedding.
* ``extract_all_embeddings`` — batch wrapper with tqdm, checkpointing, and per-file
                               error recovery; returns a song_id-indexed DataFrame.

Milestone: Phase 2 — Feature Engineering
Sub-task:  2.3 — Deep Audio Embeddings (DEFERRED)
"""

# DEFERRED: not executed during baseline phase

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src2.config import AUDIO_PREVIEWS_DIR, FEATURES_DIR
from src2.utils.io_utils import load_pickle, save_pickle

log = logging.getLogger("music_genre")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_NAME: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
_DEFAULT_TARGET_SR: int = 16_000
_DEFAULT_BATCH_SIZE: int = 16

# Expected audio duration in seconds for AST (AudioSet clips).
# The processor pads/truncates to this length automatically.
_AST_MAX_LENGTH_SEC: float = 10.0


# ---------------------------------------------------------------------------
# 1. Model loading
# ---------------------------------------------------------------------------


def load_audio_encoder(
    model_name: str = _DEFAULT_MODEL_NAME,
) -> Tuple[object, object]:
    """Load the Audio Spectrogram Transformer (AST) model and feature processor.

    Downloads weights from HuggingFace Hub on the first call and caches them
    locally.  The model is set to evaluation mode and moved to GPU when a
    CUDA-capable device is available.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.  Defaults to the AudioSet-fine-tuned AST
        checkpoint ``MIT/ast-finetuned-audioset-10-10-0.4593``.

    Returns
    -------
    Tuple[ASTModel, ASTFeatureExtractor]
        ``(model, processor)`` where *model* is an ``ASTModel`` instance ready
        for inference and *processor* is the matching ``ASTFeatureExtractor``.

    Raises
    ------
    ImportError
        If ``transformers`` or ``torch`` is not installed.
    OSError
        If the model cannot be downloaded (no internet / bad identifier).
    """
    try:
        import torch
        from transformers import ASTFeatureExtractor, ASTModel
    except ImportError as exc:
        log.error(
            "Cannot import 'transformers' or 'torch'. "
            "Install them with: pip install transformers torch"
        )
        raise ImportError(
            "Required packages 'transformers' and 'torch' are not installed."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Loading AST model '%s' on device '%s'.", model_name, device)

    try:
        processor = ASTFeatureExtractor.from_pretrained(model_name)
        model = ASTModel.from_pretrained(model_name)
    except OSError as exc:
        log.error(
            "Failed to load model '%s'. Check the model name and internet connectivity.",
            model_name,
        )
        raise

    model.eval()
    model.to(device)

    log.info(
        "AST model loaded successfully. Hidden size: %d.",
        model.config.hidden_size,
    )
    return model, processor


# ---------------------------------------------------------------------------
# 2. Audio preprocessing
# ---------------------------------------------------------------------------


def preprocess_audio(
    mp3_path: Path,
    target_sr: int = _DEFAULT_TARGET_SR,
) -> np.ndarray:
    """Load an MP3 audio preview, resample, and return a mono waveform array.

    Converts stereo to mono by averaging channels, resamples to *target_sr*
    using librosa (which uses a high-quality resampler internally), and
    returns a 1-D float32 NumPy array normalised to the range ``[-1, 1]``.

    Parameters
    ----------
    mp3_path : Path
        Absolute or relative path to the ``.mp3`` (or any librosa-compatible
        audio) file.
    target_sr : int
        Target sample rate in Hz.  The AST processor expects 16 000 Hz.

    Returns
    -------
    np.ndarray
        Mono waveform of shape ``(n_samples,)`` with dtype ``float32``.

    Raises
    ------
    FileNotFoundError
        If *mp3_path* does not point to an existing file.
    RuntimeError
        If librosa fails to decode the audio (corrupt file, unsupported codec).
    """
    try:
        import librosa
    except ImportError as exc:
        log.error(
            "Cannot import 'librosa'. Install it with: pip install librosa"
        )
        raise ImportError("Required package 'librosa' is not installed.") from exc

    mp3_path = Path(mp3_path)
    if not mp3_path.exists():
        raise FileNotFoundError(f"Audio file not found: {mp3_path}")

    try:
        # mono=True collapses stereo; res_type uses soxr for quality.
        waveform, _ = librosa.load(mp3_path, sr=target_sr, mono=True, res_type="soxr_hq")
    except Exception as exc:
        raise RuntimeError(
            f"librosa failed to decode audio file '{mp3_path}': {exc}"
        ) from exc

    # Ensure float32 (librosa already returns float32 but be explicit).
    waveform = waveform.astype(np.float32)

    # Peak-normalise to guard against clipping artefacts after resampling.
    peak = np.abs(waveform).max()
    if peak > 0.0:
        waveform = waveform / peak

    log.debug(
        "Preprocessed '%s': %d samples at %d Hz.",
        mp3_path.name,
        len(waveform),
        target_sr,
    )
    return waveform


# ---------------------------------------------------------------------------
# 3. Single embedding extraction
# ---------------------------------------------------------------------------


def extract_embedding(
    waveform: np.ndarray,
    model: object,
    processor: object,
    device: object,
) -> np.ndarray:
    """Extract a fixed-size embedding from a raw audio waveform using AST.

    Runs the HuggingFace feature extractor to create log-mel spectrograms,
    performs a forward pass through the AST encoder under ``torch.no_grad()``,
    and returns the [CLS] token hidden state as a 1-D numpy vector.

    Parameters
    ----------
    waveform : np.ndarray
        Mono float32 waveform array of shape ``(n_samples,)`` at 16 000 Hz.
    model : ASTModel
        AST encoder returned by :func:`load_audio_encoder`.
    processor : ASTFeatureExtractor
        Matching feature extractor returned by :func:`load_audio_encoder`.
    device : torch.device
        Target device (``cpu`` or ``cuda``).

    Returns
    -------
    np.ndarray
        1-D embedding vector of shape ``(embedding_dim,)`` — typically 768 for
        the default AST checkpoint.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Required package 'torch' is not installed.") from exc

    # The processor expects a list of waveform arrays and the sampling rate.
    inputs = processor(
        [waveform],
        sampling_rate=_DEFAULT_TARGET_SR,
        return_tensors="pt",
        padding=True,
    )

    # Move all input tensors to the target device.
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # ``last_hidden_state`` has shape (batch, seq_len, hidden_size).
    # The first token [CLS] aggregates global audio context.
    cls_embedding: np.ndarray = (
        outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
    )

    log.debug("Extracted embedding of shape %s.", cls_embedding.shape)
    return cls_embedding


# ---------------------------------------------------------------------------
# 4. Batch extraction
# ---------------------------------------------------------------------------


def extract_all_embeddings(  # DEFERRED: not executed during baseline phase
    song_ids: list[str],
    mp3_paths: list[Path],
    audio_dir: Optional[Path] = None,
    model_name: str = _DEFAULT_MODEL_NAME,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    checkpoint_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Extract AST embeddings for every song in *song_ids* and return a DataFrame.

    Processes audio files in mini-batches, saves a checkpoint pickle after each
    batch to support resuming interrupted runs, and logs (but does not re-raise)
    per-file errors so a single corrupt file does not abort the entire run.

    Parameters
    ----------
    song_ids : list[str]
        Ordered list of song identifiers that form the index of the output.
    mp3_paths : list[Path]
        Parallel list of audio file paths, one per entry in *song_ids*.
        May be relative; ``audio_dir`` is prepended when supplied.
    audio_dir : Path, optional
        Base directory prepended to relative paths in *mp3_paths*.  Defaults to
        :data:`src2.config.AUDIO_PREVIEWS_DIR`.
    model_name : str
        HuggingFace model identifier forwarded to :func:`load_audio_encoder`.
    batch_size : int
        Number of files to process between checkpoint saves.
    checkpoint_path : Path, optional
        Path to a ``.pkl`` checkpoint file.  When the file already exists, the
        embeddings accumulated so far are loaded and only the missing songs are
        processed.  Defaults to ``FEATURES_DIR / "ast_embeddings_ckpt.pkl"``.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by *song_id* where each row contains the 768-D (or
        model-specific) embedding as separate columns named ``ast_0``,
        ``ast_1``, ...  Songs that could not be processed are filled with
        ``NaN``.

    Raises
    ------
    ImportError
        If required packages (``torch``, ``transformers``, ``librosa``) are not
        installed.
    ValueError
        If *song_ids* and *mp3_paths* have different lengths.
    """
    try:
        import torch
        from tqdm.auto import tqdm
    except ImportError as exc:
        log.error(
            "Cannot import required packages. "
            "Install with: pip install torch transformers librosa tqdm"
        )
        raise ImportError(
            "One or more required packages are missing: torch, transformers, librosa, tqdm"
        ) from exc

    if len(song_ids) != len(mp3_paths):
        raise ValueError(
            f"song_ids length ({len(song_ids)}) != mp3_paths length ({len(mp3_paths)})."
        )

    # Resolve directories and checkpoint path.
    base_dir: Path = Path(audio_dir) if audio_dir is not None else AUDIO_PREVIEWS_DIR
    ckpt_path: Path = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else FEATURES_DIR / "ast_embeddings_ckpt.pkl"
    )

    # -----------------------------------------------------------------
    # Resume from checkpoint if it exists.
    # -----------------------------------------------------------------
    embeddings_cache: dict[str, np.ndarray] = {}
    if ckpt_path.exists():
        log.info("Resuming from checkpoint: %s", ckpt_path)
        try:
            embeddings_cache = load_pickle(ckpt_path)
            log.info(
                "Loaded %d cached embeddings from checkpoint.", len(embeddings_cache)
            )
        except Exception as exc:
            log.warning(
                "Failed to load checkpoint '%s' (%s). Starting fresh.", ckpt_path, exc
            )
            embeddings_cache = {}

    # Determine which songs still need processing.
    pending_indices = [
        i for i, sid in enumerate(song_ids) if sid not in embeddings_cache
    ]
    log.info(
        "%d / %d songs require embedding extraction.",
        len(pending_indices),
        len(song_ids),
    )

    if not pending_indices:
        log.info("All embeddings already cached; skipping model loading.")
    else:
        # -----------------------------------------------------------------
        # Load model once for the entire run.
        # -----------------------------------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, processor = load_audio_encoder(model_name=model_name)

        # -----------------------------------------------------------------
        # Iterate in mini-batches.
        # -----------------------------------------------------------------
        batch_errors: int = 0

        with tqdm(
            total=len(pending_indices),
            desc="Extracting AST embeddings",
            unit="file",
            dynamic_ncols=True,
        ) as pbar:
            for batch_start in range(0, len(pending_indices), batch_size):
                batch_slice = pending_indices[batch_start : batch_start + batch_size]

                for idx in batch_slice:
                    sid = song_ids[idx]
                    raw_path = Path(mp3_paths[idx])
                    audio_path = (
                        raw_path if raw_path.is_absolute() else base_dir / raw_path
                    )

                    try:
                        waveform = preprocess_audio(
                            audio_path, target_sr=_DEFAULT_TARGET_SR
                        )
                        embedding = extract_embedding(waveform, model, processor, device)
                        embeddings_cache[sid] = embedding
                    except FileNotFoundError:
                        log.warning(
                            "Audio file not found for song_id '%s': %s. Skipping.",
                            sid,
                            audio_path,
                        )
                        embeddings_cache[sid] = np.full(
                            model.config.hidden_size, fill_value=np.nan, dtype=np.float32
                        )
                        batch_errors += 1
                    except Exception as exc:
                        log.warning(
                            "Unexpected error processing song_id '%s' ('%s'): %s. Skipping.",
                            sid,
                            audio_path,
                            exc,
                        )
                        embeddings_cache[sid] = np.full(
                            model.config.hidden_size, fill_value=np.nan, dtype=np.float32
                        )
                        batch_errors += 1

                    pbar.update(1)

                # -----------------------------------------------------------------
                # Save checkpoint after every batch.
                # -----------------------------------------------------------------
                try:
                    save_pickle(embeddings_cache, ckpt_path)
                    log.debug(
                        "Checkpoint saved (%d embeddings) -> %s",
                        len(embeddings_cache),
                        ckpt_path,
                    )
                except Exception as exc:
                    log.warning("Failed to write checkpoint: %s", exc)

        log.info(
            "Embedding extraction complete. %d errors encountered out of %d files.",
            batch_errors,
            len(pending_indices),
        )

    # -----------------------------------------------------------------
    # Assemble DataFrame from the cache, preserving song_ids order.
    # -----------------------------------------------------------------
    rows: list[np.ndarray] = []
    for sid in song_ids:
        if sid in embeddings_cache:
            rows.append(embeddings_cache[sid])
        else:
            # Fallback: NaN row (should not happen if logic above is correct).
            # Determine embedding_dim from any valid cached entry.
            sample = next(
                (v for v in embeddings_cache.values() if v is not None), None
            )
            dim = sample.shape[0] if sample is not None else 768
            rows.append(np.full(dim, fill_value=np.nan, dtype=np.float32))
            log.warning("song_id '%s' missing from cache; filled with NaN.", sid)

    embedding_matrix = np.vstack(rows)  # shape: (n_songs, embedding_dim)
    n_dim = embedding_matrix.shape[1]
    col_names = [f"ast_{i}" for i in range(n_dim)]

    df = pd.DataFrame(embedding_matrix, index=song_ids, columns=col_names)
    df.index.name = "song_id"

    # Persist the final DataFrame alongside the checkpoint.
    output_path: Path = FEATURES_DIR / "ast_embeddings.pkl"
    try:
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        save_pickle(df, output_path)
        log.info(
            "AST embedding DataFrame saved -> %s  shape=%s", output_path, df.shape
        )
    except Exception as exc:
        log.warning("Could not persist final embedding DataFrame: %s", exc)

    return df
