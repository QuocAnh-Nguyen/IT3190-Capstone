"""Phase 2.4 — Pre-trained Text Embeddings (BERT) for Multi-Modal Music Genre Classification.

WARNING: DEFERRED EXECUTION
============================
This module is implemented but NOT executed during the baseline phase.
It requires a GPU-capable environment and the ``transformers`` library.
To activate: set USE_DEEP_TEXT=True in the pipeline and install dependencies.

    pip install transformers torch

Overview
--------
Extracts contextual sentence-level embeddings from song lyrics using a pre-trained
BERT model (default: ``bert-base-uncased``).  The [CLS] token representation from
the final hidden layer is used as the 768-dimensional feature vector for each song.

Key functions
~~~~~~~~~~~~~
* ``load_text_encoder``          — Download / cache BERT model + tokenizer.
* ``preprocess_lyrics_for_bert`` — Clean raw lyrics text before tokenisation.
* ``extract_embedding``          — Single-sample forward pass → numpy vector.
* ``extract_all_embeddings``     — Batch pipeline with checkpointing and resume logic.

Output artefact
~~~~~~~~~~~~~~~
A parquet file written to ``FEATURES_DIR / "bert_text_embeddings.parquet"`` with
one row per song, 768 columns (``dim_0`` … ``dim_767``), indexed by ``song_id``.

Dependencies (not installed in baseline environment)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``torch``          >= 1.12
* ``transformers``   >= 4.28
* ``tqdm``           >= 4.0

Phase / milestone
~~~~~~~~~~~~~~~~~
Phase 2 — Feature Engineering, Milestone 2.4 (Deep Text Features).
"""
# DEFERRED: not executed during baseline phase

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src2.config import FEATURES_DIR
from src2.utils.io_utils import load_pickle, save_pickle

logger = logging.getLogger("music_genre")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_NAME: str = "bert-base-uncased"
_EMBEDDING_DIM: int = 768           # bert-base hidden size
_CHECKPOINT_SUFFIX: str = "_ckpt.pkl"  # intermediate checkpoint file suffix

# ---------------------------------------------------------------------------
# Helpers — lazy import guard
# ---------------------------------------------------------------------------


def _require_transformers() -> None:
    """Raise a clear ``ImportError`` if ``transformers`` / ``torch`` are absent.

    Parameters
    ----------
    (none)

    Returns
    -------
    None
    """
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'transformers' and 'torch' libraries are required for Phase 2.4 "
            "(deep text embeddings) but are not installed in the current environment.\n"
            "Install them with:\n\n"
            "    pip install transformers torch\n\n"
            "Then set USE_DEEP_TEXT=True in the pipeline configuration."
        ) from exc


# ---------------------------------------------------------------------------
# 1. Model loading
# ---------------------------------------------------------------------------


def load_text_encoder(
    model_name: str = _DEFAULT_MODEL_NAME,
) -> Tuple[object, object]:
    """Load a pre-trained BERT model and its tokenizer from HuggingFace Hub.

    The model is placed in evaluation mode and moved to GPU if one is
    available; otherwise it runs on CPU (slow for large corpora).

    Parameters
    ----------
    model_name : str, optional
        HuggingFace model identifier.  Defaults to ``"bert-base-uncased"``.

    Returns
    -------
    tuple[BertModel, BertTokenizerFast]
        ``(model, tokenizer)`` ready for inference.

    Raises
    ------
    ImportError
        If ``transformers`` or ``torch`` are not installed.
    RuntimeError
        If the model cannot be downloaded or loaded.
    """
    _require_transformers()

    import torch
    from transformers import AutoModel, AutoTokenizer  # type: ignore[import]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "Loading text encoder '%s' onto device '%s' …", model_name, device
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        model.to(device)
        logger.info(
            "Text encoder loaded successfully (%d parameters).",
            sum(p.numel() for p in model.parameters()),
        )
    except Exception as exc:
        logger.exception(
            "Failed to load text encoder '%s': %s", model_name, exc
        )
        raise RuntimeError(
            f"Could not load HuggingFace model '{model_name}'."
        ) from exc

    return model, tokenizer


# ---------------------------------------------------------------------------
# 2. Lyrics preprocessing
# ---------------------------------------------------------------------------

# Regex patterns compiled once at import time for reuse
_SECTION_HEADER_RE: re.Pattern[str] = re.compile(
    r"\[.*?\]",  # matches [Verse 1], [Chorus], [Bridge], [Intro], etc.
    flags=re.IGNORECASE,
)
_EXTRA_WHITESPACE_RE: re.Pattern[str] = re.compile(r"\s{2,}")


def preprocess_lyrics_for_bert(
    text: str,
    max_length: int = 512,
) -> str:
    """Clean raw lyrics text to make it suitable for BERT tokenisation.

    Processing steps (in order):

    1. Cast to string and strip leading/trailing whitespace.
    2. Remove section-header annotations such as ``[Verse 1]``, ``[Chorus]``.
    3. Collapse multiple consecutive whitespace characters (spaces, tabs,
       newlines) into a single space.
    4. Truncate to a rough ``max_length``-token budget using a word-count
       heuristic (1 token ≈ 0.75 words for English).

    Parameters
    ----------
    text : str
        Raw lyrics string, potentially containing annotations and artefacts.
    max_length : int, optional
        Maximum BERT token budget (including [CLS] and [SEP]).  Defaults to
        512, which is the hard limit for ``bert-base-uncased``.

    Returns
    -------
    str
        Cleaned, truncated lyrics string safe to pass to a BERT tokenizer.
    """
    # Step 1 — coerce and strip
    text = str(text).strip()

    # Step 2 — remove section headers
    text = _SECTION_HEADER_RE.sub("", text)

    # Step 3 — normalise whitespace
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = _EXTRA_WHITESPACE_RE.sub(" ", text).strip()

    # Step 4 — token-budget truncation (heuristic: ~0.75 words per token)
    # Reserve 2 tokens for special tokens [CLS] / [SEP].
    effective_token_budget = max_length - 2
    word_budget = int(effective_token_budget / 0.75)
    words = text.split()
    if len(words) > word_budget:
        logger.debug(
            "Lyrics truncated from %d to %d words (max_length=%d).",
            len(words),
            word_budget,
            max_length,
        )
        text = " ".join(words[:word_budget])

    return text


# ---------------------------------------------------------------------------
# 3. Single embedding
# ---------------------------------------------------------------------------


def extract_embedding(
    text: str,
    model: object,
    tokenizer: object,
    device: object,
    max_length: int = 512,
) -> np.ndarray:
    """Compute the BERT [CLS] token embedding for a single lyrics string.

    Parameters
    ----------
    text : str
        Preprocessed lyrics text (output of :func:`preprocess_lyrics_for_bert`).
    model : transformers.PreTrainedModel
        A loaded BERT-style model in eval mode.
    tokenizer : transformers.PreTrainedTokenizer
        The matching tokenizer instance.
    device : torch.device
        Target device (``"cpu"`` or ``"cuda"``).
    max_length : int, optional
        Hard token-length cap passed to the tokenizer.  Defaults to 512.

    Returns
    -------
    np.ndarray
        1-D float32 array of shape ``(768,)`` representing the [CLS] embedding.

    Raises
    ------
    RuntimeError
        Propagated if the forward pass fails unexpectedly.
    """
    _require_transformers()
    import torch  # noqa: F401 — already checked by _require_transformers

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    # Move all input tensors to the model's device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # last_hidden_state shape: (batch=1, seq_len, hidden=768)
    cls_vector: np.ndarray = (
        outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
    )
    return cls_vector.astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Batch extraction with checkpointing
# ---------------------------------------------------------------------------


def extract_all_embeddings(
    song_ids: list[str],
    lyrics_texts: list[str],
    model_name: str = _DEFAULT_MODEL_NAME,
    batch_size: int = 32,
    checkpoint_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Extract BERT [CLS] embeddings for an entire corpus with checkpointing.

    The function:

    * Loads the model and tokenizer once via :func:`load_text_encoder`.
    * Processes the corpus in batches of size ``batch_size`` with a ``tqdm``
      progress bar.
    * Saves an intermediate checkpoint (pickle dict) after every batch.
    * Resumes from an existing checkpoint if ``checkpoint_path`` already
      contains results, skipping already-processed songs.
    * Logs and skips individual songs that raise an exception during the
      forward pass (they receive a zero vector so the row is still present).

    Parameters
    ----------
    song_ids : list[str]
        Ordered list of unique song identifiers.
    lyrics_texts : list[str]
        Corresponding lyrics strings (same length and order as ``song_ids``).
    model_name : str, optional
        HuggingFace model identifier.  Defaults to ``"bert-base-uncased"``.
    batch_size : int, optional
        Number of songs processed per iteration.  Defaults to 32.
    checkpoint_path : Path or None, optional
        Path to a pickle file used for incremental checkpointing.  If
        ``None``, defaults to
        ``FEATURES_DIR / "<model_name>_ckpt.pkl"``.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``song_id`` with 768 columns named
        ``dim_0, dim_1, …, dim_767``.  Songs that failed extraction have
        zero-valued rows.

    Raises
    ------
    ImportError
        If ``transformers`` / ``torch`` are not installed.
    ValueError
        If ``song_ids`` and ``lyrics_texts`` differ in length.
    """
    # DEFERRED: not executed during baseline phase

    _require_transformers()
    import torch  # noqa: F401
    from tqdm import tqdm  # type: ignore[import]

    if len(song_ids) != len(lyrics_texts):
        raise ValueError(
            f"song_ids ({len(song_ids)}) and lyrics_texts ({len(lyrics_texts)}) "
            "must have the same length."
        )

    # --- resolve checkpoint path ---------------------------------------------
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    if checkpoint_path is None:
        safe_model_name = model_name.replace("/", "_")
        checkpoint_path = FEATURES_DIR / f"{safe_model_name}{_CHECKPOINT_SUFFIX}"

    # --- resume from checkpoint if available ---------------------------------
    results: dict[str, np.ndarray] = {}
    if checkpoint_path.exists():
        try:
            results = load_pickle(checkpoint_path)
            logger.info(
                "Resumed checkpoint from '%s' — %d/%d songs already processed.",
                checkpoint_path,
                len(results),
                len(song_ids),
            )
        except Exception as exc:
            logger.warning(
                "Could not load checkpoint '%s' (%s); starting from scratch.",
                checkpoint_path,
                exc,
            )
            results = {}

    # --- build list of remaining items ---------------------------------------
    remaining_ids: list[str] = []
    remaining_texts: list[str] = []
    for sid, txt in zip(song_ids, lyrics_texts):
        if sid not in results:
            remaining_ids.append(sid)
            remaining_texts.append(txt)

    logger.info(
        "Extracting BERT embeddings: %d remaining / %d total  (batch_size=%d, model=%s).",
        len(remaining_ids),
        len(song_ids),
        batch_size,
        model_name,
    )

    if not remaining_ids:
        logger.info("All songs already processed — skipping model loading.")
    else:
        # --- load model once -------------------------------------------------
        model, tokenizer = load_text_encoder(model_name=model_name)
        # Retrieve device from model parameters directly
        device = next(model.parameters()).device  # type: ignore[union-attr]

        # --- batch loop -------------------------------------------------------
        n = len(remaining_ids)
        n_batches = (n + batch_size - 1) // batch_size
        t0 = time.monotonic()

        for batch_idx in tqdm(
            range(n_batches),
            desc="BERT embedding batches",
            unit="batch",
            dynamic_ncols=True,
        ):
            start = batch_idx * batch_size
            end = min(start + batch_size, n)
            batch_ids = remaining_ids[start:end]
            batch_texts = remaining_texts[start:end]

            for sid, raw_text in zip(batch_ids, batch_texts):
                try:
                    cleaned = preprocess_lyrics_for_bert(raw_text, max_length=512)
                    embedding = extract_embedding(
                        text=cleaned,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        max_length=512,
                    )
                    results[sid] = embedding
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning(
                        "Skipping song '%s' — error during embedding extraction: %s",
                        sid,
                        exc,
                    )
                    # Zero vector so the song still appears in the final DataFrame
                    results[sid] = np.zeros(_EMBEDDING_DIM, dtype=np.float32)

            # --- checkpoint after every batch --------------------------------
            try:
                save_pickle(results, checkpoint_path)
                logger.debug(
                    "Checkpoint saved: %d/%d songs → '%s'.",
                    len(results),
                    len(song_ids),
                    checkpoint_path,
                )
            except Exception as exc:
                logger.warning(
                    "Checkpoint save failed after batch %d/%d: %s",
                    batch_idx + 1,
                    n_batches,
                    exc,
                )

        elapsed = time.monotonic() - t0
        logger.info(
            "Batch extraction complete — %d songs in %.1f s (%.3f s/song).",
            len(remaining_ids),
            elapsed,
            elapsed / max(len(remaining_ids), 1),
        )

    # --- assemble DataFrame in original input order --------------------------
    col_names = [f"dim_{i}" for i in range(_EMBEDDING_DIM)]
    rows: list[np.ndarray] = []
    ordered_ids: list[str] = []

    for sid in song_ids:
        vec = results.get(sid)
        if vec is None:
            logger.warning(
                "No embedding found for song '%s'; filling row with zeros.", sid
            )
            vec = np.zeros(_EMBEDDING_DIM, dtype=np.float32)
        rows.append(vec)
        ordered_ids.append(sid)

    embedding_matrix = np.stack(rows, axis=0)  # shape: (N, 768)
    df = pd.DataFrame(embedding_matrix, index=ordered_ids, columns=col_names)
    df.index.name = "song_id"

    # --- persist final artefact ----------------------------------------------
    output_path = FEATURES_DIR / "bert_text_embeddings.parquet"
    try:
        df.to_parquet(output_path)
        logger.info(
            "BERT text embeddings saved → '%s'  shape=%s.",
            output_path,
            df.shape,
        )
    except Exception as exc:
        logger.error(
            "Failed to save BERT embeddings parquet to '%s': %s", output_path, exc
        )

    return df
