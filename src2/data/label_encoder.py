"""Phase 1.4 — Multi-Label Encoding & Class Imbalance Analysis.

This module transforms the raw ``genres`` column (semicolon-delimited strings)
of a cleaned track DataFrame into a binary label matrix suitable for
multi-label classification training.

Responsibilities
----------------
1. **Parse** the ``;``-delimited ``genres`` column into per-track genre lists,
   normalising whitespace and casing.
2. **Filter** genre labels whose track count falls below ``min_count``
   (configurable via :data:`src2.config.MIN_GENRE_COUNT`).  Tracks that have
   *no* remaining labels after filtering are dropped entirely so the label
   matrix has no all-zero rows.
3. **Binarize** the retained label lists using
   :class:`sklearn.preprocessing.MultiLabelBinarizer`, producing an
   ``(N, C)`` boolean/int8 ``numpy`` array ``Y``.
4. **Analyse** class distribution: log per-genre track counts, identify the
   most and least common genres, and flag any genres with severe imbalance.
5. **Compute** per-class positive weights for use as ``pos_weight`` in
   :class:`torch.nn.BCEWithLogitsLoss`.  The weight for class *c* is::

       w_c = (N - n_c) / n_c

   where ``N`` is the total number of tracks and ``n_c`` is the number of
   tracks positively labelled for class *c*.  Weights are clipped to
   ``[1.0, max_weight]`` to prevent numerical instability with very rare
   classes.
6. **Persist** artefacts:
   - ``Y`` matrix       → :data:`src2.config.LABEL_MATRIX_NPY`
   - label names        → :data:`src2.config.LABEL_NAMES_TXT`
   - song IDs           → :data:`src2.config.SONG_IDS_TXT`
   - fitted MLB object  → :data:`src2.config.MLB_PKL`

Public API
----------
``encode_labels(df, min_count)``
    Main entry-point.  Returns a 5-tuple:
    ``(Y, label_names, song_ids, mlb, class_weights)``

``load_encoder()``
    Re-hydrate a previously fitted :class:`~sklearn.preprocessing.MultiLabelBinarizer`
    from disk.

``build_class_weights(Y, max_weight)``
    Standalone helper: compute positive-class weights from a binary matrix.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from src2.config import (
    LABEL_MATRIX_NPY,
    LABEL_NAMES_TXT,
    MIN_GENRE_COUNT,
    MLB_PKL,
    SONG_IDS_TXT,
)
from src2.utils.io_utils import (
    load_pickle,
    save_npy,
    save_pickle,
    save_text_list,
)
from src2.utils.logging_utils import log_section

logger = logging.getLogger("music_genre")

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Maximum positive weight applied to any class.  Prevents extreme gradients
#: for labels that appear in only one or two tracks.
_MAX_POSITIVE_WEIGHT: float = 50.0

#: Number of top / bottom genres to print in the distribution summary.
_TOP_K_DISPLAY: int = 10


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_genre_column(
    genres_series: pd.Series,
) -> List[List[str]]:
    """Parse the semicolon-delimited ``genres`` column into lists.

    Each entry in the series is split on ``;``, stripped of surrounding
    whitespace, lower-cased, and deduplicated (preserving order of first
    occurrence).

    Parameters
    ----------
    genres_series : pd.Series
        Raw ``genres`` column from the cleaned dataset.  NaN values and empty
        strings should already have been removed by the data-cleaning step, but
        any remaining nulls are treated as empty lists.

    Returns
    -------
    list of list of str
        One inner list per track, each containing the genre label strings.
    """
    result: List[List[str]] = []
    for raw in genres_series:
        if pd.isna(raw) or str(raw).strip() == "":
            result.append([])
            continue
        seen: dict[str, None] = {}
        for token in str(raw).split(";"):
            label = token.strip().lower()
            if label and label not in seen:
                seen[label] = None
        result.append(list(seen.keys()))
    return result


def _count_labels(
    genre_lists: List[List[str]],
) -> pd.Series:
    """Count how many tracks each genre label appears in.

    Parameters
    ----------
    genre_lists : list of list of str
        Parsed genre lists (one per track).

    Returns
    -------
    pd.Series
        Index = genre label string, values = track count, sorted descending.
    """
    from collections import Counter

    counter: Counter[str] = Counter()
    for genres in genre_lists:
        counter.update(genres)
    return pd.Series(counter, dtype=int).sort_values(ascending=False)


def _filter_genres(
    genre_lists: List[List[str]],
    min_count: int,
    label_counts: pd.Series,
) -> Tuple[List[List[str]], List[str]]:
    """Remove genre labels below *min_count* from every track's label list.

    Tracks whose label list becomes empty after filtering are **excluded** from
    the return value; callers must track which original indices survive.

    Parameters
    ----------
    genre_lists : list of list of str
        Parsed genre lists before filtering.
    min_count : int
        Minimum number of tracks a label must appear in to be retained.
    label_counts : pd.Series
        Pre-computed per-label track counts (from :func:`_count_labels`).

    Returns
    -------
    filtered_lists : list of list of str
        Genre lists with rare labels removed; empty lists included (caller
        decides whether to drop the corresponding rows).
    retained_labels : list of str
        Sorted list of genre labels that passed the frequency threshold.
    """
    retained: set[str] = {
        label for label, cnt in label_counts.items() if cnt >= min_count
    }
    filtered: List[List[str]] = [
        [g for g in row if g in retained] for row in genre_lists
    ]
    retained_sorted = sorted(retained)
    logger.info(
        "Genre filter: %d unique labels -> %d retained (min_count=%d); "
        "%d labels dropped.",
        len(label_counts),
        len(retained_sorted),
        min_count,
        len(label_counts) - len(retained_sorted),
    )
    return filtered, retained_sorted


# ---------------------------------------------------------------------------
# Class distribution analysis
# ---------------------------------------------------------------------------


def _log_class_distribution(
    Y: np.ndarray,
    label_names: List[str],
) -> None:
    """Log detailed class distribution statistics for the binary label matrix.

    Parameters
    ----------
    Y : np.ndarray
        Binary label matrix of shape ``(N, C)``.
    label_names : list of str
        Names of the ``C`` classes in column order.

    Returns
    -------
    None
        All output goes to the ``music_genre`` logger at INFO level.
    """
    N, C = Y.shape
    counts = Y.sum(axis=0).astype(int)  # per-class positive count
    freq = counts / N * 100.0

    counts_series = pd.Series(counts, index=label_names)
    freq_series = pd.Series(freq, index=label_names)

    logger.info("-" * 70)
    logger.info("Class distribution  (N=%d tracks, C=%d genres)", N, C)
    logger.info("-" * 70)
    logger.info("  Tracks per genre -- summary statistics:")
    logger.info("    mean  : %.1f", counts.mean())
    logger.info("    median: %.1f", float(np.median(counts)))
    logger.info("    min   : %d  (%s)", counts.min(), counts_series.idxmin())
    logger.info("    max   : %d  (%s)", counts.max(), counts_series.idxmax())
    logger.info(
        "  Avg labels per track: %.2f",
        Y.sum(axis=1).mean(),
    )

    top_k = min(_TOP_K_DISPLAY, C)
    logger.info("  Top %d most common genres:", top_k)
    for genre, cnt, pct in zip(
        counts_series.nlargest(top_k).index,
        counts_series.nlargest(top_k).values,
        freq_series.loc[counts_series.nlargest(top_k).index].values,
    ):
        logger.info("    %-30s  %6d tracks  (%5.1f%%)", genre, cnt, pct)

    logger.info("  Bottom %d least common genres:", top_k)
    for genre, cnt, pct in zip(
        counts_series.nsmallest(top_k).index,
        counts_series.nsmallest(top_k).values,
        freq_series.loc[counts_series.nsmallest(top_k).index].values,
    ):
        logger.info("    %-30s  %6d tracks  (%5.1f%%)", genre, cnt, pct)

    # Imbalance severity
    imbalance_ratio = counts.max() / max(counts.min(), 1)
    logger.info(
        "  Max/min imbalance ratio: %.1fx -- %s",
        imbalance_ratio,
        "SEVERE (>50x)" if imbalance_ratio > 50 else "moderate",
    )
    logger.info("-" * 70)


# ---------------------------------------------------------------------------
# Positive-weight computation
# ---------------------------------------------------------------------------


def build_class_weights(
    Y: np.ndarray,
    max_weight: float = _MAX_POSITIVE_WEIGHT,
) -> np.ndarray:
    """Compute per-class positive weights for :class:`torch.nn.BCEWithLogitsLoss`.

    The weight for class *c* is defined as::

        w_c = (N - n_c) / n_c

    where ``N`` is the total number of samples and ``n_c`` is the number of
    positive samples for class *c*.  Weights are clipped to
    ``[1.0, max_weight]`` so that extremely rare labels do not dominate
    gradient updates.

    Parameters
    ----------
    Y : np.ndarray
        Binary label matrix of shape ``(N, C)`` with dtype int or bool.
    max_weight : float
        Upper bound for any single class weight.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(C,)`` containing the per-class weights.
    """
    N, C = Y.shape
    pos_counts = Y.sum(axis=0).astype(float)
    # Guard against division by zero for labels with no positive examples
    # (should not occur after filtering, but defensive coding is warranted).
    pos_counts = np.where(pos_counts == 0, 1.0, pos_counts)

    weights = (N - pos_counts) / pos_counts
    weights = np.clip(weights, 1.0, max_weight)

    logger.info(
        "Class weights  |  min: %.3f  max: %.3f  mean: %.3f",
        weights.min(),
        weights.max(),
        weights.mean(),
    )
    return weights.astype(np.float32)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _save_artefacts(
    Y: np.ndarray,
    label_names: List[str],
    song_ids: List[str],
    mlb: MultiLabelBinarizer,
) -> None:
    """Persist label matrix, label names, song IDs, and fitted MLB to disk.

    Parameters
    ----------
    Y : np.ndarray
        Binary label matrix.
    label_names : list of str
        Ordered genre label names (columns of ``Y``).
    song_ids : list of str
        Track identifiers (rows of ``Y``).
    mlb : MultiLabelBinarizer
        Fitted sklearn binarizer object.

    Returns
    -------
    None
    """
    try:
        save_npy(Y.astype(np.int8), LABEL_MATRIX_NPY)
        logger.info(
            "Label matrix saved  -> %s  shape=%s",
            LABEL_MATRIX_NPY,
            Y.shape,
        )
    except Exception:
        logger.exception("Failed to save label matrix to %s", LABEL_MATRIX_NPY)
        raise

    try:
        save_text_list(label_names, LABEL_NAMES_TXT)
        logger.info(
            "Label names saved   -> %s  (%d labels)",
            LABEL_NAMES_TXT,
            len(label_names),
        )
    except Exception:
        logger.exception("Failed to save label names to %s", LABEL_NAMES_TXT)
        raise

    try:
        save_text_list(song_ids, SONG_IDS_TXT)
        logger.info(
            "Song IDs saved      -> %s  (%d tracks)",
            SONG_IDS_TXT,
            len(song_ids),
        )
    except Exception:
        logger.exception("Failed to save song IDs to %s", SONG_IDS_TXT)
        raise

    try:
        save_pickle(mlb, MLB_PKL)
        logger.info("MLB saved           -> %s", MLB_PKL)
    except Exception:
        logger.exception(
            "Failed to save MultiLabelBinarizer to %s", MLB_PKL
        )
        raise


def load_encoder() -> MultiLabelBinarizer:
    """Load a previously fitted :class:`~sklearn.preprocessing.MultiLabelBinarizer`.

    Looks for the serialised object at :data:`src2.config.MLB_PKL`.

    Returns
    -------
    MultiLabelBinarizer
        Fitted binarizer that can be used to transform new label lists.

    Raises
    ------
    FileNotFoundError
        If the pickle file does not exist at the configured path.
    """
    if not MLB_PKL.exists():
        raise FileNotFoundError(
            f"MultiLabelBinarizer pickle not found at '{MLB_PKL}'. "
            "Run encode_labels() first to generate it."
        )
    mlb: MultiLabelBinarizer = load_pickle(MLB_PKL)
    logger.info(
        "Loaded MultiLabelBinarizer from '%s'  (%d classes)",
        MLB_PKL,
        len(mlb.classes_),
    )
    return mlb


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------


def encode_labels(
    df: pd.DataFrame,
    min_count: int = MIN_GENRE_COUNT,
) -> Tuple[np.ndarray, List[str], List[str], MultiLabelBinarizer, np.ndarray]:
    """Encode the ``genres`` column into a binary label matrix.

    This is the primary entry-point for Phase 1.4.  It orchestrates parsing,
    filtering, binarization, distribution analysis, weight computation, and
    artefact persistence.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset produced by
        :func:`src2.data.data_cleaner.clean_dataset`.
        Must contain at minimum:

        - ``"song_id"`` -- unique track identifier (falls back to the index
          if the column is absent).
        - ``"genres"`` -- semicolon-delimited genre string (non-null rows
          expected after cleaning).

    min_count : int, optional
        Minimum number of tracks a genre must appear in to be kept.
        Defaults to :data:`src2.config.MIN_GENRE_COUNT`.

    Returns
    -------
    Y : np.ndarray
        Binary label matrix of shape ``(N, C)``, dtype ``int8``.
    label_names : list of str
        Alphabetically sorted list of the ``C`` retained genre labels.
    song_ids : list of str
        Track identifiers corresponding to the ``N`` rows of ``Y``.
    mlb : MultiLabelBinarizer
        Fitted :class:`~sklearn.preprocessing.MultiLabelBinarizer`.
    class_weights : np.ndarray
        Float32 array of shape ``(C,)`` -- per-class positive weights
        suitable for use as ``pos_weight`` in
        :class:`torch.nn.BCEWithLogitsLoss`.

    Raises
    ------
    ValueError
        If the DataFrame is missing the ``genres`` column, or if no tracks
        survive the ``min_count`` filter.
    """
    log_section(
        logger,
        "Phase 1.4 -- Multi-Label Encoding & Class Imbalance Analysis",
    )

    # ------------------------------------------------------------------
    # Validate input
    # ------------------------------------------------------------------
    if "genres" not in df.columns:
        raise ValueError(
            "Input DataFrame must contain a 'genres' column. "
            f"Found columns: {df.columns.tolist()}"
        )

    # Resolve song ID column
    if "song_id" in df.columns:
        id_col = df["song_id"].astype(str)
    else:
        logger.warning(
            "'song_id' column not found -- using DataFrame index as "
            "track identifiers."
        )
        id_col = df.index.astype(str).to_series(index=df.index)

    logger.info("Input DataFrame: %d rows x %d columns", *df.shape)

    # ------------------------------------------------------------------
    # Step 1 -- Parse genre strings
    # ------------------------------------------------------------------
    logger.info("Step 1/5  Parsing genre strings ...")
    try:
        raw_genre_lists = _parse_genre_column(df["genres"])
    except Exception:
        logger.exception("Error while parsing the 'genres' column.")
        raise

    # ------------------------------------------------------------------
    # Step 2 -- Count labels and filter rare genres
    # ------------------------------------------------------------------
    logger.info(
        "Step 2/5  Counting label frequencies and filtering rare genres ..."
    )
    label_counts = _count_labels(raw_genre_lists)

    logger.info(
        "Before filtering: %d unique genre labels across %d tracks.",
        len(label_counts),
        len(raw_genre_lists),
    )

    filtered_genre_lists, retained_labels = _filter_genres(
        raw_genre_lists, min_count, label_counts
    )

    if not retained_labels:
        raise ValueError(
            f"No genre labels survived the min_count={min_count} filter.  "
            "Lower MIN_GENRE_COUNT in src2/config.py or provide more data."
        )

    # Drop tracks whose genre list is now empty after filtering
    non_empty_mask = [len(gl) > 0 for gl in filtered_genre_lists]
    n_dropped_tracks = sum(1 for m in non_empty_mask if not m)
    if n_dropped_tracks > 0:
        logger.warning(
            "%d tracks had ALL their genre labels removed by the frequency "
            "filter and will be excluded from the label matrix.",
            n_dropped_tracks,
        )

    kept_indices = [i for i, keep in enumerate(non_empty_mask) if keep]
    filtered_genre_lists = [filtered_genre_lists[i] for i in kept_indices]
    song_ids: List[str] = id_col.iloc[kept_indices].tolist()

    logger.info(
        "After filtering: %d tracks retained, %d genre labels.",
        len(song_ids),
        len(retained_labels),
    )

    if len(song_ids) == 0:
        raise ValueError(
            "No tracks remain after applying the rare-genre filter.  "
            "Check your dataset or reduce MIN_GENRE_COUNT."
        )

    # ------------------------------------------------------------------
    # Step 3 -- Fit & transform MultiLabelBinarizer
    # ------------------------------------------------------------------
    logger.info("Step 3/5  Fitting MultiLabelBinarizer ...")
    try:
        mlb = MultiLabelBinarizer(classes=retained_labels, sparse_output=False)
        Y_dense: np.ndarray = mlb.fit_transform(filtered_genre_lists)
    except Exception:
        logger.exception("MultiLabelBinarizer fit/transform failed.")
        raise

    Y = Y_dense.astype(np.int8)
    logger.info(
        "Label matrix: shape=%s  dtype=%s  non-zero elements=%d",
        Y.shape,
        Y.dtype,
        int(Y.sum()),
    )

    # Sanity check -- classes_ should match retained_labels exactly
    assert list(mlb.classes_) == retained_labels, (
        "MultiLabelBinarizer classes do not match retained_labels -- "
        "this is an internal consistency error."
    )

    # ------------------------------------------------------------------
    # Step 4 -- Analyse class distribution
    # ------------------------------------------------------------------
    logger.info("Step 4/5  Analysing class distribution ...")
    _log_class_distribution(Y, retained_labels)

    # ------------------------------------------------------------------
    # Step 5 -- Compute per-class positive weights
    # ------------------------------------------------------------------
    logger.info("Step 5/5  Computing per-class positive weights ...")
    class_weights = build_class_weights(Y, max_weight=_MAX_POSITIVE_WEIGHT)

    # ------------------------------------------------------------------
    # Persist artefacts
    # ------------------------------------------------------------------
    logger.info("Persisting artefacts to disk ...")
    _save_artefacts(Y, retained_labels, song_ids, mlb)

    log_section(logger, "Phase 1.4 complete", char="-")
    logger.info(
        "encode_labels() finished  |  tracks=%d  genres=%d  "
        "weight_min=%.3f  weight_max=%.3f",
        len(song_ids),
        len(retained_labels),
        float(class_weights.min()),
        float(class_weights.max()),
    )

    return Y, retained_labels, song_ids, mlb, class_weights
