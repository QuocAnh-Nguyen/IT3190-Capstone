"""Label converter — Multi-Label to Single-Label conversion.

Step 2 of the improve_plan: pivots from multi-label to single-label (multi-class)
classification. Given that 67.3% of tracks already have exactly 1 label and the
average is 1.39 labels/track, multi-label complexity is unwarranted for traditional ML.

Strategies
----------
* ``"rarest"`` (default, recommended) — for each track, pick the genre with the
  lowest overall frequency in the dataset.  Rationale: a track labelled
  "jazz + pop" is more meaningfully a *jazz* track since "pop" is ubiquitous.
  This naturally redistributes samples toward underrepresented classes.
* ``"first"`` — use the first genre in the semicolon-delimited list (preserves
  the dataset author's original intent).

Public API
----------
``convert_to_single_label(df, label_counts, strategy)``
    Convert a multi-label genres column to a single primary genre per track.

``compute_label_counts(df)``
    Compute per-genre track counts from the cleaned genres column.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger("music_genre")


def compute_label_counts(
    df: pd.DataFrame,
    genre_col: str = "genres",
) -> pd.Series:
    """Count how many tracks each genre label appears in (multi-label aware).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset with a semicolon-delimited ``genres`` column.
    genre_col : str
        Name of the genre column.

    Returns
    -------
    pd.Series
        Index = genre label string, values = track count, sorted descending.
    """
    from collections import Counter

    counter: Counter[str] = Counter()
    for raw in df[genre_col].dropna():
        for token in str(raw).split(";"):
            label = token.strip().lower()
            if label:
                counter[label] += 1

    return pd.Series(counter, dtype=int).sort_values(ascending=False)


def convert_to_single_label(
    df: pd.DataFrame,
    label_counts: Optional[pd.Series] = None,
    strategy: str = "rarest",
    genre_col: str = "genres",
    exclude_genres: Optional[list[str]] = None,
    min_count: int = 50,
) -> pd.DataFrame:
    """Convert multi-label genres to a single primary genre per track.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset containing ``song_id`` and semicolon-delimited
        ``genres`` columns.
    label_counts : pd.Series or None
        Pre-computed per-genre track counts.  If ``None``, they are computed
        from *df* via :func:`compute_label_counts`.
    strategy : str
        Selection strategy: ``"rarest"`` (pick least frequent genre) or
        ``"first"`` (pick first listed genre).
    genre_col : str
        Column name containing the semicolon-delimited genre string.
    exclude_genres : list[str] or None
        Genres to remove from all tracks before conversion.  Tracks whose only
        genres are all excluded are dropped.
    min_count : int
        Minimum number of tracks a genre must appear in to be retained.
        Genres below this threshold are added to the exclusion list.
        Defaults to 50 (matches MIN_GENRE_COUNT).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with two new columns:
        - ``primary_genre`` — the single selected genre string
        - ``num_original_genres`` — how many genres the track originally had
        Rows dropped: tracks with no genres after exclusion.
    """
    if exclude_genres is None:
        exclude_genres = []

    # Build exclusion list from min_count threshold + explicit exclusions
    if label_counts is None:
        label_counts = compute_label_counts(df, genre_col=genre_col)

    # Add genres below min_count to the exclusion list
    below_threshold = label_counts[label_counts < min_count].index.tolist()
    if below_threshold:
        logger.info(
            "Genres below min_count=%d (will be excluded): %s",
            min_count, below_threshold,
        )
    exclude_set = set(g.lower().strip() for g in exclude_genres)
    exclude_set.update(g.lower().strip() for g in below_threshold)

    # Build frequency map: lower count → "rarer" → preferred
    freq_map: dict[str, int] = {}
    if label_counts is not None:
        freq_map = {k.lower(): v for k, v in label_counts.items()}

    n_dropped_total = 0
    n_tracks_original = len(df)
    primary_labels: list[str] = []
    num_original: list[int] = []

    for raw in df[genre_col]:
        if pd.isna(raw) or str(raw).strip() == "":
            primary_labels.append("")
            num_original.append(0)
            continue

        # Parse genres, excluding unwanted ones
        parsed: list[str] = []
        for token in str(raw).split(";"):
            label = token.strip().lower()
            if label and label not in exclude_set:
                parsed.append(label)

        if not parsed:
            primary_labels.append("")
            num_original.append(0)
            n_dropped_total += 1
            continue

        if strategy == "rarest":
            # Pick the genre with the smallest overall frequency
            # Ties broken by alphabetical order (deterministic)
            selected = min(parsed, key=lambda g: (freq_map.get(g, 0), g))
        elif strategy == "first":
            selected = parsed[0]
        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Use 'rarest' or 'first'."
            )

        primary_labels.append(selected)
        num_original.append(len(parsed))

    result = df.copy()
    result["primary_genre"] = primary_labels
    result["num_original_genres"] = num_original

    # Drop tracks that lost all labels after exclusion
    if n_dropped_total > 0:
        logger.info(
            "Dropping %d tracks that had no remaining genres after exclusion of %s.",
            n_dropped_total,
            exclude_genres,
        )
        result = result[result["primary_genre"] != ""].copy()

    logger.info(
        "Single-label conversion complete: %d/%d tracks retained "
        "(%d dropped — no valid genre after exclusion).",
        len(result),
        n_tracks_original,
        n_dropped_total,
    )

    # Log new distribution
    dist = result["primary_genre"].value_counts()
    logger.info("Single-label distribution (top-10):")
    for genre, cnt in dist.head(10).items():
        logger.info("  %-30s  %6d tracks  (%5.1f%%)", genre, cnt,
                    100.0 * cnt / len(result))
    logger.info("  ...")
    for genre, cnt in dist.tail(5).items():
        logger.info("  %-30s  %6d tracks  (%5.1f%%)", genre, cnt,
                    100.0 * cnt / len(result))
    logger.info("Total classes: %d", len(dist))

    return result


def consolidate_genres(
    df: pd.DataFrame,
    consolidation_map: dict[str, list[str]],
    label_col: str = "primary_genre",
) -> pd.DataFrame:
    """Merge tail genres into consolidated parent categories (Step 3C).

    Each entry in ``consolidation_map`` defines a group of source genres that
    are merged into a single parent name.  Only genres still present after
    exclusion and single-label conversion are affected.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``primary_genre`` (or *label_col*) column with string
        genre labels.
    consolidation_map : dict[str, list[str]]
        Mapping ``{merged_name: [source_genre_1, source_genre_2, ...]}``.
    label_col : str
        Column name of the single-label genre.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``primary_genre`` values remapped.  The original
        column is copied to ``primary_genre_before_consolidation``.

    Raises
    ------
    ValueError
        If any source genre in consolidation_map is not found in the data.
    """
    df = df.copy()
    df["primary_genre_before_consolidation"] = df[label_col]

    for merged_name, source_genres in consolidation_map.items():
        source_set = set(g.lower().strip() for g in source_genres)
        mask = df[label_col].str.lower().str.strip().isin(source_set)
        count = mask.sum()
        if count > 0:
            df.loc[mask, label_col] = merged_name
            logger.info(
                "Consolidated %d tracks from %s → '%s'",
                count, source_genres, merged_name,
            )
        else:
            logger.warning(
                "Consolidation rule %s → '%s' matched 0 tracks.",
                source_genres, merged_name,
            )

    logger.info(
        "Genre consolidation complete: %d → %d classes.",
        df["primary_genre_before_consolidation"].nunique(),
        df[label_col].nunique(),
    )

    new_dist = df[label_col].value_counts()
    logger.info("Consolidated distribution:")
    for genre, cnt in new_dist.items():
        logger.info("  %-30s  %6d tracks  (%5.1f%%)", genre, cnt,
                    100.0 * cnt / len(df))

    return df


def single_label_train_test_split(
    df: pd.DataFrame,
    label_col: str = "primary_genre",
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple:
    """Split data into train/test for single-label classification.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *label_col* with string genre labels.
    label_col : str
        Column containing single-label genre strings.
    test_size : float
        Fraction of data for test split.
    random_state : int
        Seed for reproducibility.
    stratify : bool
        If True (default), use stratified split to preserve class proportions.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Feature matrices and label vectors for train/test.
    """
    from sklearn.model_selection import train_test_split

    y = df[label_col].values
    stratify_arg = y if stratify else None

    idx_train, idx_test = train_test_split(
        range(len(df)),
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )

    return idx_train, idx_test