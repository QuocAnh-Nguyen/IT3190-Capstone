"""Phase 1.3 — Missing data handling.

Implements the PRD strategy:
1. Compute comprehensive data-availability statistics.
2. Flag records missing the target variable (genres) for removal.
3. Flag records missing one modality (audio XOR lyrics) but retain them —
   the model should handle single-modality prediction via masking.
4. Apply minimum genre frequency threshold to filter ultra-rare labels.
5. Produce a cleaned DataFrame ready for label encoding.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import pandas as pd

from src2.config import MIN_GENRE_COUNT
from src2.utils.logging_utils import log_section

logger = logging.getLogger("music_genre")


# ---------------------------------------------------------------------------
# Availability report
# ---------------------------------------------------------------------------


def compute_availability(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-modality availability statistics.

    Returns a summary DataFrame with counts and percentages.
    """
    total = len(df)

    has_genres = (~df["genres"].isna()) & (df["genres"].str.strip() != "")
    has_lyrics = (~df["lyrics"].isna()) & (df["lyrics"].str.strip() != "")
    has_audio = (~df["mp3_path"].isna()) & (df["mp3_path"].str.strip() != "")

    stats = {
        "total_records": total,
        "has_genres": has_genres.sum(),
        "has_genres_pct": round(100 * has_genres.sum() / total, 2),
        "has_lyrics": has_lyrics.sum(),
        "has_lyrics_pct": round(100 * has_lyrics.sum() / total, 2),
        "has_audio_file": has_audio.sum(),
        "has_audio_file_pct": round(100 * has_audio.sum() / total, 2),
        "has_all_three": (has_genres & has_lyrics & has_audio).sum(),
        "has_genres_only": (has_genres & ~has_lyrics & ~has_audio).sum(),
        "has_lyrics_only": (~has_genres & has_lyrics & ~has_audio).sum(),
        "has_audio_only": (~has_genres & ~has_lyrics & has_audio).sum(),
        "missing_all_three": (~has_genres & ~has_lyrics & ~has_audio).sum(),
        "missing_genres": (~has_genres).sum(),
        "missing_lyrics": (~has_lyrics).sum(),
        "missing_audio": (~has_audio).sum(),
        "has_genres_and_lyrics_only": (has_genres & has_lyrics & ~has_audio).sum(),
        "has_genres_and_audio_only": (has_genres & ~has_lyrics & has_audio).sum(),
    }

    report = pd.DataFrame([stats]).T
    report.columns = ["count"]
    return report


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def drop_missing_genres(df: pd.DataFrame) -> pd.DataFrame:
    """Drop records that are missing the target variable (genres)."""
    before = len(df)
    mask = (~df["genres"].isna()) & (df["genres"].str.strip() != "")
    df_clean = df.loc[mask].copy()
    dropped = before - len(df_clean)
    logger.info(
        "Dropped %d records with missing/empty genres (%d → %d)",
        dropped, before, len(df_clean),
    )
    return df_clean


def flag_modality_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean flags indicating which modalities are available per record.

    New columns:
        ``has_lyrics`` — True if lyrics text is non-empty
        ``has_audio`` — True if mp3_path is valid and file exists (already validated during load)
    """
    df = df.copy()
    df["has_lyrics"] = (~df["lyrics"].isna()) & (df["lyrics"].str.strip() != "")
    # All records post-validation have valid audio
    df["has_audio"] = True
    logger.info("Modality flags added: has_lyrics=%d, has_audio=%d",
                df["has_lyrics"].sum(), df["has_audio"].sum())
    return df


def filter_rare_genres(
    df: pd.DataFrame,
    min_count: int = MIN_GENRE_COUNT,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Identify and report genre labels that appear below *min_count*.

    This function does NOT remove individual genre labels from multi-label
    rows yet — it only reports which labels are rare.  The actual filtering
    happens in ``label_encoder.py`` during multi-label binarization.

    Returns
    -------
    df : pd.DataFrame
        Unmodified input.
    rare_summary : pd.DataFrame
        Genre counts, sorted ascending, with a ``keep`` column.
    """
    log_section(logger, "Genre frequency analysis")

    genre_counts = (
        df["genres"]
        .dropna()
        .str.split(";")
        .explode()
        .str.strip()
        .value_counts()
    )
    rare_summary = pd.DataFrame({
        "genre": genre_counts.index,
        "count": genre_counts.values,
    })
    rare_summary["keep"] = rare_summary["count"] >= min_count

    n_rare = (rare_summary["count"] < min_count).sum()
    logger.info(
        "Unique genres: %d  |  Min count threshold: %d  |  Dropped genres: %d",
        len(rare_summary), min_count, n_rare,
    )
    if n_rare > 0:
        logger.info(
            "Rare genres (< %d): %s",
            min_count,
            rare_summary.loc[~rare_summary["keep"], "genre"].tolist(),
        )

    return df, rare_summary


# ---------------------------------------------------------------------------
# Main cleaning pipeline
# ---------------------------------------------------------------------------


def clean_dataset(
    df: pd.DataFrame,
    min_genre_count: int = MIN_GENRE_COUNT,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full cleaning pipeline.

    1. Compute and log data availability
    2. Drop records missing the target (genres)
    3. Flag modality presence (has_lyrics, has_audio)
    4. Analyse genre frequency distribution

    Returns
    -------
    cleaned : pd.DataFrame
    availability_report : pd.DataFrame
    genre_frequency : pd.DataFrame
    """
    log_section(logger, "Phase 1.3 — Missing Data Handling")

    # 1. Availability
    avail = compute_availability(df)
    logger.info("Data availability report:\n%s", avail.to_string())

    # 2. Drop missing target
    cleaned = drop_missing_genres(df)

    # 3. Flag modalities
    cleaned = flag_modality_mask(cleaned)

    # 4. Genre frequency
    cleaned, genre_freq = filter_rare_genres(cleaned, min_count=min_genre_count)

    logger.info("Cleaned dataset: %d rows × %d columns", *cleaned.shape)
    return cleaned, avail, genre_freq