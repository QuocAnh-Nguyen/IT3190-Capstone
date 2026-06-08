"""Data loading: read CSVs and execute JOIN logic to build the unified Feature Matrix."""

import ast
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    ACOUSTIC_FEATURE_COLS,
    ARTISTS_CSV,
    LYRICS_CSV,
    MIN_SAMPLES_PER_GENRE,
    RELEASES_CSV,
    SONGS_CSV,
    TARGET_COL,
    TRACKS_CSV,
)
from src.utils.helpers import get_logger

logger = get_logger(__name__)


def _parse_artists_column(raw) -> dict:
    """Parse the songs.artists column (Python dict literal string)."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return {}
    if isinstance(raw, dict):
        return raw
    raw_str = str(raw).strip()
    try:
        return ast.literal_eval(raw_str)
    except (ValueError, SyntaxError):
        logger.warning(f"Could not parse artists value: {raw_str[:100]}")
        return {}


def load_songs(songs_path: Path = SONGS_CSV) -> pd.DataFrame:
    """Load songs table and extract primary artist from the artists dict."""
    logger.info(f"Loading songs from {songs_path}")
    df = pd.read_csv(songs_path, sep="\t")
    logger.info(f"  Loaded {len(df):,} songs")

    # Parse artists dict column
    df["_artists_dict"] = df["artists"].apply(_parse_artists_column)

    # Extract first (primary) artist ID
    df["primary_artist_id"] = df["_artists_dict"].apply(
        lambda d: next(iter(d.keys()), None) if d else None
    )

    # Count number of artists on the track
    df["num_artists"] = df["_artists_dict"].apply(len)

    # Boolean: collaborative track
    df["is_collaborative"] = df["num_artists"] > 1

    # Extract list of all artist IDs (for graph features later)
    df["all_artist_ids"] = df["_artists_dict"].apply(lambda d: list(d.keys()) if d else [])

    # Drop the intermediate parsed dict
    df = df.drop(columns=["_artists_dict"])

    return df


def load_artists(artists_path: Path = ARTISTS_CSV) -> pd.DataFrame:
    """Load artists table with main_genre as target label."""
    logger.info(f"Loading artists from {artists_path}")
    df = pd.read_csv(artists_path, sep="\t")

    # Parse the genres column (Python list-string)
    def parse_genres(raw):
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return []
        if isinstance(raw, list):
            return raw
        try:
            return ast.literal_eval(str(raw))
        except (ValueError, SyntaxError):
            return []

    df["genres_list"] = df["genres"].apply(parse_genres)

    logger.info(f"  Loaded {len(df):,} artists")
    logger.info(f"  Unique main_genre values: {df[TARGET_COL].nunique()}")
    return df


def load_acoustic_features(features_path: Path) -> pd.DataFrame:
    """Load acoustic features CSV."""
    logger.info(f"Loading acoustic features from {features_path}")
    df = pd.read_csv(features_path, sep="\t")
    logger.info(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def load_lyrics(lyrics_path: Path = LYRICS_CSV) -> pd.DataFrame:
    """Load lyrics CSV. Handles TSV format with embedded newlines in lyrics."""
    logger.info(f"Loading lyrics from {lyrics_path}")
    df = pd.read_csv(lyrics_path, sep="\t")
    logger.info(f"  Loaded {len(df):,} rows")
    return df


def load_tracks_releases(
    tracks_path: Path = TRACKS_CSV,
    releases_path: Path = RELEASES_CSV,
) -> pd.DataFrame:
    """Load tracks and releases, join them to get song_id → artist_id mapping."""
    logger.info("Loading tracks and releases for song-artist linkage")
    tracks = pd.read_csv(tracks_path, sep="\t")
    releases = pd.read_csv(releases_path, sep="\t")

    # Join tracks → releases on album_id
    song_artist_map = tracks.merge(releases, on="album_id", how="inner")

    # Keep only the needed columns
    song_artist_map = song_artist_map[["song_id", "artist_id", "album_id", "release_date_x"]].rename(
        columns={"release_date_x": "release_date"}
    )

    logger.info(f"  Song-artist links: {len(song_artist_map):,}")
    return song_artist_map


def build_merged_dataframe(
    songs_df: pd.DataFrame,
    artists_df: pd.DataFrame,
    acoustic_df: pd.DataFrame,
    lyrics_df: pd.DataFrame,
    song_artist_map: pd.DataFrame,
    min_genre_samples: int = MIN_SAMPLES_PER_GENRE,
) -> pd.DataFrame:
    """Execute the full JOIN chain and produce the unified feature matrix.

    JOIN path:
      songs + tracks → releases → artists (genre)
      songs + acoustic_features (on song_id)
      songs + lyrics (on song_id)

    Returns:
        DataFrame with song-level rows, acoustic features, lyrics text,
        artist metadata, and main_genre target.
    """
    logger.info("Building merged feature matrix...")

    # --- Step 1: Link songs to artists via tracks→releases ---
    sa = song_artist_map[["song_id", "artist_id"]].drop_duplicates()

    # One song may map to multiple artists (collaborations).
    # We'll keep all links and handle multi-artist songs later.
    merged = songs_df.merge(sa, left_on="song_id", right_on="song_id", how="inner")

    # --- Step 2: Join artist metadata (main_genre) ---
    artists_subset = artists_df[
        ["artist_id", "name", "main_genre", "followers", "popularity", "genres_list"]
    ].rename(columns={"name": "artist_name", "popularity": "artist_popularity"})

    merged = merged.merge(artists_subset, on="artist_id", how="left")

    # --- Step 3: LEFT JOIN acoustic features ---
    merged = merged.merge(acoustic_df, on="song_id", how="left")

    # --- Step 4: LEFT JOIN lyrics ---
    merged = merged.merge(lyrics_df, on="song_id", how="left")

    # --- Step 5: Handle multi-artist songs ---
    # For songs with multiple artists, we now have multiple rows.
    # Strategy: prefer the primary artist's row, but keep all rows
    # for graph feature computation later.
    # Add a flag for whether this is the primary artist
    merged["_is_primary"] = merged["artist_id"] == merged["primary_artist_id"]

    # --- Step 6: Clean genre labels — basic filtering only ---
    # Drop rows with missing or unclassified genre marker
    before = len(merged)
    merged = merged.dropna(subset=[TARGET_COL])
    logger.info(f"  Dropped {before - len(merged)} rows with NaN genre")

    # Drop artists with unclassified / placeholder genre ("-")
    before = len(merged)
    merged = merged[merged[TARGET_COL] != "-"]
    logger.info(f"  Dropped {before - len(merged)} rows with unclassified genre ('-')")

    logger.info(f"  Pre-consolidation rows: {len(merged):,}")

    return merged


def prepare_single_artist_dataset(
    merged: pd.DataFrame,
    min_genre_samples: int = MIN_SAMPLES_PER_GENRE,
) -> pd.DataFrame:
    """For supervised training, collapse multi-artist songs to one row per song,
    then consolidate rare genres into an 'Other' category.

    Strategy: Keep only the primary artist's row for each song.
    This avoids label ambiguity (one song having multiple genres).
    """
    logger.info("Collapsing to one row per song (primary artist only)...")
    before = len(merged)
    df = merged[merged["_is_primary"]].copy()
    df = df.drop(columns=["_is_primary"])
    logger.info(f"  {before:,} → {len(df):,} rows")

    # --- Consolidate rare genres (run AFTER collapse for accurate counts) ---
    genre_counts = df[TARGET_COL].value_counts()
    rare_genres = genre_counts[genre_counts < min_genre_samples].index.tolist()
    if rare_genres:
        logger.info(
            f"  Consolidating {len(rare_genres)} rare genres "
            f"(< {min_genre_samples} samples) → 'Other'"
        )
        df[TARGET_COL] = df[TARGET_COL].apply(
            lambda g: "Other" if g in rare_genres else g
        )

    # Drop "Other" rows if the consolidated group is still too small
    genre_counts = df[TARGET_COL].value_counts()
    if "Other" in genre_counts.index and genre_counts["Other"] < min_genre_samples:
        before = len(df)
        df = df[df[TARGET_COL] != "Other"]
        logger.info(
            f"  Dropped {before - len(df)} 'Other' rows (below min samples)"
        )

    final_genres = df[TARGET_COL].value_counts()
    logger.info(f"  Final genres: {len(final_genres)} classes")
    logger.info(f"  Top-10 class distribution:")
    for genre, count in final_genres.head(10).items():
        logger.info(f"    {genre:<30s} {count:>5d}")

    return df