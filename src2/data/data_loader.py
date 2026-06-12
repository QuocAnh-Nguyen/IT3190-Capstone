"""Phase 1.2 — Data loading, merging, and audio-file validation.

Loads the core CSV files from `data/raw/` and `data/processed/`, merges them on
`song_id`, and validates that each merged record has a corresponding MP3 file on
disk.
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from src2.config import (
    AUDIO_METADATA_CSV,
    AUDIO_PREVIEWS_DIR,
    LYRICS_CSV,
    SONGS_CSV,
)
from src2.utils.logging_utils import log_section

# Handle potentially large lyrics fields
csv.field_size_limit(sys.maxsize)

logger = logging.getLogger("music_genre")


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------


def load_songs(path: Path = SONGS_CSV) -> pd.DataFrame:
    """Load the songs table (tab-separated).

    Columns: song_id, song_name, billboard, artists, popularity, explicit,
             song_type, genres
    """
    log_section(logger, "Loading songs.csv")
    df = pd.read_csv(path, sep="\t", dtype={"song_id": str, "genres": str})
    logger.info("songs.csv → %d rows, %d cols", *df.shape)
    return df


def load_lyrics(path: Path = LYRICS_CSV) -> pd.DataFrame:
    """Load the lyrics table (tab-separated).

    Columns: song_id, lyrics
    Note: ``lyrics`` is a full song text — fields can be very large, so we use
    the Python csv engine and set ``field_size_limit`` to max.
    """
    log_section(logger, "Loading lyrics.csv")
    df = pd.read_csv(
        path,
        sep="\t",
        dtype={"song_id": str, "lyrics": str},
        quoting=csv.QUOTE_ALL,
        engine="python",
    )
    logger.info("lyrics.csv → %d rows, %d cols", *df.shape)
    return df


def load_audio_metadata(path: Path = AUDIO_METADATA_CSV) -> pd.DataFrame:
    """Load the audio-metadata table (comma-separated).

    Columns include: song_id, song_name, artist_id, artist_name, genre_fine,
    genre_consolidated, mp3_path, mp3_path_abs, file_size_bytes, duration_ms,
    danceability, energy, valence, tempo, downloaded_at
    """
    log_section(logger, "Loading audio_metadata.csv")
    df = pd.read_csv(path, dtype={"song_id": str, "mp3_path": str})
    logger.info("audio_metadata.csv → %d rows, %d cols", *df.shape)
    return df


def load_artists(path: Path) -> pd.DataFrame:
    """Load the artists table (tab-separated) — optional enrichment."""
    logger.info("Loading artists.csv …")
    df = pd.read_csv(path, sep="\t", dtype={"artist_id": str})
    logger.info("artists.csv → %d rows, %d cols", *df.shape)
    return df


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def merge_core_tables(
    songs: pd.DataFrame,
    lyrics: pd.DataFrame,
    audio_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Merge songs ↔ lyrics ↔ audio_metadata on ``song_id`` via inner joins.

    Returns a single DataFrame with columns:
        song_id, song_name, billboard, artists, popularity, explicit,
        song_type, genres (from songs), lyrics (from lyrics), and all
        audio_metadata columns (prefixed ``am_`` where they collide).

    Notes
    -----
    * We use *inner* joins so every record must exist in all three tables.
    * The ``artists`` column from songs.csv is a JSON-like dict string; it is
      preserved as-is.  The plan prefers the ``genres`` column for ground truth.
    """
    log_section(logger, "Merging core tables")

    # songs ↔ lyrics
    merged = pd.merge(
        songs,
        lyrics[["song_id", "lyrics"]],
        on="song_id",
        how="inner",
        suffixes=("", "_lyrics"),
    )
    logger.info("After songs↔lyrics merge: %d rows", len(merged))

    # Add audio_metadata (drop mp3_path_abs if it duplicates mp3_path logic)
    audio_cols = [
        "song_id",
        "artist_id",
        "artist_name",
        "genre_fine",
        "genre_consolidated",
        "mp3_path",
        "file_size_bytes",
        "duration_ms",
        "danceability",
        "energy",
        "valence",
        "tempo",
    ]
    # Only keep cols that exist
    audio_cols = [c for c in audio_cols if c in audio_meta.columns]
    merged = pd.merge(
        merged,
        audio_meta[audio_cols],
        on="song_id",
        how="inner",
        suffixes=("", "_am"),
    )
    logger.info("After songs↔lyrics↔audio_metadata merge: %d rows", len(merged))

    return merged


# ---------------------------------------------------------------------------
# Audio file validation
# ---------------------------------------------------------------------------


def validate_audio_files(
    df: pd.DataFrame,
    audio_dir: Path = AUDIO_PREVIEWS_DIR,
    mp3_col: str = "mp3_path",
) -> pd.DataFrame:
    """Filter the DataFrame to only rows whose MP3 exists on disk.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain an ``mp3_path`` column (e.g. ``"audio_previews/abc.mp3"``).
    audio_dir : Path
        Root directory containing MP3 preview files.
    mp3_col : str
        Column name with the relative mp3 path.

    Returns
    -------
    pd.DataFrame
        Subset of *df* where the file exists.
    """
    log_section(logger, "Validating audio files on disk")

    existing_files = set(
        f.name for f in audio_dir.iterdir() if f.is_file() and f.suffix == ".mp3"
    )

    def _exists(row: pd.Series) -> bool:
        mp3_val = row[mp3_col]
        if pd.isna(mp3_val) or not isinstance(mp3_val, str):
            return False
        fname = Path(mp3_val).name
        return fname in existing_files

    before = len(df)
    mask = df.apply(_exists, axis=1)
    df_valid = df.loc[mask].copy()

    dropped = before - len(df_valid)
    logger.info(
        "Audio validation: %d / %d rows retained (%d dropped — missing file)",
        len(df_valid),
        before,
        dropped,
    )
    return df_valid


# ---------------------------------------------------------------------------
# Convenience: load everything in one call
# ---------------------------------------------------------------------------


def load_and_merge(
    songs_path: Path = SONGS_CSV,
    lyrics_path: Path = LYRICS_CSV,
    audio_meta_path: Path = AUDIO_METADATA_CSV,
    audio_dir: Path = AUDIO_PREVIEWS_DIR,
    validate_audio: bool = True,
) -> pd.DataFrame:
    """Run the full load → merge → validate pipeline.

    Parameters
    ----------
    validate_audio : bool
        If True, cross-reference mp3_path against files on disk.

    Returns
    -------
    pd.DataFrame
        The unified, audio-validated dataset.
    """
    songs = load_songs(songs_path)
    lyrics = load_lyrics(lyrics_path)
    audio_meta = load_audio_metadata(audio_meta_path)

    merged = merge_core_tables(songs, lyrics, audio_meta)

    if validate_audio:
        merged = validate_audio_files(merged, audio_dir=audio_dir)

    logger.info("Final merged dataset: %d rows × %d columns", *merged.shape)
    return merged