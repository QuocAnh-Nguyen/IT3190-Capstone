#!/usr/bin/env python3
"""Phase 1: Data Engineering pipeline — load, join, clean, and inspect.

Usage: python scripts/run_phase1.py
Output: Writes summary to stdout; serializes merged data to data/processed/
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Keep logs at INFO but send them to stderr
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from src.data.loader import (
    load_songs,
    load_artists,
    load_acoustic_features,
    load_lyrics,
    load_tracks_releases,
    build_merged_dataframe,
    prepare_single_artist_dataset,
)
from src.config import (
    ACOUSTIC_FEATURES_CSV,
    DATA_PROCESSED_DIR,
)
from src.utils.helpers import save_joblib

MERGED_PKL = DATA_PROCESSED_DIR / "merged_data.pkl"

# ---------------------------------------------------------------------------
# 1. Load all raw tables
# ---------------------------------------------------------------------------
songs_df = load_songs()
artists_df = load_artists()
acoustic_df = load_acoustic_features(ACOUSTIC_FEATURES_CSV)
lyrics_df = load_lyrics()
song_artist_map = load_tracks_releases()

# ---------------------------------------------------------------------------
# 2. Build merged feature matrix (JOIN chain)
# ---------------------------------------------------------------------------
merged = build_merged_dataframe(
    songs_df,
    artists_df,
    acoustic_df,
    lyrics_df,
    song_artist_map,
)

# ---------------------------------------------------------------------------
# 3. Collapse to one row per song + consolidate rare genres
# ---------------------------------------------------------------------------
df = prepare_single_artist_dataset(merged, min_genre_samples=50)

# ---------------------------------------------------------------------------
# 4. Summary statistics
# ---------------------------------------------------------------------------
genre_counts = df["main_genre"].value_counts()

print("\n" + "=" * 60)
print("PHASE 1 — DATA ENGINEERING COMPLETE")
print("=" * 60)
print(f"Total training samples:  {len(df):,}")
print(f"Number of genres:        {df['main_genre'].nunique()}")
print(f"Songs with lyrics:       {df['lyrics'].notna().sum():,}")
print(f"Songs missing lyrics:    {df['lyrics'].isna().sum():,}")
print(f"Collaborative songs:     {df['is_collaborative'].sum():,}")
print(f"Feature columns:         {len(df.columns)}")

print("\n--- Genre Distribution ---")
for genre, count in genre_counts.items():
    pct = count / len(df) * 100
    bar = "█" * int(pct / 2)
    print(f"  {genre:<30s} {count:>5d}  ({pct:5.1f}%) {bar}")

print("\n--- Imbalance Metrics ---")
print(f"  Majority class:  {genre_counts.index[0]} ({genre_counts.iloc[0]})")
print(f"  Minority class:  {genre_counts.index[-1]} ({genre_counts.iloc[-1]})")
print(f"  Imbalance ratio: {genre_counts.iloc[0] / genre_counts.iloc[-1]:.1f}:1")

# ---------------------------------------------------------------------------
# 5. Persist merged data
# ---------------------------------------------------------------------------
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
save_joblib(df, MERGED_PKL)
file_size_mb = MERGED_PKL.stat().st_size / 1024**2
print(f"\nSaved merged data → {MERGED_PKL}")
print(f"File size: {file_size_mb:.1f} MB")
print("\nDone.")