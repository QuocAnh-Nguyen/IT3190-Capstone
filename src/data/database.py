"""Database utilities: SQLite import and query execution.

Provides functions to import the MySQL dump into SQLite and execute
JOIN queries directly. Prefer in-memory pandas joins (loader.py) over
this module for faster iteration, but keep this as a fallback.
"""

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.helpers import get_logger

logger = get_logger(__name__)


def create_sqlite_db(
    sql_dump_path: Path,
    db_path: Path,
) -> sqlite3.Connection:
    """Import a MySQL dump into a local SQLite database.

    NOTE: This is slow for large dumps (~234MB musicoset.sql).
    Use only if CSV data is incomplete.
    """
    logger.info(f"Creating SQLite database at {db_path} from {sql_dump_path}")
    logger.warning(
        "MySQL→SQLite conversion is approximate. "
        "MySQL-specific syntax may cause errors. Prefer CSVs when possible."
    )

    conn = sqlite3.connect(str(db_path))

    # Read and execute the SQL dump
    with open(sql_dump_path, "r", encoding="utf-8", errors="replace") as f:
        sql_script = f.read()

    # Basic MySQL→SQLite compatibility fixes
    sql_script = sql_script.replace("ENGINE=MyISAM", "")
    sql_script = sql_script.replace("DEFAULT CHARSET=utf8", "")
    sql_script = sql_script.replace("COLLATE=utf8_general_mysql500_ci", "")
    sql_script = sql_script.replace("\\'", "''")

    try:
        conn.executescript(sql_script)
        conn.commit()
        logger.info("  SQLite database created successfully")
    except sqlite3.OperationalError as e:
        logger.error(f"  SQLite import error (this is expected for MySQL dumps): {e}")
        logger.info("  Falling back to CSV-based loading in loader.py")

    return conn


def query_genre_feature_matrix(conn: sqlite3.Connection) -> pd.DataFrame:
    """Execute the core JOIN query to build the feature matrix from SQL.

    This is the reference query that the CSV-based loader replicates.
    """
    query = """
    SELECT
        s.song_id,
        s.song_name,
        s.popularity,
        s.explicit,
        s.song_type,
        a.artist_id,
        a.name AS artist_name,
        a.main_genre,
        a.followers,
        a.popularity AS artist_popularity,
        af.duration_ms,
        af.key,
        af.mode,
        af.time_signature,
        af.acousticness,
        af.danceability,
        af.energy,
        af.instrumentalness,
        af.liveness,
        af.loudness,
        af.speechiness,
        af.valence,
        af.tempo,
        l.lyrics
    FROM songs s
    JOIN tracks t ON s.song_id = t.song_id
    JOIN releases r ON t.album_id = r.album_id
    JOIN artists a ON r.artist_id = a.artist_id
    LEFT JOIN acoustic_features af ON s.song_id = af.song_id
    LEFT JOIN lyrics l ON s.song_id = l.song_id
    WHERE a.main_genre IS NOT NULL
    """
    logger.info("Executing genre feature matrix query...")
    df = pd.read_sql_query(query, conn)
    logger.info(f"  Returned {len(df):,} rows")
    return df