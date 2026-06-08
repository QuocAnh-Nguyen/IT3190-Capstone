"""Utility functions for logging, serialization, and plotting."""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a console logger with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def save_joblib(obj: Any, path: Path) -> None:
    """Serialize an object to disk.

    Uses pandas' to_pickle for DataFrames (handles mixed types reliably)
    and joblib.dump for everything else (sklearn estimators, etc.).
    """
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        obj.to_pickle(path)
    else:
        joblib.dump(obj, path, compress=3)


def load_joblib(path: Path) -> Any:
    """Deserialize an object from disk.

    Tries pandas.read_pickle first, falls back to joblib.load.
    """
    import pandas as pd

    path = Path(path)
    try:
        return pd.read_pickle(path)
    except Exception:
        return joblib.load(path)


def save_json(obj: Any, path: Path) -> None:
    """Save a JSON-serializable object to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Path) -> Any:
    """Load a JSON object from disk."""
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def flatten_artists(raw: str) -> list[str]:
    """Parse the artists column (semicolon-separated or string-list)
    into a list of artist IDs."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    raw = str(raw).strip()
    # Handle Python list-string format: ['id1', 'id2']
    if raw.startswith("[") and raw.endswith("]"):
        import ast
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            pass
    # Fallback: split by separator
    return [a.strip().strip("'\"") for a in raw.replace(";", ",").split(",") if a.strip()]