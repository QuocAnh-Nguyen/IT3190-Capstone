"""I/O helpers: save/load pickle, CSV, NPY, and model artefacts."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Pickle / joblib
# ---------------------------------------------------------------------------

def save_pickle(obj: Any, path: Path) -> None:
    """Save a Python object to disk via pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> Any:
    """Load a pickled Python object."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_joblib(obj: Any, path: Path) -> None:
    """Save a scikit-learn compatible object via joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path: Path) -> Any:
    """Load a joblib-serialised object."""
    return joblib.load(path)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def save_csv(df: pd.DataFrame, path: Path, index: bool = True) -> None:
    """Save a DataFrame to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def load_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    """Load a CSV into a DataFrame."""
    return pd.read_csv(path, **kwargs)


# ---------------------------------------------------------------------------
# NumPy arrays
# ---------------------------------------------------------------------------

def save_npy(arr: np.ndarray, path: Path) -> None:
    """Save a NumPy array to .npy format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def load_npy(path: Path) -> np.ndarray:
    """Load a NumPy array from .npy format."""
    return np.load(path)


# ---------------------------------------------------------------------------
# Text / list helpers
# ---------------------------------------------------------------------------

def save_text_list(items: list[str], path: Path) -> None:
    """Write a list of strings, one per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item}\n")


def load_text_list(path: Path) -> list[str]:
    """Read a list of strings, one per line."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def save_json(obj: Any, path: Path) -> None:
    """Save an object to JSON (handles numpy numeric types)."""
    path.parent.mkdir(parents=True, exist_ok=True)

    class NpEncoder(json.JSONEncoder):
        def default(self, o: Any) -> Any:
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, cls=NpEncoder, ensure_ascii=False)


def load_json(path: Path) -> Any:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)