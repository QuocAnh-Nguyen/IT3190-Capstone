"""NLP feature extraction from song lyrics.

Extracts:
- Lexical richness metrics (TTR, hapax legomena ratio)
- Sentiment polarity & subjectivity
- TF-IDF vector embeddings (reduced via TruncatedSVD)
- Structural features (line count, avg word length, repetitiveness)
"""

import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

from src.config import (
    PCA_VARIANCE_THRESHOLD,
    RANDOM_SEED,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
)
from src.utils.helpers import get_logger, save_joblib

logger = get_logger(__name__)

# Compile regexps once
BRACKET_PATTERN = re.compile(r"\[.*?\]")
MULTI_SPACE_PATTERN = re.compile(r"\s+")
NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z\s]")


def _ensure_nltk_resources():
    """Download required NLTK corpora if missing."""
    try:
        word_tokenize("test")
    except LookupError:
        import nltk

        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("stopwords", quiet=True)


def clean_lyrics(text: Optional[str]) -> str:
    """Preprocess raw lyrics text.

    - Remove bracket annotations ([Verse], [Chorus], etc.)
    - Lowercase
    - Collapse whitespace
    - Keep only alphabetic characters + spaces
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = str(text)
    # Remove bracket annotations
    text = BRACKET_PATTERN.sub(" ", text)
    # Lowercase
    text = text.lower()
    # Remove non-alpha characters
    text = NON_ALPHA_PATTERN.sub(" ", text)
    # Collapse whitespace
    text = MULTI_SPACE_PATTERN.sub(" ", text).strip()
    return text


def lexical_richness(text: str) -> dict:
    """Compute lexical richness metrics for a text."""
    if not text.strip():
        return {
            "ttr": 0.0,
            "hapax_ratio": 0.0,
            "avg_word_length": 0.0,
            "word_count": 0,
        }

    tokens = text.split()
    word_count = len(tokens)
    unique_count = len(set(tokens))

    # Type-Token Ratio
    ttr = unique_count / word_count if word_count > 0 else 0.0

    # Hapax Legomena ratio (words appearing exactly once)
    from collections import Counter

    counts = Counter(tokens)
    hapax_count = sum(1 for c in counts.values() if c == 1)
    hapax_ratio = hapax_count / word_count if word_count > 0 else 0.0

    # Average word length
    avg_wl = np.mean([len(t) for t in tokens]) if tokens else 0.0

    return {
        "ttr": ttr,
        "hapax_ratio": hapax_ratio,
        "avg_word_length": avg_wl,
        "word_count": word_count,
    }


def sentiment_scores(text: str) -> dict:
    """Extract sentiment polarity and subjectivity via TextBlob.

    Polarity: -1.0 (negative) to +1.0 (positive)
    Subjectivity: 0.0 (objective) to 1.0 (subjective)
    """
    if not text.strip():
        return {"polarity": 0.0, "subjectivity": 0.0}
    blob = TextBlob(text)
    return {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity,
    }


def structural_features(text: str) -> dict:
    """Compute structural metrics of the lyrics."""
    if not text.strip():
        return {"line_count": 0, "repetition_score": 0.0}

    lines = text.split("\n")
    lines = [l.strip() for l in lines if l.strip()]
    line_count = len(lines)

    # Repetition score: fraction of lines that appear more than once
    from collections import Counter

    line_counts = Counter(lines)
    repeated = sum(c for c in line_counts.values() if c > 1)
    repetition_score = repeated / line_count if line_count > 0 else 0.0

    return {"line_count": line_count, "repetition_score": repetition_score}


def extract_nlp_features(
    df: pd.DataFrame,
    lyrics_col: str = "lyrics",
    fit_vectorizer: bool = True,
    save_dir: Optional[Path] = None,
    tfidf_matrix: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, Optional[TfidfVectorizer], Optional[TruncatedSVD]]:
    """Extract all NLP features from lyrics.

    Returns:
        DataFrame with lexical, sentiment, structural features +
        TF-IDF vector columns appended.
        Also returns the fitted vectorizer and SVD reducer (if fitting).
    """
    logger.info("Extracting NLP features from lyrics...")
    _ensure_nltk_resources()

    # Clean lyrics
    df = df.copy()
    df["_clean_lyrics"] = df[lyrics_col].apply(clean_lyrics)

    empty_count = (df["_clean_lyrics"] == "").sum()
    logger.info(f"  Empty lyrics after cleaning: {empty_count:,} / {len(df):,}")

    # --- Lexical Richness ---
    logger.info("  Computing lexical richness...")
    lr_feats = df["_clean_lyrics"].apply(lexical_richness).apply(pd.Series)
    for col in lr_feats.columns:
        df[f"nlp_{col}"] = lr_feats[col]

    # --- Sentiment ---
    logger.info("  Computing sentiment...")
    sent_feats = df["_clean_lyrics"].apply(sentiment_scores).apply(pd.Series)
    df["nlp_polarity"] = sent_feats["polarity"]
    df["nlp_subjectivity"] = sent_feats["subjectivity"]

    # --- Structural ---
    logger.info("  Computing structural features...")
    struct_feats = df["_clean_lyrics"].apply(structural_features).apply(pd.Series)
    df["nlp_line_count"] = struct_feats["line_count"]
    df["nlp_repetition_score"] = struct_feats["repetition_score"]

    # --- TF-IDF Vectorization ---
    logger.info("  Vectorizing with TF-IDF...")
    vectorizer = None
    svd = None

    if fit_vectorizer:
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            stop_words="english",
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
        )
        tfidf_mat = vectorizer.fit_transform(df["_clean_lyrics"])
    elif tfidf_matrix is not None:
        # Use pre-provided TF-IDF matrix (e.g., from training set transform)
        tfidf_mat = tfidf_matrix
    else:
        raise ValueError("Either fit_vectorizer=True or tfidf_matrix must be provided")

    logger.info(f"    TF-IDF shape: {tfidf_mat.shape}")

    # Reduce dimensionality with TruncatedSVD
    n_components = min(50, tfidf_mat.shape[1] - 1)
    if fit_vectorizer:
        svd = TruncatedSVD(
            n_components=n_components,
            random_state=RANDOM_SEED,
        )
        tfidf_reduced = svd.fit_transform(tfidf_mat)
        var_explained = svd.explained_variance_ratio_.sum()
        logger.info(
            f"    Reduced to {n_components} components "
            f"({var_explained:.1%} variance explained)"
        )
    else:
        # Use existing SVD for transform
        tfidf_reduced = tfidf_mat  # caller must handle reduction

    # Add TF-IDF reduced features as columns
    tfidf_cols = [f"nlp_tfidf_{i}" for i in range(n_components)]
    if fit_vectorizer:
        tfidf_df = pd.DataFrame(tfidf_reduced, index=df.index, columns=tfidf_cols)
    else:
        tfidf_df = pd.DataFrame(
            tfidf_reduced, index=df.index if hasattr(df, "index") else None,
            columns=tfidf_cols[:tfidf_reduced.shape[1]],
        )

    df = pd.concat([df, tfidf_df], axis=1)

    # Serialize
    if save_dir and fit_vectorizer:
        save_dir = Path(save_dir)
        save_joblib(vectorizer, save_dir / "tfidf_vectorizer.joblib")
        save_joblib(svd, save_dir / "svd_reducer.joblib")
        logger.info(f"  Saved vectorizer & SVD → {save_dir}")

    # Track NLP feature column names
    nlp_feature_cols = (
        ["nlp_ttr", "nlp_hapax_ratio", "nlp_avg_word_length", "nlp_word_count",
         "nlp_polarity", "nlp_subjectivity",
         "nlp_line_count", "nlp_repetition_score"]
        + tfidf_cols
    )

    logger.info(f"  Total NLP features: {len(nlp_feature_cols)}")
    logger.info("  NLP feature extraction complete")

    return df, vectorizer, svd