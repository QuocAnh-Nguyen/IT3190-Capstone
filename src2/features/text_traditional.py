"""Phase 2.2 — Traditional Text Features (improved).

Post-improve_plan: TF-IDF max_features reduced from 5000 → 500, and
TruncatedSVD applied after TF-IDF to further reduce text dimensions to
~100 latent dimensions (configurable via TEXT_SVD_COMPONENTS).
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from typing import Tuple

from src2.config import (
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TEXT_SVD_COMPONENTS,
    RANDOM_SEED,
)

logger = logging.getLogger("music_genre")


def extract_text_features(
    df: pd.DataFrame,
    apply_svd: bool = True,
    svd_components: int = TEXT_SVD_COMPONENTS,
) -> Tuple[np.ndarray, list[str]]:
    """Extract TF-IDF, Sentiment, and Lexical metrics from lyrics.

    Post-improve_plan changes:
    - TF-IDF max_features reduced (5000 → 500)
    - TruncatedSVD applied after TF-IDF to get ~100 latent text dimensions
    - This rebalances audio-to-text feature ratio from 209:5005 to 209:100

    Parameters
    ----------
    df : pd.DataFrame
        Must have ``has_lyrics`` (bool) and ``lyrics`` (str) columns.
    apply_svd : bool
        If True, apply TruncatedSVD after TF-IDF to reduce dimensions.
    svd_components : int
        Number of SVD components (latent dimensions). Defaults to
        TEXT_SVD_COMPONENTS from config.

    Returns
    -------
    text_matrix : np.ndarray
        Feature matrix of shape (n_songs, n_text_features).
    song_ids : list[str]
        Ordered song IDs corresponding to rows of text_matrix.
    """
    # Filter for rows that actually have lyrics
    valid_df = df[df["has_lyrics"].astype(bool)].copy()
    lyrics = valid_df["lyrics"].fillna("").astype(str)
    song_ids = valid_df["song_id"].tolist()

    if len(lyrics) == 0:
        logger.warning("No valid lyrics found. Returning empty array.")
        return np.array([]).reshape(0, 0), []

    logger.info(
        "Extracting TF-IDF features (max_features=%d, ngram_range=%s)...",
        TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE,
    )
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        stop_words="english",
        strip_accents="unicode",
        lowercase=True,
    )

    # TF-IDF Matrix
    tfidf_matrix = vectorizer.fit_transform(lyrics)
    logger.info("TF-IDF matrix shape: %s (sparse)", tfidf_matrix.shape)

    # Apply TruncatedSVD for dimensionality reduction (Step 4A)
    if apply_svd and tfidf_matrix.shape[1] > svd_components:
        effective_components = min(svd_components, tfidf_matrix.shape[1] - 1,
                                   tfidf_matrix.shape[0] - 1)
        logger.info(
            "Applying TruncatedSVD: %d components (from %d TF-IDF features)...",
            effective_components, tfidf_matrix.shape[1],
        )
        svd = TruncatedSVD(
            n_components=effective_components,
            random_state=RANDOM_SEED,
        )
        tfidf_reduced = svd.fit_transform(tfidf_matrix)
        explained = svd.explained_variance_ratio_.sum()
        logger.info(
            "TruncatedSVD explained variance: %.4f  |  reduced shape: %s",
            explained, tfidf_reduced.shape,
        )
        tfidf_dense = tfidf_reduced
    elif apply_svd:
        logger.info(
            "Skipping SVD: TF-IDF features (%d) <= target components (%d).",
            tfidf_matrix.shape[1], svd_components,
        )
        tfidf_dense = tfidf_matrix.toarray()
    else:
        tfidf_dense = tfidf_matrix.toarray()

    # ------------------------------------------------------------------
    # Lexical & Sentiment features (Step 4A: kept — they're small but useful)
    # ------------------------------------------------------------------
    logger.info("Extracting Lexical & Sentiment features...")
    lexical_features = []

    for text in lyrics:
        blob = TextBlob(text)
        words = blob.words
        vocab_size = len(set(words))
        word_count = len(words)

        # Avoid division by zero
        ttr = vocab_size / word_count if word_count > 0 else 0
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        lexical_features.append([vocab_size, word_count, ttr, polarity, subjectivity])

    lexical_matrix = np.array(lexical_features)
    logger.info("Lexical features shape: %s", lexical_matrix.shape)

    # Concatenate TF-IDF (possibly SVD-reduced) with lexical features
    final_text_matrix = np.hstack([tfidf_dense, lexical_matrix])
    logger.info(
        "Final text feature matrix: %s  (TF-IDF=%d + lexical=%d)",
        final_text_matrix.shape, tfidf_dense.shape[1], lexical_matrix.shape[1],
    )

    return final_text_matrix, song_ids