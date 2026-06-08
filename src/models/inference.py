"""Inference pipeline: loads the full trained pipeline and produces predictions.

This is the non-UI entry point used by the Streamlit app (and any future
API). It handles:
- Loading the trained model pipeline, scaler, label encoder
- Loading the TF-IDF vectorizer and SVD reducer for lyrics preprocessing
- Loading the PolynomialFeatures transformer for interaction features
- Producing genre predictions with probabilities
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import MODELS_DIR
from src.features.nlp_features import clean_lyrics, lexical_richness, sentiment_scores, structural_features
from src.utils.helpers import get_logger, load_joblib

logger = get_logger(__name__)


class GenrePredictor:
    """Complete inference pipeline for multi-modal music genre prediction."""

    def __init__(self, models_dir: Optional[Path] = None):
        """Load all pipeline artifacts.

        Args:
            models_dir: Directory containing all saved model artifacts.
                Defaults to MODELS_DIR from config.
        """
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load all serialized pipeline components."""
        d = self.models_dir

        # Core ML pipeline
        self.pipeline = load_joblib(d / "best_model_pipeline.joblib")
        self.scaler = load_joblib(d / "feature_scaler.joblib")
        self.label_encoder = load_joblib(d / "label_encoder.joblib")
        self.feature_cols = load_joblib(d / "feature_columns.joblib")
        logger.info(f"Loaded pipeline: {len(self.feature_cols)} features, "
                     f"{len(self.label_encoder.classes_)} genres")

        # NLP components
        self.tfidf_vectorizer = load_joblib(d / "tfidf_vectorizer.joblib")
        self.svd_reducer = load_joblib(d / "svd_reducer.joblib")
        logger.info("Loaded NLP vectorizer + SVD reducer")

        # Interaction features transformer
        if (d / "poly_features.joblib").exists():
            self.poly_transformer = load_joblib(d / "poly_features.joblib")
            logger.info("Loaded PolynomialFeatures transformer")
        else:
            self.poly_transformer = None

        # Genre names
        self.genre_names = list(self.label_encoder.classes_)

    def _extract_lyrics_features(self, lyrics: str) -> Dict[str, float]:
        """Extract NLP features from raw lyrics text."""
        cleaned = clean_lyrics(lyrics)

        features = {}

        # Lexical richness
        lr = lexical_richness(cleaned)
        features["nlp_ttr"] = lr["ttr"]
        features["nlp_hapax_ratio"] = lr["hapax_ratio"]
        features["nlp_avg_word_length"] = lr["avg_word_length"]
        features["nlp_word_count"] = lr["word_count"]

        # Sentiment
        sent = sentiment_scores(cleaned)
        features["nlp_polarity"] = sent["polarity"]
        features["nlp_subjectivity"] = sent["subjectivity"]

        # Structural
        struct = structural_features(cleaned)
        features["nlp_line_count"] = struct["line_count"]
        features["nlp_repetition_score"] = struct["repetition_score"]

        # TF-IDF features
        if self.tfidf_vectorizer and self.svd_reducer:
            tfidf_vec = self.tfidf_vectorizer.transform([cleaned])
            tfidf_reduced = self.svd_reducer.transform(tfidf_vec)
            for i in range(tfidf_reduced.shape[1]):
                features[f"nlp_tfidf_{i}"] = tfidf_reduced[0, i]

        return features

    def _build_feature_vector(
        self,
        acoustic_features: Dict[str, float],
        lyrics: str,
        artist_popularity: float = 50.0,
        followers: float = 0,
        popularity: float = 50.0,
        explicit: bool = False,
        num_artists: int = 1,
    ) -> np.ndarray:
        """Build the complete feature vector matching the training schema.

        Args:
            acoustic_features: Dict with keys: duration_ms, key, mode,
                time_signature, acousticness, danceability, energy,
                instrumentalness, liveness, loudness, speechiness, valence, tempo.
            lyrics: Raw lyrics text.
            artist_popularity: Artist Spotify popularity (0-100).
            followers: Artist follower count.
            popularity: Track popularity (0-100).
            explicit: Whether the track has explicit lyrics.
            num_artists: Number of artists on the track.

        Returns:
            1D numpy array matching the training feature columns.
        """
        # Start with basic numeric features
        row = {}

        # Acoustic features
        acoustic_keys = [
            "duration_ms", "key", "mode", "time_signature",
            "acousticness", "danceability", "energy", "instrumentalness",
            "liveness", "loudness", "speechiness", "valence", "tempo",
        ]
        for k in acoustic_keys:
            row[k] = acoustic_features.get(k, 0.0)

        # Metadata features
        row["popularity"] = popularity
        row["explicit"] = 1 if explicit else 0
        row["num_artists"] = num_artists
        row["artist_popularity"] = artist_popularity
        row["followers"] = followers

        # --- Graph features ---
        row["graph_num_artists"] = num_artists
        row["graph_is_collaborative"] = 1 if num_artists > 1 else 0
        # Default graph metrics for unknown artists
        row["graph_degree_centrality"] = 0.0
        row["graph_betweenness_centrality"] = 0.0
        row["graph_clustering_coefficient"] = 0.0
        row["graph_eigenvector_centrality"] = 0.0

        # --- NLP features ---
        lyrics_features = self._extract_lyrics_features(lyrics)
        row.update(lyrics_features)

        # --- Interaction features ---
        # Generate interaction features using the fitted PolynomialFeatures
        acoustic_array = np.array([row[k] for k in acoustic_keys]).reshape(1, -1)
        if self.poly_transformer:
            X_poly = self.poly_transformer.transform(acoustic_array)
            # Interaction columns are all columns after the bias and original features
            n_orig = len(acoustic_keys)
            for i in range(n_orig, X_poly.shape[1]):
                row[f"interact_x{i - n_orig}"] = X_poly[0, i]

        # --- Cluster features (use mean=0 since unknown track) ---
        row["cluster_distance"] = 0.0
        # One-hot cluster columns — use cluster_0 as default
        # (These will be ~0 after scaling)

        # Build array aligned to feature_cols
        feature_vec = []
        for col in self.feature_cols:
            if col in row:
                feature_vec.append(row[col])
            else:
                feature_vec.append(0.0)

        return np.array(feature_vec, dtype=np.float64)

    def predict(
        self,
        acoustic_features: Dict[str, float],
        lyrics: str = "",
        artist_popularity: float = 50.0,
        followers: float = 0,
        popularity: float = 50.0,
        explicit: bool = False,
        num_artists: int = 1,
    ) -> Dict:
        """Predict genre from acoustic features and lyrics.

        Args:
            acoustic_features: Dict of 13 Spotify acoustic feature values.
            lyrics: Raw lyrics text (can be empty).
            artist_popularity: Artist popularity score (0-100).
            followers: Artist follower count.
            popularity: Track popularity score (0-100).
            explicit: Whether the track has explicit lyrics.
            num_artists: Number of artists on the track.

        Returns:
            Dict with:
                - predicted_genre: str
                - probabilities: Dict[str, float] (genre → probability)
                - top3_genres: List[Tuple[str, float]]
        """
        # Build feature vector
        X = self._build_feature_vector(
            acoustic_features, lyrics,
            artist_popularity, followers, popularity, explicit, num_artists,
        )

        # Scale
        X_scaled = self.scaler.transform(X.reshape(1, -1))

        # Predict
        if hasattr(self.pipeline, "predict_proba"):
            proba = self.pipeline.predict_proba(X_scaled)[0]
        else:
            # Fallback for models without predict_proba
            pred = self.pipeline.predict(X_scaled)[0]
            proba = np.zeros(len(self.genre_names))
            proba[pred] = 1.0

        # Build probability map
        proba_dict = {
            genre: float(prob)
            for genre, prob in zip(self.genre_names, proba)
        }

        # Sort by probability
        sorted_genres = sorted(proba_dict.items(), key=lambda x: -x[1])
        predicted_genre = sorted_genres[0][0]

        return {
            "predicted_genre": predicted_genre,
            "probabilities": proba_dict,
            "top3_genres": sorted_genres[:3],
        }

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Return feature importance if the model supports it."""
        # Try to get the classifier from inside the pipeline
        if hasattr(self.pipeline, "named_steps") and "clf" in self.pipeline.named_steps:
            clf = self.pipeline.named_steps["clf"]
            if hasattr(clf, "feature_importances_"):
                importance = clf.feature_importances_
                return pd.DataFrame({
                    "feature": self.feature_cols,
                    "importance": importance,
                }).sort_values("importance", ascending=False)
        return None