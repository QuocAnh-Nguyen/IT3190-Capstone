"""Streamlit web application for Multi-modal Music Genre Classification.

Allows users to:
1. Input acoustic parameters via sliders
2. Paste song lyrics for NLP analysis
3. Get real-time genre predictions with probabilities
4. View SHAP feature importance explanations (bonus)
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from src.models.inference import GenrePredictor

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load model (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_predictor():
    return GenrePredictor()


predictor = load_predictor()

# ---------------------------------------------------------------------------
# Sidebar — Acoustic Parameters
# ---------------------------------------------------------------------------
st.sidebar.title("🎛️ Acoustic Parameters")
st.sidebar.caption("Adjust Spotify acoustic features (or use the presets below)")

# Presets
preset = st.sidebar.selectbox(
    "Quick Preset",
    ["Custom", "Pop (Danceable & Energetic)", "Rock (Loud & Live)",
     "Hip Hop (Speech-heavy)", "Acoustic Ballad", "EDM (Electronic)"],
    index=0,
)

# Map presets to defaults
presets = {
    "Pop (Danceable & Energetic)": {
        "duration_ms": 210000, "key": 5, "mode": 1, "time_signature": 4,
        "acousticness": 0.10, "danceability": 0.75, "energy": 0.72,
        "instrumentalness": 0.0, "liveness": 0.12, "loudness": -5.5,
        "speechiness": 0.05, "valence": 0.65, "tempo": 120.0,
    },
    "Rock (Loud & Live)": {
        "duration_ms": 240000, "key": 2, "mode": 1, "time_signature": 4,
        "acousticness": 0.05, "danceability": 0.45, "energy": 0.85,
        "instrumentalness": 0.02, "liveness": 0.30, "loudness": -4.0,
        "speechiness": 0.06, "valence": 0.40, "tempo": 135.0,
    },
    "Hip Hop (Speech-heavy)": {
        "duration_ms": 200000, "key": 1, "mode": 0, "time_signature": 4,
        "acousticness": 0.20, "danceability": 0.80, "energy": 0.65,
        "instrumentalness": 0.0, "liveness": 0.10, "loudness": -6.5,
        "speechiness": 0.35, "valence": 0.50, "tempo": 95.0,
    },
    "Acoustic Ballad": {
        "duration_ms": 200000, "key": 0, "mode": 1, "time_signature": 3,
        "acousticness": 0.90, "danceability": 0.30, "energy": 0.15,
        "instrumentalness": 0.05, "liveness": 0.08, "loudness": -12.0,
        "speechiness": 0.03, "valence": 0.25, "tempo": 75.0,
    },
    "EDM (Electronic)": {
        "duration_ms": 200000, "key": 7, "mode": 0, "time_signature": 4,
        "acousticness": 0.01, "danceability": 0.70, "energy": 0.95,
        "instrumentalness": 0.60, "liveness": 0.05, "loudness": -3.5,
        "speechiness": 0.04, "valence": 0.55, "tempo": 128.0,
    },
}

if preset != "Custom":
    defaults = presets[preset]
else:
    defaults = {}

duration_ms = st.sidebar.slider(
    "Duration (ms)", 30000, 600000, defaults.get("duration_ms", 200000), 5000,
    help="Track length in milliseconds"
)
key = st.sidebar.slider(
    "Key", -1, 11, defaults.get("key", 5), 1,
    help="Estimated overall key (-1 = unknown, 0=C, 1=C#, ..., 11=B)"
)
mode = st.sidebar.selectbox(
    "Mode", [0, 1], index=defaults.get("mode", 1),
    help="0 = Minor, 1 = Major"
)
time_signature = st.sidebar.selectbox(
    "Time Signature", [3, 4, 5, 6, 7],
    index=[3, 4, 5, 6, 7].index(defaults.get("time_signature", 4))
    if defaults.get("time_signature", 4) in [3, 4, 5, 6, 7] else 1,
)
acousticness = st.sidebar.slider(
    "Acousticness", 0.0, 1.0, defaults.get("acousticness", 0.3), 0.01,
    help="Confidence the track is acoustic (0=electronic, 1=acoustic)"
)
danceability = st.sidebar.slider(
    "Danceability", 0.0, 1.0, defaults.get("danceability", 0.6), 0.01,
    help="How suitable for dancing (tempo, rhythm, beat strength)"
)
energy = st.sidebar.slider(
    "Energy", 0.0, 1.0, defaults.get("energy", 0.6), 0.01,
    help="Perceptual intensity and activity"
)
instrumentalness = st.sidebar.slider(
    "Instrumentalness", 0.0, 1.0, defaults.get("instrumentalness", 0.01), 0.01,
    help="Likelihood of no vocals (0=vocals, 1=instrumental)"
)
liveness = st.sidebar.slider(
    "Liveness", 0.0, 1.0, defaults.get("liveness", 0.12), 0.01,
    help="Presence of audience (0=studio, 1=live)"
)
loudness = st.sidebar.slider(
    "Loudness (dB)", -30.0, 5.0, defaults.get("loudness", -6.0), 0.1,
    help="Overall loudness in decibels"
)
speechiness = st.sidebar.slider(
    "Speechiness", 0.0, 1.0, defaults.get("speechiness", 0.06), 0.01,
    help="Presence of spoken words (0=music, 1=speech)"
)
valence = st.sidebar.slider(
    "Valence", 0.0, 1.0, defaults.get("valence", 0.5), 0.01,
    help="Musical positiveness (0=sad/angry, 1=happy/euphoric)"
)
tempo = st.sidebar.slider(
    "Tempo (BPM)", 40.0, 220.0, defaults.get("tempo", 120.0), 1.0,
    help="Estimated tempo in beats per minute"
)

# Additional metadata
st.sidebar.divider()
st.sidebar.subheader("📊 Track Metadata")
popularity = st.sidebar.slider("Track Popularity", 0, 100, 50, 1)
explicit = st.sidebar.checkbox("Explicit Lyrics", value=False)
num_artists = st.sidebar.number_input("Num Artists", 1, 20, 1)
artist_popularity = st.sidebar.slider("Artist Popularity", 0, 100, 50, 1)

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
st.title("🎵 Multi-modal Music Genre Classification")
st.caption(
    "Predict a track's genre by combining acoustic features with lyrics analysis. "
    f"Trained on {len(predictor.genre_names)} consolidated genres."
)

# --- Lyrics Input ---
st.subheader("📝 Lyrics (optional)")
lyrics = st.text_area(
    "Paste song lyrics for NLP analysis",
    value="",
    height=150,
    placeholder="Paste song lyrics here to enable NLP features...",
    help="Lyrics help the model distinguish genres with similar sounds but different themes "
         "(e.g., Country vs Folk, Hip Hop vs Pop).",
)

if lyrics.strip():
    with st.expander("📊 Lyrics Analysis", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        word_count = len(lyrics.split())
        col1.metric("Word Count", word_count)
        col2.metric("Unique Words", len(set(lyrics.lower().split())))
        clean_text = lyrics.replace("\n", " ")
        col3.metric("Avg Word Length", f"{np.mean([len(w) for w in clean_text.split()]):.1f}" if clean_text.split() else "0")
        col4.metric("Lines", lyrics.count("\n") + 1 if lyrics else 0)

# ---------------------------------------------------------------------------
# Predict button
# ---------------------------------------------------------------------------
st.divider()
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button(
        "🔮 Predict Genre",
        type="primary",
        use_container_width=True,
    )

if predict_btn:
    # Build acoustic features dict
    acoustic_features = {
        "duration_ms": duration_ms,
        "key": key,
        "mode": mode,
        "time_signature": time_signature,
        "acousticness": acousticness,
        "danceability": danceability,
        "energy": energy,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "loudness": loudness,
        "speechiness": speechiness,
        "valence": valence,
        "tempo": tempo,
    }

    with st.spinner("Analyzing..."):
        result = predictor.predict(
            acoustic_features=acoustic_features,
            lyrics=lyrics,
            artist_popularity=artist_popularity,
            followers=0,
            popularity=popularity,
            explicit=explicit,
            num_artists=num_artists,
        )

    # --- Results ---
    st.divider()
    st.subheader("🏆 Prediction Results")

    # Main prediction
    pred_col, conf_col = st.columns([1, 2])
    with pred_col:
        st.metric(
            label="Predicted Genre",
            value=result["predicted_genre"],
            delta=f"{result['top3_genres'][0][1]:.1%} confidence",
        )

    # Top-3 probabilities
    with conf_col:
        st.write("**Top-3 Genre Probabilities**")
        for genre, prob in result["top3_genres"]:
            st.progress(prob, text=f"{genre}: {prob:.1%}")

    # All probabilities as a bar chart
    st.write("**All Genres**")
    proba_df = pd.DataFrame(
        list(result["probabilities"].items()),
        columns=["Genre", "Probability"],
    ).sort_values("Probability", ascending=True)

    st.bar_chart(proba_df.set_index("Genre"))

    # --- SHAP Explanation ---
    st.divider()
    st.subheader("🔍 Why This Prediction? (SHAP Explanation)")

    st.info(
        "SHAP values explain how each feature pushed the prediction toward "
        "or away from each genre. Positive SHAP values push toward this genre, "
        "negative push away."
    )

    # Get feature importance from the XGBoost model
    importance_df = predictor.get_feature_importance()
    if importance_df is not None:
        with st.expander("📊 Global Feature Importance (Top 20)", expanded=False):
            top20 = importance_df.head(20)

            # Horizontal bar chart
            chart_df = top20.set_index("feature").sort_values("importance", ascending=True)
            st.bar_chart(chart_df)

            st.caption(
                "These are the most important features globally across all predictions. "
                "Feature importance is computed by the XGBoost model."
            )

    # Key features driving this prediction (acoustic only, for interpretability)
    st.write("**Key Acoustic Features for This Prediction**")
    acoustic_df = pd.DataFrame(
        list(acoustic_features.items()),
        columns=["Feature", "Value"],
    )
    st.dataframe(
        acoustic_df.style.background_gradient(subset=["Value"], cmap="RdYlGn"),
        use_container_width=True,
    )
    st.caption(
        "Green = high value, Red = low value. These acoustic features "
        "are the primary signals the model uses."
    )

else:
    # --- Default state: show genre info ---
    st.info("👈 Adjust acoustic parameters in the sidebar and paste lyrics, then click **Predict Genre**.")

    st.subheader("📋 Available Genres")
    cols = st.columns(4)
    for i, genre in enumerate(predictor.genre_names):
        cols[i % 4].markdown(f"- {genre}")

    st.divider()
    st.subheader("💡 Tips")
    st.markdown("""
    - **Lyrics help the most** when distinguishing between similar-sounding genres
    - Try the **presets** in the sidebar for quick experimentation
    - The model is 8 broad genre categories consolidated from 34+ fine-grained Spotify genres
    - **SHAP explanations** show which features most influenced the prediction
    """)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "IT3190 Capstone Project — Multi-modal Music Genre Classification | "
    "Built with Streamlit, scikit-learn, XGBoost, and NLTK"
)