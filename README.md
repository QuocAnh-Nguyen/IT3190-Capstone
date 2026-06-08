# 🎵 Multi-Modal Music Genre Classification

**An end-to-end machine learning pipeline that predicts a track's genre by fusing acoustic features, lyrical semantics, artist collaboration networks, and unsupervised acoustic clusters.**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/xgboost-2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-explainability-red.svg)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Running the Main Notebook](#-running-the-main-notebook)
- [Dataset](#-dataset)
- [Pipeline Architecture](#-pipeline-architecture)
- [Key Results](#-key-results)
- [Model Explainability (SHAP)](#-model-explainability-shap)
- [Scripts & Modules](#-scripts--modules)
- [Contributing](#-contributing)

---

## 🎯 Overview

This project implements a **multi-modal music genre classification system** using the [MusicOSet](https://marianaossilva.github.io/DSW2019/index.html) academic dataset. Unlike traditional approaches that rely solely on acoustic features, our pipeline exploits multiple complementary data modalities:

| Modality | Features Extracted | Rationale |
|----------|-------------------|-----------|
| **Acoustic** | 13 Spotify audio descriptors (energy, valence, tempo, etc.) | Low-level audio signal properties |
| **Network** | 4 graph centrality metrics from artist collaboration graph | Captures genre homophily — similar artists collaborate within genres |
| **NLP / Lyrics** | 10 lexical & sentiment features from song lyrics | Semantic and emotional content encoded in text |
| **Interaction** | 15 polynomial cross-terms of key acoustic features | Non-linear genre boundaries (e.g., high energy × high danceability → EDM) |
| **Clustering** | K-Means acoustic cluster assignment | Data-driven acoustic groupings independent of human labels |

**Final feature matrix**: 50 engineered features across 7,710 songs spanning 35 genres.

---

## 📁 Project Structure

```
.
├── README.md                          # ← You are here
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Project metadata & build config
├── .gitignore                         # Git ignore rules (CSVs, models, etc.)
│
├── docs/
│   ├── PRD.md                         # Product Requirements Document
│   └── musicoset_schema.sql           # Original MusicOSet DB schema
│
├── data/                               # ⚠️ NOT tracked in git (see .gitignore)
│   └── raw/                           # Raw MusicOSet CSVs — must download separately
│       ├── musicoset_metadata/        # songs, artists, albums, tracks, releases
│       ├── musicoset_songfeatures/    # acoustic_features, lyrics
│       ├── musicoset_popularity/      # song/artist/album chart & popularity
│       └── additional/                # hits_dataset, nonhits_dataset (pre-merged)
│
├── notebooks/
│   └── 01_music_genre_classification.ipynb   # ⭐ MAIN NOTEBOOK — full pipeline
│
├── src/
│   ├── config.py                      # Centralized paths, constants, model defaults
│   ├── data/                          # Data loading & preprocessing modules
│   ├── features/                      # Feature engineering (NLP, graph, interaction)
│   ├── models/                        # Model training, evaluation, CV
│   ├── deployment/                    # Model export, serialization
│   ├── ui/                            # Streamlit/Gradio app code
│   └── utils/                         # Helpers, logging, metrics
│
├── scripts/
│   ├── run_phase1.py                  # Phase 1: Data loading & preprocessing
│   ├── run_phase2.py                  # Phase 2: Feature engineering
│   └── run_phase3*.py                 # Phase 3: Model training & evaluation
│
└── models/                            # Serialized models & artifacts
    ├── best_model.joblib
    ├── scaler.joblib
    ├── label_encoder.joblib
    └── ...
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **pip** (or conda/mamba)
- **Jupyter Notebook** or **VS Code** (recommended for viewing `.ipynb`)

### 1. Clone & navigate

```bash
git clone <repo-url>
cd IT3190-Capstone
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The key packages are:

| Package | Version | Purpose |
|---------|---------|---------|
| `pandas`, `numpy` | ≥2.0 | Data manipulation |
| `scikit-learn` | ≥1.3 | ML models, preprocessing, metrics |
| `xgboost` | ≥2.0 | Gradient boosting classifier |
| `shap` | ≥0.42 | Model explainability |
| `imbalanced-learn` | ≥0.12 | SMOTE for class imbalance |
| `networkx` | ≥3.0 | Artist collaboration graph |
| `textblob` | ≥0.17 | Sentiment analysis on lyrics |
| `matplotlib`, `seaborn` | ≥3.7 | Static visualizations |
| `plotly` | latest | Interactive 3D and network charts |
| `streamlit` | ≥1.28 | Web app deployment (Phase 4) |

### 3. Obtain the dataset

> ⚠️ **The data is NOT included in this repository.** The CSV files are listed in `.gitignore` and must be downloaded separately. Follow the steps below to set up the data directory.

#### Step 3a: Download MusicOSet

The dataset is available from the official MusicOSet repository:

```bash
# Option 1: Download from Kaggle
# Visit: https://www.kaggle.com/datasets/yamqwe/musicoset
# Download the archive and extract it

# Option 2: Download from the academic source
# Visit: https://marianaossilva.github.io/DSW2019/index.html
# Navigate to the "Downloads" section
```

#### Step 3b: Create the data directory structure

```bash
mkdir -p data/raw/musicoset_metadata
mkdir -p data/raw/musicoset_songfeatures
mkdir -p data/raw/musicoset_popularity
mkdir -p data/raw/additional
```

#### Step 3c: Place the CSV files

Copy the downloaded files into the correct subdirectories. After setup, your `data/raw/` should look like this:

```
data/raw/
├── musicoset_metadata/
│   ├── songs.csv              # 20,405 rows × 7 cols
│   ├── artists.csv            # 11,518 rows × 8 cols
│   ├── albums.csv             # 26,519 rows × 8 cols
│   ├── tracks.csv             # 20,405 rows × 5 cols
│   └── releases.csv           # 26,522 rows × 4 cols
│
├── musicoset_songfeatures/
│   ├── acoustic_features.csv  # 20,405 rows × 14 cols
│   └── lyrics.csv             # 20,404 rows × 2 cols
│
├── musicoset_popularity/
│   ├── song_chart.csv         # Billboard weekly chart data
│   ├── song_pop.csv           # Year-end popularity scores
│   ├── artist_chart.csv
│   ├── artist_pop.csv
│   ├── album_chart.csv
│   └── album_pop.csv
│
└── additional/
    ├── hits_dataset.csv       # 11,959 pre-merged songs (songs + acoustic features)
    └── nonhits_dataset.csv    # 899,068 rows (full pre-merged dataset)
```

> **Note**: All CSV files use **tab** (`\t`) as the delimiter. The notebook reads them with `pd.read_csv(path, sep='\t')`.

#### Step 3d: Verify your setup

Run this quick check to confirm all required files are present:

```bash
python3 -c "
import os
required = [
    'data/raw/musicoset_metadata/songs.csv',
    'data/raw/musicoset_metadata/artists.csv',
    'data/raw/musicoset_songfeatures/acoustic_features.csv',
    'data/raw/musicoset_songfeatures/lyrics.csv',
    'data/raw/additional/hits_dataset.csv',
]
missing = [f for f in required if not os.path.exists(f)]
if missing:
    print(f'MISSING {len(missing)} files:')
    for f in missing: print(f'  ✗ {f}')
else:
    print('All required files present! You are ready to run the notebook.')
"
```

---

## 📓 Running the Main Notebook

### Option A: Open in VS Code (Recommended)

```bash
code notebooks/01_music_genre_classification.ipynb
```

Then click **"Run All"** in the notebook toolbar. The notebook has been pre-executed and contains all cell outputs (charts, tables, logs).

### Option B: Jupyter Lab / Notebook

```bash
jupyter lab notebooks/01_music_genre_classification.ipynb
```

### Option C: Execute headlessly

```bash
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/01_music_genre_classification.ipynb
```

### What the notebook does

The notebook runs a **complete Kaggle-style pipeline** across 97 cells in four phases:

| Phase | Duration (approx.) | What happens |
|-------|-------------------|--------------|
| **Phase 1** — Data Engineering | ~30 sec | Loads & merges MusicOSet tables, imputes missing values, visualizes class imbalance |
| **Phase 2** — Feature Engineering | ~60 sec | Builds collaboration graph, extracts NLP features, creates polynomial interactions, runs K-Means clustering |
| **Phase 3** — Model Building | ~5 min | Trains Logistic Regression, Random Forest, XGBoost, and MLP; runs 3-Fold CV; generates confusion matrices |
| **Phase 4** — Explainability | ~2 min | Computes SHAP values; generates summary, beeswarm, and waterfall plots |

> ⚠️ **Memory note**: The full pipeline requires ~4-8 GB RAM. If you encounter memory issues, reduce `n_estimators` in the model cells or run cells incrementally.

---

## 📊 Dataset

We use the **MusicOSet** dataset, an open-source academic resource containing:

- **20,405 songs** with Spotify acoustic features
- **11,518 artists** with genre labels across 992 unique genres
- **20,000+ lyrics** for NLP analysis
- **Billboard chart data** and popularity metrics

After filtering to genres with ≥50 samples, our working dataset is **7,710 songs across 35 genres**.

**Top genres**: dance pop (1,333), contemporary country (1,139), adult standards (728), album rock (479), brill building pop (464)

---

## 🔬 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA INGESTION                           │
│  songs.csv ─┬─ artists.csv ──→ Genre Labels                 │
│             ├─ acoustic_features.csv ──→ 13 Audio Features  │
│             └─ lyrics.csv ──→ Raw Lyrics Text               │
└─────────────────────────┬───────────────────────────────────┘
                          │
            ┌─────────────▼─────────────┐
            │    FEATURE ENGINEERING    │
            │                          │
            │  • NetworkX Graph ──→ 4   │
            │    Centrality Metrics     │
            │  • TextBlob ──→ 10 NLP   │
            │    Features               │
            │  • PolynomialFeatures ──→ │
            │    15 Interaction Terms   │
            │  • K-Means ──→ Cluster   │
            │    Assignment             │
            └─────────────┬─────────────┘
                          │
            ┌─────────────▼─────────────┐
            │   PREPROCESSING           │
            │  • StandardScaler         │
            │  • SMOTE (35 → 1,067/cls) │
            │  • Train/Test Split (8:2) │
            └─────────────┬─────────────┘
                          │
            ┌─────────────▼─────────────┐
            │   MODEL TRAINING          │
            │  • Logistic Regression    │
            │  • Random Forest          │
            │  • XGBoost ★ (best)       │
            │  • MLP Neural Network     │
            │  • 3-Fold CV              │
            └─────────────┬─────────────┘
                          │
            ┌─────────────▼─────────────┐
            │   EXPLAINABILITY          │
            │  • SHAP Global Importance │
            │  • SHAP Beeswarm          │
            │  • SHAP Waterfall         │
            │  • Per-Prediction Force   │
            └───────────────────────────┘
```

---

## 🏆 Key Results

### Model Performance (35-class classification)

| Model | Accuracy | Macro F1 | Weighted F1 | 3-Fold CV F1 |
|-------|----------|----------|-------------|-------------|
| **Logistic Regression** | 0.3275 | 0.2979 | 0.3532 | 0.3559 ± 0.0072 |
| **MLP Neural Network** | 0.4345 | 0.3254 | 0.4256 | 0.3961 ± 0.0093 |
| **Random Forest** | 0.5629 | 0.4894 | 0.5634 | 0.5023 ± 0.0065 |
| 🥇 **XGBoost** | **0.6206** | **0.5140** | **0.6116** | **0.5671 ± 0.0031** |

### Top performing genres (XGBoost)

| Genre | F1-Score | Support |
|-------|----------|---------|
| canadian hip hop | 1.000 | 17 |
| hollywood | 0.919 | 20 |
| detroit hip hop | 0.909 | 12 |
| hip hop | 0.857 | 26 |
| pop | 0.852 | 30 |
| dance pop | 0.814 | 266 |

### Key findings

1. **Multi-modality matters**: Adding network, NLP, and interaction features to acoustic-only baselines consistently improves F1 by 5-15 percentage points.
2. **Network centrality** is a surprisingly strong signal — well-connected artists cluster within genre communities.
3. **XGBoost dominates** tree-based and neural approaches for this tabular, multi-class problem.
4. **Confusion patterns** reveal acoustically similar pairs (e.g., "dance pop" ↔ "electropop", "classic soul" ↔ "chicago soul") that even the best model struggles with.

---

## 🔍 Model Explainability (SHAP)

We use **SHAP (SHapley Additive exPlanations)** to explain why the model makes each prediction. The notebook generates:

| Plot | Description |
|------|-------------|
| **Summary Bar** | Global feature importance across all predictions |
| **Beeswarm** | Feature impact direction (red = pushes prediction up, blue = down) |
| **Waterfall** | Per-prediction decomposition starting from the expected value |
| **Force Plot** | Alternative per-prediction view (ideal for dashboard integration) |

**Top SHAP features**: `acousticness`, `loudness`, `speechiness`, `energy`, `degree_centrality`, and NLP `lexical_richness` — confirming that **all modalities** contribute meaningful signal.

Example waterfall explanation for a misclassified track:

```
True: classic uk pop  →  Predicted: chicago soul
Top positive drivers:  mode (+0.83), lexical_richness (+0.37), loudness (+0.35)
Top negative drivers:  speechiness (-1.58), betweenness_centrality (-0.88)
```

---

## 🔧 Scripts & Modules

### Standalone scripts (`scripts/`)

Run individual phases independently:

```bash
python scripts/run_phase1.py    # Data loading, merging, imputation
python scripts/run_phase2.py    # Feature engineering (graph, NLP, interaction, K-Means)
python scripts/run_phase3.py    # Model training, CV, evaluation
```

### Python source modules (`src/`)

For integration into the Streamlit app or production pipelines:

```python
from src.config import ACOUSTIC_FEATURE_COLS, BEST_MODEL_PATH, RANDOM_SEED
from src.features import build_network_features, extract_lyrics_features
from src.models import train_xgboost, evaluate_model
```

### Notebook helper scripts (`notebooks/`)

- `run_pipeline.py` — Executes all notebook code cells as a standalone Python script
- `inject_outputs.py` — Injects captured outputs (PNG/HTML/text) into the `.ipynb` file
- `fix_shap.py` — Re-runs SHAP cells with corrected multi-class API handling

---

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make changes and verify the notebook still executes cleanly
4. Submit a pull request

### Code style

- Follow existing patterns in `src/config.py` for constants and paths
- Use the centralized `RANDOM_SEED = 42` for reproducibility
- Prefer `pathlib.Path` over raw strings for file paths

---

## 📚 References

- [MusicOSet Dataset](https://marianaossilva.github.io/DSW2019/index.html) — Mariana O. Silva et al.
- [SHAP Documentation](https://shap.readthedocs.io/) — Lundberg & Lee, 2017
- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754) — Chen & Guestrin, 2016
- [NetworkX Documentation](https://networkx.org/)

---

*Built as part of the IT3190 Capstone Project — HUST, Academic Year 2 | June 2026*