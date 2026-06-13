# Multi-Modal Music Genre Classification — Execution Plan

## Project Overview

Build an end-to-end multi-label genre classification pipeline that fuses **raw audio processing** with **NLP semantic analysis of lyrics**. The pipeline explicitly compares traditional ML feature extraction against pre-trained deep learning embeddings, using a Late Fusion architecture.

All new code will be written in `src2/`. The legacy `src-old/` directory is strictly off-limits.

---

## Workspace Context (Discovered)

| Asset | Location | Key Details |
|---|---|---|
| Songs metadata | [songs.csv](file:///d:/Workspace/hust_academic/year2/IT3190-Capstone/data/raw/musicoset_metadata/songs.csv) | **~20,405 tracks**, tab-separated, has `genres` column (multi-label, `;`-delimited e.g. `"Electronic; Pop"`) |
| Artists metadata | [artists.csv](file:///d:/Workspace/hust_academic/year2/IT3190-Capstone/data/raw/musicoset_metadata/artists.csv) | Tab-separated, has `main_genre` and `genres` (fine-grained list) |
| Lyrics | [lyrics.csv](file:///d:/Workspace/hust_academic/year2/IT3190-Capstone/data/raw/musicoset_songfeatures/lyrics.csv) | ~149 MB, tab-separated, `song_id` → `lyrics` (full song lyrics text) |
| Audio files | [audio_previews/](file:///d:/Workspace/hust_academic/year2/IT3190-Capstone/audio_previews) | **19,897 MP3 preview files** (~30s each), named `{song_id}.mp3` |
| Audio metadata | [audio_metadata.csv](file:///d:/Workspace/hust_academic/year2/IT3190-Capstone/data/processed/audio_metadata.csv) | Comma-separated mapping: `song_id` → `genre_fine`, `genre_consolidated`, `mp3_path` |
| SQL database | [musicoset.sql](file:///d:/Workspace/hust_academic/year2/IT3190-Capstone/musicoset.sql) | ~245 MB dump with all tables including `lyrics`, `songs`, `artists`, `audio_metadata` |
| DB schema | [musicoset_schema.sql](file:///d:/Workspace/hust_academic/year2/IT3190-Capstone/docs/musicoset_schema.sql) | 12 tables total |
| PRD | [PRD.md](file:///d:/Workspace/hust_academic/year2/IT3190-Capstone/docs/PRD.md) | 3 phases: Data Engineering, Feature Engineering, Model Building |

### Genre Labels
- The `songs.csv` `genres` column contains **high-level multi-label genre tags** (`;`-delimited): e.g., `"Electronic; Pop"`, `"Hip-Hop; R&B"`
- The `audio_metadata.csv` contains `genre_fine` (e.g., `"album rock"`) and `genre_consolidated` (e.g., `"Rock"`) — artist-level genre mappings
- **We will use the `songs.csv` `genres` column as ground truth** for multi-label classification, as it directly maps genre labels to individual songs

---

## Execution Strategy

> [!IMPORTANT]
> **Traditional Methods** (librosa, TF-IDF, scikit-learn, XGBoost) → **Fully implemented AND executed** to establish a working baseline with concrete metrics.
>
> **Pre-trained Deep Learning Methods** (BERT, AST/VGGish/Wav2Vec) → **Code written only**, execution deferred to GPU-capable environment.

---

## Proposed `src2/` Directory Structure

```
src2/
├── config.py                  # Paths, constants, hyperparameter defaults
├── data/
│   ├── __init__.py
│   ├── data_loader.py         # Load & merge CSVs, SQL queries
│   ├── data_cleaner.py        # Missing data handling, filtering
│   └── label_encoder.py       # Multi-label binarization, class analysis
├── features/
│   ├── __init__.py
│   ├── audio_traditional.py   # librosa feature extraction
│   ├── audio_deeplearning.py  # AST/VGGish/Wav2Vec embeddings (code-only)
│   ├── text_traditional.py    # TF-IDF, BoW, lexical richness, sentiment
│   ├── text_deeplearning.py   # BERT/RoBERTa embeddings (code-only)
│   └── feature_reducer.py     # PCA, feature importance, dimensionality reduction
├── models/
│   ├── __init__.py
│   ├── traditional_models.py  # RF, XGBoost, Classifier Chains
│   ├── neural_models.py       # MLP fusion network (code-only for DL embeddings)
│   └── evaluation.py          # Metrics, cross-validation, confusion matrix
├── pipeline/
│   ├── __init__.py
│   ├── train_baseline.py      # End-to-end training for traditional pipeline
│   └── train_deep.py          # End-to-end training for DL pipeline (code-only)
└── utils/
    ├── __init__.py
    ├── logging_utils.py       # Structured logging
    └── io_utils.py            # Save/load features, models, results
```

---

## Phase 1: Data Engineering & Preprocessing

### Milestone 1.1 — Project Scaffolding
- [ ] Create the `src2/` directory structure and all `__init__.py` files
- [ ] Create `config.py` with all path constants (audio dir, CSV paths, output dirs), random seed, and configurable parameters
- [ ] Set up structured logging in `logging_utils.py`
- [ ] Create `io_utils.py` with helper functions for saving/loading pickle, CSV, and model artifacts

### Milestone 1.2 — Data Loading & Integration
- [ ] **`data_loader.py`**: Build functions to load the core CSV files from `data/raw/`:
  - Load `songs.csv` (tab-separated) — the primary track table with `song_id`, `song_name`, `genres`
  - Load `lyrics.csv` (tab-separated) — keyed by `song_id`
  - Load `audio_metadata.csv` (comma-separated, from `data/processed/`) — maps `song_id` to `mp3_path`, `genre_fine`, `genre_consolidated`
  - Load `artists.csv` (tab-separated) — for artist-level genre enrichment if needed
- [ ] **Merge Strategy**: Join `songs` ↔ `lyrics` ↔ `audio_metadata` on `song_id` to produce a unified dataframe with columns: `song_id`, `genres` (multi-label), `lyrics`, `mp3_path`
- [ ] **Audio File Validation**: Cross-reference merged records against files that actually exist in `audio_previews/` — only retain tracks with valid, non-corrupt MP3 files

### Milestone 1.3 — Missing Data Handling
- [ ] **`data_cleaner.py`**: Implement the missing data strategy from the PRD:
  - Identify tracks missing lyrics, missing audio, or missing genre labels
  - Log comprehensive statistics on data availability per modality
  - **Decision point**: Drop records missing the target variable (genres). For records missing one modality (audio XOR lyrics), flag them but retain — the model should be able to handle single-modality prediction via masking
  - Filter out tracks with empty/null `genres`, or genres that appear below a minimum frequency threshold (to avoid ultra-rare single-occurrence labels)

### Milestone 1.4 — Multi-Label Encoding & Class Imbalance Analysis
- [ ] **`label_encoder.py`**:
  - Parse the `;`-delimited `genres` string into a list of genre labels per track
  - Use `sklearn.preprocessing.MultiLabelBinarizer` to produce the binary label matrix `Y`
  - Analyze and report class distribution: count per genre, co-occurrence heatmap
  - Identify the set of target genre classes (likely filtering to genres with sufficient representation)
- [ ] **Class Imbalance Strategy**:
  - Compute per-class sample weights for use in loss functions
  - Explore MLSMOTE (from `imblearn`) or class-weighted loss as per PRD
  - Store the fitted `MultiLabelBinarizer` for inverse-transform during evaluation

---

## Phase 2: Advanced Feature Engineering

### Milestone 2.1 — Traditional Audio Feature Extraction (EXECUTE)
- [ ] **`audio_traditional.py`**: Using `librosa`, extract the following features from each MP3 file:
  - **MFCCs** (Mel-Frequency Cepstral Coefficients) — mean and std across time frames
  - **Mel-Spectrogram** statistics — mean energy per mel band
  - **Chroma Features** — pitch class profiles, mean and std
  - **Spectral Contrast** — mean across bands
  - **Additional**: Zero Crossing Rate, Spectral Centroid, Spectral Bandwidth, Spectral Rolloff, RMS Energy — all summarized as mean/std
- [ ] Implement batch processing with progress bars (`tqdm`), error handling for corrupt files, and intermediate checkpoint saving (write partial results to disk periodically)
- [ ] Output: A feature matrix (DataFrame/array) with one row per track, indexed by `song_id`

> [!NOTE]
> Audio previews are ~30s clips. Feature extraction over ~19,897 files will take significant time. Implement chunked processing with resume capability.

### Milestone 2.2 — Traditional NLP Feature Extraction (EXECUTE)
- [ ] **`text_traditional.py`**: Process lyrics text through:
  - **Text Preprocessing**: Lowercase, remove section headers (e.g., `[Verse 1]`, `[Chorus]`), strip special characters, tokenize
  - **TF-IDF Vectorization**: Fit a TF-IDF vectorizer on the corpus with configurable max features and n-gram range
  - **Bag of Words**: As an alternative sparse representation
  - **Lexical Richness Metrics**: Type-Token Ratio (TTR), Hapax Legomena ratio, vocabulary size per song
  - **Sentiment Analysis**: Using `TextBlob` or `NLTK`'s VADER — polarity, subjectivity scores
- [ ] Output: A combined feature matrix (TF-IDF matrix + lexical/sentiment scalar features) per track

### Milestone 2.3 — Pre-trained Audio Embeddings (CODE ONLY — DO NOT EXECUTE)
- [ ] **`audio_deeplearning.py`**: Write implementation for:
  - Loading a pre-trained audio encoder (AST, VGGish, or Wav2Vec 2.0) from HuggingFace / `transformers`
  - Audio preprocessing pipeline (resampling, chunking to model's expected input length)
  - Forward pass through the encoder to extract dense embedding vectors
  - Batched inference with GPU support
- [ ] Output shape: `(n_tracks, embedding_dim)` — e.g., 768-d for AST
- [ ] **This code will NOT be executed** during the baseline phase — tag it clearly with docstrings noting deferred execution

### Milestone 2.4 — Pre-trained Text Embeddings (CODE ONLY — DO NOT EXECUTE)
- [ ] **`text_deeplearning.py`**: Write implementation for:
  - Loading a pre-trained text encoder (BERT or RoBERTa) from HuggingFace
  - Lyrics tokenization with the model's tokenizer (handling max sequence length, truncation)
  - Forward pass to extract `[CLS]` token embeddings or mean-pooled hidden states
  - Batched inference with GPU support
- [ ] Output shape: `(n_tracks, 768)` for BERT-base
- [ ] **This code will NOT be executed** during the baseline phase

### Milestone 2.5 — Feature Selection & Dimensionality Reduction (EXECUTE)
- [ ] **`feature_reducer.py`**: Apply to traditional features:
  - **PCA**: Fit on the concatenated audio + text feature matrix, determine optimal number of components via explained variance ratio
  - **Tree-based Feature Importance**: Use a quick Random Forest to rank features, optionally prune low-importance features
  - Store the fitted PCA/selector transformers for use at inference time
- [ ] Output: A reduced-dimensionality feature matrix ready for model training

---

## Phase 3: Model Building & Evaluation

### Milestone 3.1 — Evaluation Framework
- [ ] **`evaluation.py`**: Implement comprehensive multi-label evaluation:
  - **Metrics**: Macro F1-Score, Micro F1-Score, Weighted F1-Score, Hamming Loss, Exact Match Ratio (Subset Accuracy), per-class Precision/Recall
  - **Cross-Validation**: K-Fold CV adapted for multi-label (stratified where possible, using `iterative-stratification` or manual stratification)
  - **Multi-label Confusion Matrix**: Per-class confusion matrix visualization
  - **Results Logging**: Save all metrics to a structured JSON/CSV report

### Milestone 3.2 — Traditional ML Baseline Models (EXECUTE)
- [ ] **`traditional_models.py`**: Implement and train the following multi-label classifiers:
  - **OneVsRest + Random Forest**: Wrap sklearn's `RandomForestClassifier` in `OneVsRestClassifier`
  - **Classifier Chains**: Using `ClassifierChain` with a base estimator
  - **XGBoost Multi-label**: XGBoost configured for multi-label output (either via `OneVsRest` wrapper or native multi-output)
  - Each model trained on the fused traditional feature matrix (audio + text features, post-dimensionality-reduction)
- [ ] **Late Fusion Implementation**:
  - Concatenate the audio feature vector and text feature vector per track into a single unified representation
  - Feed the concatenated vector into each classifier
- [ ] Run K-Fold Cross-Validation for each model and log all evaluation metrics
- [ ] Save trained models and results

### Milestone 3.3 — Neural Network Fusion Model (EXECUTE for traditional features, CODE ONLY for DL embeddings)
- [ ] **`neural_models.py`**: Implement an MLP fusion network:
  - **Architecture**: Two input branches (audio stream, text stream), each with dense layers → concatenation → shared dense layers → sigmoid output layer
  - **Loss**: Binary Cross-Entropy (BCE)
  - **Framework**: PyTorch or scikit-learn `MLPClassifier` for the baseline
  - For the traditional feature inputs: **fully train and evaluate**
  - For the DL embedding inputs: **write the code path but do not execute**
- [ ] Apply class weights in the BCE loss to address imbalance

### Milestone 3.4 — Training Pipeline Orchestration
- [ ] **`train_baseline.py`**: End-to-end orchestration script that:
  1. Loads the cleaned, merged dataset
  2. Encodes multi-labels
  3. Extracts traditional audio features (or loads from cache)
  4. Extracts traditional text features (or loads from cache)
  5. Applies dimensionality reduction
  6. Performs train/test split (with multi-label stratification)
  7. Trains all traditional models
  8. Evaluates and generates comparison report
  9. Saves all artifacts (models, metrics, plots)
- [ ] **`train_deep.py`**: Equivalent pipeline for DL embeddings — **code written, not executed**

---

## Phase Summary & Execution Matrix

| Component | Code Written | Executed | Output |
|---|:---:|:---:|---|
| Data loading & merging | ✅ | ✅ | Unified DataFrame |
| Missing data handling | ✅ | ✅ | Cleaned dataset |
| Multi-label encoding | ✅ | ✅ | Binary label matrix |
| Class imbalance analysis | ✅ | ✅ | Distribution report |
| Audio features (librosa) | ✅ | ✅ | Feature matrix |
| Text features (TF-IDF) | ✅ | ✅ | Feature matrix |
| Audio embeddings (AST) | ✅ | ❌ | — |
| Text embeddings (BERT) | ✅ | ❌ | — |
| PCA / Feature reduction | ✅ | ✅ | Reduced feature matrix |
| Random Forest multi-label | ✅ | ✅ | Trained model + metrics |
| Classifier Chains | ✅ | ✅ | Trained model + metrics |
| XGBoost multi-label | ✅ | ✅ | Trained model + metrics |
| MLP fusion (traditional) | ✅ | ✅ | Trained model + metrics |
| MLP fusion (DL embeddings) | ✅ | ❌ | — |
| Evaluation framework | ✅ | ✅ | Comparison report |

---

## Verification Plan

### Automated Tests
- Run `train_baseline.py` end-to-end and verify it produces:
  - Saved model artifacts in `models/`
  - Metrics JSON/CSV in `outputs/`
  - No runtime errors across the full pipeline
- Verify that all metric values are within reasonable ranges (e.g., F1 > 0, Hamming Loss < 1)

### Manual Verification
- Review the class distribution report to ensure label encoding is correct
- Spot-check a sample of audio feature extractions against manual librosa calls
- Inspect the generated confusion matrices and metric comparisons
- Verify the DL code modules import cleanly and have correct function signatures (even though they won't be executed)

---

## Open Questions

> [!IMPORTANT]
> **Genre Granularity**: The dataset has two levels of genre labels:
> - `songs.csv` → `genres` column: coarse multi-label (e.g., `"Electronic; Pop"`, `"Hip-Hop; R&B"`) — directly on songs
> - `audio_metadata.csv` → `genre_fine` / `genre_consolidated`: artist-level genre (e.g., `"dance pop"` / `"Pop"`)
>
> **The plan assumes we use `songs.csv` `genres` as the multi-label target.** Should we instead use `genre_consolidated` from `audio_metadata.csv`, or merge both sources?

> [!IMPORTANT]
> **Minimum Genre Frequency Threshold**: Some genres may appear very rarely. Should we set a minimum count (e.g., at least 50 or 100 tracks per genre) before including a genre as a target label? This affects the number of output classes.

> [!WARNING]
> **Audio Feature Extraction Time**: Processing ~19,897 MP3 files with librosa will take considerable time (estimated 2-6 hours depending on hardware). The plan includes checkpoint/resume logic, but be aware this is the most time-intensive step. Should we start with a smaller sample (e.g., 5,000 tracks) for rapid iteration, then scale to the full dataset?

> [!NOTE]
> **Neural Framework for MLP**: For the MLP fusion model on traditional features, we can use either:
> - `sklearn.neural_network.MLPClassifier` (simpler, CPU-only, limited customization)
> - PyTorch (more flexible, GPU-capable, needed anyway for the DL embedding path)
>
> The plan defaults to PyTorch for consistency with the DL code paths. Is this acceptable, or do you prefer sklearn for the baseline?
