# Diagnostic Report & Mitigation Plan — Traditional ML Genre Classification

## Part 1: Diagnostic Findings

### 1.1 Current Performance Baseline

We have **two pipelines** with drastically different performance, and understanding why reveals the core problems.

| Pipeline | Task Type | Genres | Best F1-Macro | Best F1-Micro | Accuracy |
|---|---|---|---|---|---|
| **OLD** (`models/`) | Single-label (multi-class) | 8 consolidated | **0.559** | — | 0.582 |
| **NEW** (`outputs/models/`) | Multi-label | 11 fine-grained | **0.296** | 0.454 | 0.214 (exact match) |

> [!CAUTION]
> The new multi-label pipeline's F1-macro (0.27–0.30) is **less than half** the old single-label pipeline's (0.559). The F1-score nearly halved when we switched from single-label to multi-label AND expanded from 8 consolidated to 11 fine-grained genres.

**New pipeline detailed results (latest run, 11 genres, multi-label):**

| Model | CV F1-Macro | Test F1-Macro | Test F1-Micro | Hamming Loss | Exact Match |
|---|---|---|---|---|---|
| ClassifierChain(LR) | 0.293 ± 0.004 | 0.296 | 0.454 | 0.139 | 0.214 |
| OVR-XGBoost | 0.266 ± 0.006 | 0.272 | 0.445 | 0.108 | 0.263 |

---

### 1.2 Root Cause #1 — Severe Class Imbalance (51.4x ratio)

The class distribution is **catastrophically imbalanced** across the 11 genres:

| Genre | Tracks | % of Total | Imbalance Ratio |
|---|---|---|---|
| rock | 6,431 | 37.4% | 1.0x (majority) |
| pop | 5,168 | 30.0% | 1.2x |
| funk / soul | 4,345 | 25.2% | 1.5x |
| electronic | 2,612 | 15.2% | 2.5x |
| hip hop | 2,182 | 12.7% | 2.9x |
| folk, world, & country | 2,070 | 12.0% | 3.1x |
| jazz | 336 | 2.0% | **19.1x** |
| stage & screen | 300 | 1.7% | **21.4x** |
| blues | 195 | 1.1% | **33.0x** |
| latin | 170 | 1.0% | **37.8x** |
| reggae | 125 | 0.7% | **51.4x** |

> [!WARNING]
> **5 genres** (jazz, stage & screen, blues, latin, reggae) have a combined 1,126 tracks — just **6.5% of the dataset** — but represent **45% of the classes**. Traditional models with OVR/Chain wrappers are essentially learning to predict "rock", "pop", and "funk/soul" while ignoring these tail genres. The F1-macro metric is dragged down catastrophically by near-zero F1 on these rare classes.

---

### 1.3 Root Cause #2 — Multi-Label Complexity is Unwarranted

#### Label Distribution Per Track
| Labels per Track | Tracks | % |
|---|---|---|
| **1 label** | **11,595** | **67.3%** |
| 2 labels | 4,656 | 27.0% |
| 3 labels | 855 | 5.0% |
| 4+ labels | 112 | 0.7% |

> [!IMPORTANT]
> **67.3% of tracks have exactly 1 genre label.** The average is only 1.39 labels per track. This is a near-single-label problem being forced into a multi-label framework, which introduces massive overhead for minimal benefit.

#### Label Combination Fragmentation
- **2,048** possible label combinations (2^11)
- Only **166** unique combinations observed
- **68.7%** of observed combinations have fewer than 10 samples
- **25.9%** of observed combinations appear **exactly once**

The top 5 most common combinations account for the majority of data:

| Combination | Tracks | % |
|---|---|---|
| rock (alone) | 3,615 | 21.0% |
| funk / soul (alone) | 2,725 | 15.8% |
| pop + rock | 1,606 | 9.3% |
| folk, world, & country (alone) | 1,501 | 8.7% |
| pop (alone) | 1,351 | 7.8% |

> [!WARNING]
> Multi-label classification with OVR/Chain wrappers treats each label independently or sequentially. With 166 fragmented combinations, most having very few samples, the models cannot learn meaningful per-label decision boundaries. The "exact match accuracy" of 21–26% reflects this — the model almost never gets the *entire label vector* correct.

---

### 1.4 Root Cause #3 — Feature Engineering Issues

1. **Dimension explosion after PCA**: The fused feature matrix is (16,736 × 5,214) before PCA, reduced to (13,388 × 3,577). Having **3,577 features for only ~13K training samples** is a very unfavorable ratio for traditional models — especially Logistic Regression.

2. **TF-IDF dominates the feature space**: Audio features = 209 dims vs. Text (TF-IDF) features = 5,005 dims. After fusion and PCA, the text modality drowns out audio signals, even though genre is fundamentally an *acoustic* property.

3. **Feature importance concentration**: The top 2 features (`followers`, `artist_popularity`) account for **10.4%** of total XGBoost importance. These are **metadata features, not content features**. The model is partly learning "popular artists → pop/rock" rather than genuine audio/text patterns.

4. **Graph features are useless**: `graph_eigenvector_centrality`, `graph_degree_centrality`, `graph_betweenness_centrality`, `graph_clustering_coefficient` all have **exactly 0.0 importance**. They are dead weight.

5. **160 interaction features**: The pipeline generates exhaustive pairwise interaction features (e.g., `interact_duration_ms_danceability`). Many of these add noise rather than signal with only 13K training samples.

---

### 1.5 Root Cause #4 — "Stage & Screen" is Noise

- **300 tracks** (1.7% of dataset), with 275 co-occurring with other genres
- Only 25 tracks are "Stage & Screen" alone
- High co-occurrence with pop (69 co-occur = 23%), rock (66 = 22%), funk/soul (37 = 12%)
- This genre describes a *use-case* (film/theater soundtracks), not an acoustic/lyrical style — it's inherently unclassifiable from audio/text features alone

---

### 1.6 The Old Pipeline's "Secret": It Was Single-Label + Consolidated Genres

The old pipeline in `models/` achieved 55.9% F1-macro because:
1. It used **single-label classification** (multi-class, not multi-label)
2. It used **8 consolidated genres** that merged fine-grained categories (e.g., "blues" + "jazz" → something like "Traditional/Standards"; "folk" → "Country & Folk")
3. It had **163 features** (tight, hand-engineered), not 3,577 PCA components
4. Even so, it showed **massive overfitting**: train accuracy = 99.94% vs test accuracy = 58.2%

---

## Part 2: Mitigation Plan — Achieving >60% F1-Score

### Overview of Strategy

We will make **4 surgical changes** to the pipeline, in priority order:

1. **Drop "Stage & Screen"** — remove noise class
2. **Pivot to Single-Label Classification** — match the problem's true structure  
3. **Fix Class Imbalance** — address the 51.4x ratio in the remaining 10 genres
4. **Fix Feature Engineering** — reduce dimensionality, balance modalities, remove dead features

---

### Step 1: Drop "Stage & Screen" Genre

> [!IMPORTANT]
> This is the easiest and most uncontroversial change.

#### What to Change
- [MODIFY] [label_encoder.py](file:///d:/Workspace/hust_academic/year2/IT3190-Capstone/src2/data/label_encoder.py): Add a hardcoded exclusion list (or config parameter) that removes "stage & screen" before binarization
- [MODIFY] [config.py](file:///d:/Workspace/hust_academic/year2/IT3190-Capstone/src2/config.py): Add `EXCLUDED_GENRES: list[str] = ["stage & screen"]`

#### Impact
- 25 tracks lose all labels and are dropped → 17,193 remaining tracks
- 10 remaining genre classes
- Max/min imbalance drops marginally (51.4x → 51.4x because reggae is still the minority)

---

### Step 2: Pivot from Multi-Label to Single-Label (Multi-Class) Classification

> [!IMPORTANT]
> **This is the single highest-impact change.** Given that 67.3% of tracks already have exactly 1 label, and the average is 1.39 labels/track, multi-label adds massive complexity with minimal benefit for traditional ML.

#### Primary Genre Selection Strategy

For the 32.7% of tracks that are multi-label, we need to select a **primary genre**. The recommended approach:

**Strategy: "Rarest Label First" (most specific genre wins)**
- For each track, pick the genre with the **lowest overall frequency** in the dataset
- Rationale: A track labeled "jazz + pop" is more meaningfully a *jazz* track (since "pop" is generic/ubiquitous)
- This naturally redistributes samples toward underrepresented classes, partially alleviating imbalance

Projected distribution after single-label conversion + "Stage & Screen" removal:

| Genre | Tracks (est.) | % |
|---|---|---|
| rock | ~3,600 | ~21% |
| funk / soul | ~3,280 | ~19% |
| pop | ~2,960 | ~17% |
| electronic | ~2,150 | ~13% |
| hip hop | ~2,130 | ~12% |
| folk, world, & country | ~2,030 | ~12% |
| jazz | ~280 | ~1.6% |
| blues | ~195 | ~1.1% |
| latin | ~165 | ~1.0% |
| reggae | ~125 | ~0.7% |

> [!WARNING]
> Even with "rarest first", jazz/blues/latin/reggae remain very underrepresented. Step 3 addresses this.

#### What to Change
- [NEW] `src2/data/label_converter.py`: New module to convert multi-label → single-label using the "rarest label first" strategy
- [MODIFY] [train_baseline.py](file:///d:/Workspace/hust_academic/year2/IT3190-Capstone/src2/pipeline/train_baseline.py): Replace multi-label pipeline with multi-class pipeline
- [MODIFY] [traditional_models.py](file:///d:/Workspace/hust_academic/year2/IT3190-Capstone/src2/models/traditional_models.py): Remove OVR/Chain wrappers; use native multi-class classifiers (RF, XGBoost, LR all support multi-class natively)
- [MODIFY] [evaluation.py](file:///d:/Workspace/hust_academic/year2/IT3190-Capstone/src2/models/evaluation.py): Switch from multi-label metrics (hamming loss, exact match) to standard multi-class metrics (confusion matrix, per-class F1, classification_report)

---

### Step 3: Fix Class Imbalance

With 10 single-label classes, the imbalance is still ~51x. A multi-pronged approach:

#### 3A. Class Weights (Primary Strategy)
- Use `class_weight='balanced'` for all sklearn models (already done for RF, but NOT for XGBoost or LR)
- For XGBoost: compute `sample_weight` array using inverse class frequency
- This is lightweight and doesn't alter the data distribution

#### 3B. SMOTE Oversampling (Secondary Strategy)
- Apply SMOTE to the **training set only** (never the test set) to oversample minority classes (reggae, latin, blues, jazz) up to ~500–1,000 synthetic samples
- Use `imblearn.over_sampling.SMOTE` with `k_neighbors=5`
- Cap the oversampling ratio to avoid synthetic data dominating

#### 3C. Consider Genre Consolidation (Fallback)
If Steps 3A+3B don't reach 60%, consider merging the 4 tail genres:
- "blues" + "jazz" → **"Blues & Jazz"** (merged: ~475 samples)
- "latin" + "reggae" → **"Latin & Caribbean"** (merged: ~290 samples)
- This reduces to **8 classes** and substantially improves balance
- Only do this if the 10-class approach fails to hit 60%

#### What to Change
- [MODIFY] [traditional_models.py](file:///d:/Workspace/hust_academic/year2/IT3190-Capstone/src2/models/traditional_models.py): Add `class_weight='balanced'` and `sample_weight` support
- [MODIFY] [train_baseline.py](file:///d:/Workspace/hust_academic/year2/IT3190-Capstone/src2/pipeline/train_baseline.py): Add SMOTE step after train/test split, before model training
- [MODIFY] [config.py](file:///d:/Workspace/hust_academic/year2/IT3190-Capstone/src2/config.py): Add SMOTE configuration parameters

---

### Step 4: Fix Feature Engineering Pipeline

#### 4A. Reduce TF-IDF Dimensionality
- Current: `TFIDF_MAX_FEATURES = 5000` with `(1,2)` n-grams → 5,005 text features
- **Proposed**: Reduce to `TFIDF_MAX_FEATURES = 500` or apply SVD/TruncatedSVD directly to get ~50–100 latent text dimensions
- This rebalances the audio-to-text feature ratio from 209:5005 to roughly 209:100

#### 4B. Remove Dead Features
Remove features with 0.0 importance:
- `graph_eigenvector_centrality`, `graph_degree_centrality`, `graph_betweenness_centrality`, `graph_clustering_coefficient`, `nlp_repetition_score`

#### 4C. Reduce/Remove Interaction Features  
- The 100+ pairwise interaction features (e.g., `interact_duration_ms_danceability`) bloat the feature space
- **Option A**: Remove all interaction features and rely on tree-based models to learn interactions natively
- **Option B**: Keep only the top-20 interaction features by importance

#### 4D. Rethink PCA
- Current: PCA at 95% variance → 3,577 components (68.6% of original dims retained!)
- **Proposed**: Lower threshold to 90% or use a fixed `n_components=200–300`
- Alternatively: skip PCA entirely and use tree-based models (RF, XGBoost) that handle high-dimensional data without linear projection

#### 4E. Feature Scaling Strategy
- PCA already includes `StandardScaler` (good)
- For non-PCA path: ensure all features are scaled before Logistic Regression

#### What to Change
- [MODIFY] [config.py](file:///d:/Workspace/hust_academic/year2/IT3190-Capstone/src2/config.py): `TFIDF_MAX_FEATURES = 500`, `PCA_VARIANCE_THRESHOLD = 0.90` or `PCA_N_COMPONENTS = 200`
- [MODIFY] [text_traditional.py](file:///d:/Workspace/hust_academic/year2/IT3190-Capstone/src2/features/text_traditional.py): Add TruncatedSVD after TF-IDF, reduce to ~50–100 dims
- [MODIFY] [audio_traditional.py](file:///d:/Workspace/hust_academic/year2/IT3190-Capstone/src2/features/audio_traditional.py): Remove or reduce interaction features; remove dead graph features
- [MODIFY] [feature_reducer.py](file:///d:/Workspace/hust_academic/year2/IT3190-Capstone/src2/features/feature_reducer.py): Lower PCA threshold or fix n_components

---

### Step 5: Model Tuning & Additional Models

Once Steps 1–4 are in place, optimize the models:

#### 5A. Hyperparameter Tuning
- XGBoost: tune `max_depth` (try 6–10), `learning_rate` (0.05–0.2), `n_estimators` (300–500), `min_child_weight`, `subsample`, `colsample_bytree`
- RandomForest: tune `n_estimators` (300–500), `max_depth`, `min_samples_leaf`
- LogisticRegression: tune `C` (0.01–100), `solver` (saga for large data)

#### 5B. Add SVM Model
- `SVC(kernel='rbf', class_weight='balanced')` — strong baseline for moderate-dimensional data
- Wrap with proper scaling (StandardScaler already in PCA pipeline)

#### 5C. Add Gradient Boosting Ensemble
- LightGBM as an alternative to XGBoost (often faster, sometimes better)

#### What to Change
- [MODIFY] [traditional_models.py](file:///d:/Workspace/hust_academic/year2/IT3190-Capstone/src2/models/traditional_models.py): Add SVM builder, LightGBM builder, hyperparameter grids
- [MODIFY] [train_baseline.py](file:///d:/Workspace/hust_academic/year2/IT3190-Capstone/src2/pipeline/train_baseline.py): Add grid/random search for top-2 models

---

## Part 3: Expected Impact & Execution Order

| Step | Change | Expected F1-Macro Improvement | Effort |
|---|---|---|---|
| 1 | Drop "Stage & Screen" | +1–3% (noise removal) | Low |
| 2 | Pivot to Single-Label | **+15–25%** (eliminates multi-label fragmentation) | Medium |
| 3 | Fix Class Imbalance (weights + SMOTE) | +5–10% (rare class F1 boost) | Medium |
| 4 | Fix Features (reduce dims, balance modalities) | +3–8% (reduce curse of dimensionality) | Medium |
| 5 | Model Tuning | +2–5% (squeeze remaining performance) | Low–Medium |

**Conservative estimate**: Steps 1–4 should bring F1-macro from ~30% to **55–65%**, meeting or exceeding the 60% target.  
**Optimistic estimate**: With Step 5 tuning, **65–72%** is achievable.

---

## Part 4: Verification Plan

### Automated Tests
```bash
# After implementing changes, run the full pipeline
wsl -- bash -ic "conda activate it3190-capstone && cd /mnt/d/Workspace/hust_academic/year2/IT3190-Capstone && python -m src2.pipeline.train_baseline"
```

### Key Metrics to Track
1. **F1-Macro** ≥ 0.60 (primary target)
2. **F1-Weighted** ≥ 0.60 
3. **Per-class F1**: No class below 0.30
4. **Confusion Matrix**: Verify rare classes (reggae, latin, blues, jazz) are being predicted at all

### Manual Verification
- Generate confusion matrix visualization
- Generate per-class F1 bar chart
- Compare before/after results in a summary table
- Verify no data leakage from SMOTE (applied only to training split)

---

## Open Questions

> [!IMPORTANT]
> **Q1: Genre Consolidation Threshold** — If 10-class single-label still fails to hit 60%, should we merge tail genres (blues+jazz, latin+reggae) to get 8 classes? The old pipeline's 8-class approach worked.

> [!IMPORTANT]
> **Q2: Primary Genre Selection** — I recommend "rarest label first" for single-label conversion. An alternative is "first genre listed" (which preserves the dataset author's intent). Which do you prefer, or should we try both and compare?

> [!IMPORTANT]
> **Q3: Cache Invalidation** — Implementing these changes requires invalidating all cached artifacts in `data/processed/` and `outputs/features/`. Should I delete all caches before the re-run, or add a `--force-rebuild` flag?
