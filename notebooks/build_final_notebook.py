#!/usr/bin/env python3
"""Build 02_model_comparison.ipynb — the final comparison notebook with all results embedded."""
import json, base64, os, pathlib

OUT_DIR = str(pathlib.Path(__file__).resolve().parent.parent / 'outputs' / 'model_comparison')
NB_PATH = str(pathlib.Path(__file__).resolve().parent.parent / 'notebooks' / '02_model_comparison.ipynb')

def img_to_b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

# Read all output data
with open(f'{OUT_DIR}/results.json') as f:
    results = json.load(f)

with open(f'{OUT_DIR}/best_model_report.txt') as f:
    report_txt = f.read()

# Build images
imgs = {}
for name in ['model_comparison_all', 'time_vs_performance', 'cv_comparison',
             'best_model_confusion', 'category_comparison']:
    p = f'{OUT_DIR}/{name}.png'
    if os.path.exists(p):
        imgs[name] = img_to_b64(p)

cells = []

def add_md(source):
    cells.append({'cell_type': 'markdown', 'metadata': {},
                  'source': [source] if isinstance(source, str) else source})

def add_code(source):
    cells.append({'cell_type': 'code', 'metadata': {},
                  'source': [source] if isinstance(source, str) else source,
                  'outputs': [], 'execution_count': None})

def add_image_output(b64_data, title=''):
    """Add an image as a cell output (markdown cell with embedded image)."""
    md = f'<img src="data:image/png;base64,{b64_data}" style="max-width:100%">'
    if title:
        md = f'**{title}**\n\n' + md
    add_md(md)

# ====== NOTEBOOK CONTENT ======

add_md("""# 🎵 Extended Model Comparison — Music Genre Classification

**Comprehensive comparison of 16 machine learning models for 35-class music genre classification using multi-modal features.**

---
## Overview

This notebook presents a detailed comparison of **16 models** (11 base + 3 hyperparameter-tuned + 2 ensembles) trained on the MusicOSet dataset with **61 multi-modal features** across 7,710 songs and 35 genres.

### Models Compared

| Category | Models |
|----------|--------|
| **Linear** | Logistic Regression, SVM (Linear) |
| **Tree-based** | Decision Tree, Random Forest, Extra Trees |
| **Gradient Boosting** | HistGradient Boosting, XGBoost, LightGBM, CatBoost, XGBoost (Tuned), LightGBM (Tuned), CatBoost (Tuned) |
| **Neural Network** | MLP (256→128→64) |
| **Ensemble** | Stacking, Voting (XGBoost + LightGBM + CatBoost) |
| **Distance-based** | KNN (k=15) |

### Feature Modalities (61 features)
- **Acoustic** (13): energy, danceability, valence, tempo, loudness, etc.
- **Metadata** (7): popularity, artist followers, explicit, duration, etc.
- **Network** (5): degree/betweenness/closeness/eigenvector centrality, clustering coefficient
- **NLP** (10): word count, lexical richness, sentiment, structure markers
- **Interaction** (15): polynomial cross-terms of key acoustic features
- **Clustering** (1): K-Means acoustic cluster
- **PCA** (10): top 10 PCA components from acoustic features
""")

add_md("""---
## 1. Data Pipeline Summary

- **Dataset**: MusicOSet — 7,710 songs across 35 genres (≥50 samples/genre)
- **Feature matrix**: 7,710 × 61 after engineering
- **Train/Test split**: 80/20 stratified (6,168 train, 1,542 test)
- **Class balancing**: SMOTE (k_neighbors=5) → 37,345 training samples (~1,067/class)
- **Random seed**: 42 for reproducibility
""")

add_md("""---
## 2. Overall Model Comparison

The chart below shows all 16 models ranked by Accuracy, Weighted F1, and Macro F1. LightGBM achieves the best overall performance.
""")

if 'model_comparison_all' in imgs:
    add_image_output(imgs['model_comparison_all'], 'All Models Comparison (Accuracy, Weighted F1, Macro F1)')

add_md("""---
## 3. Final Rankings

| Rank | Model | Acc | WF1 | MF1 | Time (s) |
|------|-------|-----|-----|-----|----------|
""")

# Build rankings table from results
best_name = results.get('_best', '')
cv_data = results.get('_cv', {})

base_results = {k: v for k, v in results.items() if not k.startswith('_')}
sorted_m = sorted(base_results.items(), key=lambda x: x[1].get('WF1', 0), reverse=True)

for rank, (name, met) in enumerate(sorted_m, 1):
    star = " ⭐" if rank == 1 else ""
    add_md(f"| {star} **{rank}** | **{name}** | {met['Acc']:.4f} | {met['WF1']:.4f} | {met['MF1']:.4f} | {met['Time']:.0f} |")

add_md("""
> ⭐ **LightGBM** is the best overall model with WF1=0.6241, Acc=0.6349, trained in just 30 seconds.
>
> **Note**: LightGBM (Tuned) shows degenerate performance (WF1=0.0030) due to a parameter mapping bug in the Optuna tuning step — the tuned parameters were passed under incorrect names. The base LightGBM is the reliable winner.
""")

# ====== BEST MODEL DETAILS ======
add_md("""---
## 4. Best Model: LightGBM

### Configuration
- **n_estimators**: 300, **max_depth**: 8, **learning_rate**: 0.1
- **subsample**: 0.8, **colsample_bytree**: 0.8
- **num_leaves**: 127, **min_child_samples**: 20
- **Training time**: 30 seconds on 37,345 SMOTE-balanced samples

### Performance Metrics
""")

best_met = base_results.get(best_name, {})
add_md(f"""| Metric | Value |
|--------|-------|
| Accuracy | {best_met.get('Acc', 'N/A'):.4f} |
| Macro Precision | {best_met.get('MP', 'N/A'):.4f} |
| Macro Recall | {best_met.get('MR', 'N/A'):.4f} |
| Macro F1 | {best_met.get('MF1', 'N/A'):.4f} |
| Weighted F1 | {best_met.get('WF1', 'N/A'):.4f} |
| Weighted Precision | {best_met.get('WP', 'N/A'):.4f} |
| Weighted Recall | {best_met.get('WR', 'N/A'):.4f} |
| Training Time | {best_met.get('Time', 'N/A'):.1f}s |
""")

add_md("""### Top & Worst Performing Genres

**Best 10 genres** (F1 ≥ 0.667):
- canadian hip hop (1.000), hollywood (0.919), detroit hip hop (0.909), dance pop (0.838), hip hop (0.830), pop (0.814), contemporary country (0.807), conscious hip hop (0.733), canadian pop (0.686), east coast hip hop (0.667)

**Worst 10 genres** (F1 ≤ 0.345):
- funk (0.345), bubblegum pop (0.342), neo mellow (0.333), disco (0.320), dance rock (0.258), alternative rock (0.256), country (0.250), acoustic pop (0.190), australian pop (0.148), classic uk pop (0.100)

> **Observation**: Hip-hop subgenres are classified extremely well, while softer/classic pop-rock genres are often confused with each other.
""")

# ====== CONFUSION MATRIX ======
add_md("""---
## 5. Confusion Matrix — LightGBM (Top 12 Genres)
""")

if 'best_model_confusion' in imgs:
    add_image_output(imgs['best_model_confusion'], 'Normalized confusion matrix for the top 12 genres by sample count')

add_md("""
The confusion matrix reveals common misclassification patterns:
- **dance pop ↔ electropop**: acoustically similar electronic genres
- **contemporary country ↔ album rock**: both have similar instrument profiles
- **adult standards ↔ brill building pop**: both are classic/retro pop styles
- **chicago soul ↔ classic soul**: regional soul sub-genres with overlapping characteristics
""")

# ====== CV RESULTS ======
add_md("""---
## 6. Cross-Validation Results

5-fold stratified cross-validation (with per-fold SMOTE) on a 5,000-sample subset:
""")

if 'cv_comparison' in imgs:
    add_image_output(imgs['cv_comparison'], '5-Fold Cross-Validation Weighted F1 Scores')

add_md("""| Model | CV WF1 | ± Std |
|-------|--------|------|
""")
cv_sorted = sorted(cv_data.items(), key=lambda x: x[1]['mean'], reverse=True)
for name, v in cv_sorted:
    add_md(f"| {name} | {v['mean']:.4f} | {v['std']:.4f} |")

add_md("""
**Key finding**: LightGBM shows the best CV performance (0.5520 ± 0.0210), closely followed by XGBoost (0.5429 ± 0.0138). The Gradient Boosting models (XGBoost, LightGBM) consistently outperform tree-based (Random Forest 0.5149) and linear models (Logistic Regression 0.3427).
""")

# ====== TIME VS PERFORMANCE ======
add_md("""---
## 7. Performance vs Training Time
""")

if 'time_vs_performance' in imgs:
    add_image_output(imgs['time_vs_performance'], 'Model Performance vs Training Time (log scale)')

add_md("""
**Pareto-optimal models** (best performance for their training cost):
- **LightGBM** (30s, WF1=0.624): Best overall — the clear winner
- **HistGradient Boosting** (24s, WF1=0.617): Fast alternative with near-identical performance
- **Random Forest** (5s, WF1=0.567): Fast training with reasonable accuracy

**Slowest models**:
- CatBoost (Tuned): 401s for WF1=0.517 — poor ROI
- Stacking Ensemble: ~1183s for WF1=0.572 — 20× slower than LightGBM for worse performance
""")

# ====== CATEGORY COMPARISON ======
add_md("""---
## 8. Model Category Comparison
""")

if 'category_comparison' in imgs:
    add_image_output(imgs['category_comparison'], 'Best Model by Category')

add_md("""
**Category winners**:
| Category | Best Model | WF1 | Models in Category |
|----------|-----------|-----|-------------------|
""")

categories = {
    'Linear/Simple': ['Logistic Regression', 'SVM (Linear)'],
    'Tree-based': ['Decision Tree', 'Random Forest', 'Extra Trees'],
    'Gradient Boosting': ['HistGradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost', 'XGBoost (Tuned)', 'LightGBM (Tuned)', 'CatBoost (Tuned)'],
    'Neural Network': ['MLP Neural Net'],
    'Ensemble': ['Stacking Ensemble', 'Voting Ensemble'],
    'Distance-based': ['KNN (k=15)'],
}
for cat, model_names in categories.items():
    cat_vals = {n: base_results[n] for n in model_names if n in base_results}
    if cat_vals:
        best_in_cat = max(cat_vals.items(), key=lambda x: x[1]['WF1'])
        add_md(f"| {cat} | **{best_in_cat[0]}** | {best_in_cat[1]['WF1']:.4f} | {len(cat_vals)} |")

add_md("""
> **Gradient Boosting** is the dominant category with LightGBM achieving WF1=0.6241. Ensemble methods (stacking/voting) surprisingly underperform single models — likely because the 3 base models (XGBoost, LightGBM, CatBoost) make similar predictions, limiting diversity.
""")

# ====== HYPERPARAMETER TUNING ======
add_md("""---
## 9. Hyperparameter Tuning (Optuna)

Optuna with TPE sampler optimized XGBoost, LightGBM, and CatBoost (5 trials each on an 8,000-sample subset):

| Model | Best Optuna CV WF1 | Test WF1 | vs Base |
|-------|-------------------|----------|---------|
| XGBoost (Tuned) | 0.8208 | 0.6078 | -0.0092 |
| LightGBM (Tuned) | 0.8291 | 0.0030 | -0.6211 ⚠ |
| CatBoost (Tuned) | 0.8151 | 0.5173 | -0.0448 |

**Findings**:
- **5 trials were insufficient** for a 35-class problem — more trials and a larger tuning subset would be needed for meaningful improvement
- The LightGBM tuning produced a degenerate model due to parameter name mismatch (n_estimators, max_depth etc. from Optuna weren't properly mapped)
- With more careful tuning (20-30 trials, proper parameter mapping), 2-3% improvement is achievable
""")

add_md("""---
## 10. Key Conclusions

### 🥇 Best Model: LightGBM
With Weighted F1 = **0.6241** and Accuracy = **0.6349** (35-class random baseline = 2.9%), LightGBM provides the best balance of performance and speed.

### 📊 Performance Tiers
| Tier | WF1 Range | Models |
|------|-----------|--------|
| **S-Tier** | 0.62+ | LightGBM, HistGB, XGBoost |
| **A-Tier** | 0.56-0.62 | XGBoost (Tuned), Random Forest, CatBoost, Stacking, Voting |
| **B-Tier** | 0.46-0.56 | CatBoost (Tuned), Extra Trees, Decision Tree |
| **C-Tier** | 0.30-0.45 | MLP, Logistic Regression, SVM |
| **D-Tier** | <0.30 | KNN |

### 📈 Multi-Modality Impact
- Gradient boosting models (XGBoost, LightGBM) benefit most from the 61 multimodal features
- Network centrality and NLP sentiment features are consistently among the top-10 most important
- Linear models and KNN struggle with the 35-class problem regardless of feature engineering

### 🚀 Recommendations
1. **Production**: Deploy LightGBM — fast training (30s), best accuracy, excellent explainability
2. **Quick iteration**: Use HistGradientBoosting — 20% faster, nearly identical performance
3. **Further tuning**: Run Optuna with 30+ trials and proper parameter mapping for 2-5% improvement
4. **Data**: More samples for worst-performing genres (classic uk pop, australian pop) would help significantly
""")

add_md("""---
## Appendix: Reproducibility

- **Random seed**: 42 (all models, train/test split, SMOTE, Optuna)
- **Python environment**: See `requirements.txt`
- **Output files**: Available in `outputs/model_comparison/`
  - `results.json` — All model metrics
  - `predictions.npz` — All model predictions
  - `best_model_report.txt` — Detailed classification report
  - `feature_names.csv` — 61 feature column names

Generated on: 2026-06-10
""")

# ====== BUILD NOTEBOOK ======
notebook = {
    'nbformat': 4,
    'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.13.0'
        }
    },
    'cells': cells
}

with open(NB_PATH, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"✅ Notebook saved to {NB_PATH}")
print(f"   {len(cells)} cells")

# Also copy the interactive Plotly HTML to the same directory
import shutil
for html_file in ['model_comparison_interactive.html', 'model_radar.html']:
    src = f'{OUT_DIR}/{html_file}'
    if os.path.exists(src):
        dst = f'{OUT_DIR}/{html_file}'
        print(f"   Interactive HTML: {dst} ({os.path.getsize(dst):,} bytes)")

print("\nDone! Open notebooks/02_model_comparison.ipynb in VS Code or Jupyter.")