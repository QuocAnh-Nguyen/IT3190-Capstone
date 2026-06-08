#!/usr/bin/env python3
"""Phase 3: Model Building & Evaluation.

Usage: python scripts/run_phase3.py

Trains and evaluates the PRD's model progression:
  1. Logistic Regression (baseline)
  2. Random Forest
  3. XGBoost
  4. MLP

Performs stratified K-fold CV, selects the best model, evaluates on a
held-out test set, and saves all artifacts for Phase 4 deployment.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import DATA_PROCESSED_DIR, MODELS_DIR, RANDOM_SEED
from src.models.train import train_models
from src.utils.helpers import get_logger, load_joblib

logger = get_logger("phase3")

# ---------------------------------------------------------------------------
# Load Phase 2 data
# ---------------------------------------------------------------------------
logger.info("Loading feature matrix from Phase 2...")
df = load_joblib(DATA_PROCESSED_DIR / "feature_matrix.pkl")
logger.info(f"  {len(df):,} rows × {len(df.columns)} columns")

# Quick data sanity check
target_col = "main_genre"
vc = df[target_col].value_counts()
logger.info(f"  Genres: {len(vc)}")
logger.info(f"  Class range: {vc.iloc[0]} → {vc.iloc[-1]}")

# ---------------------------------------------------------------------------
# Train & Evaluate
# ---------------------------------------------------------------------------
results = train_models(
    df,
    target_col=target_col,
    encoded_col="genre_encoded",
    test_size=0.2,
    save_dir=MODELS_DIR,
)

# ---------------------------------------------------------------------------
# Visualizations — Confusion Matrix
# ---------------------------------------------------------------------------
logger.info("\nGenerating plots...")

label_enc = results["label_encoder"]
genre_names = label_enc.classes_
cm = results["confusion_matrix"]

fig, ax = plt.subplots(figsize=(18, 14))
cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

sns.heatmap(
    cm_normalized,
    xticklabels=genre_names,
    yticklabels=genre_names,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    ax=ax,
    cbar_kws={"label": "Proportion"},
)
ax.set_title(
    f"Confusion Matrix — {results['best_model_name']}\n"
    f"(F1 Macro: {results['metrics']['f1_macro']:.4f})",
    fontsize=14,
)
ax.set_xlabel("Predicted Genre")
ax.set_ylabel("True Genre")
plt.xticks(rotation=45, ha="right", fontsize=7)
plt.yticks(fontsize=7)
plt.tight_layout()

cm_path = MODELS_DIR / "confusion_matrix.png"
fig.savefig(cm_path, dpi=150)
logger.info(f"  Confusion matrix → {cm_path}")
plt.close()

# ---------------------------------------------------------------------------
# CV Summary Bar Chart
# ---------------------------------------------------------------------------
cv_df = results["cv_summary"]

# Extract mean F1 macro from the "mean ± std" string
cv_df["f1_mean"] = cv_df["f1_macro"].str.extract(r"([\d.]+)")[0].astype(float)
cv_df_sorted = cv_df.sort_values("f1_mean", ascending=True)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(cv_df_sorted["model"], cv_df_sorted["f1_mean"], color="steelblue")
for bar, val in zip(bars, cv_df_sorted["f1_mean"]):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=11)
ax.set_xlabel("F1 Score (Macro)")
ax.set_title("Model Comparison — Stratified K-Fold CV (F1 Macro)")
ax.set_xlim(0, max(cv_df_sorted["f1_mean"]) * 1.15)
plt.tight_layout()

cv_path = MODELS_DIR / "cv_comparison.png"
fig.savefig(cv_path, dpi=150)
logger.info(f"  CV comparison → {cv_path}")
plt.close()

# ---------------------------------------------------------------------------
# Print final summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 3 — MODEL BUILDING COMPLETE")
print("=" * 60)
print(f"Best model:   {results['best_model_name']}")
print(f"Test accuracy: {results['metrics']['accuracy']:.4f}")
print(f"Test F1 (Macro):  {results['metrics']['f1_macro']:.4f}")
print(f"Test F1 (Weighted): {results['metrics']['f1_weighted']:.4f}")
print(f"Test precision (Macro): {results['metrics']['precision_macro']:.4f}")
print(f"Test recall (Macro):    {results['metrics']['recall_macro']:.4f}")

print(f"\nCV Results:")
for _, row in cv_df.iterrows():
    print(f"  {row['model']:<25s}  F1={row['f1_macro']}")

print(f"\nArtifacts saved → {MODELS_DIR}")
print(f"  {len(list(MODELS_DIR.glob('*')))} files")
print("Done.")