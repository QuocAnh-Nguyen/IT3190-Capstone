#!/usr/bin/env python3
"""Phase 3 consolidated: Rebuild dataset with broad genre categories,
then train models with SMOTE and improved hyperparameters.

Usage: python scripts/run_phase3_consolidated.py
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    CV_FOLDS,
    RF_N_ESTIMATORS,
    XGB_N_ESTIMATORS,
    MLP_HIDDEN_LAYERS,
)
from src.data.genre_mapping import apply_genre_consolidation
from src.utils.helpers import get_logger, load_joblib, save_joblib, save_json

logger = get_logger("phase3_consolidated")

# ---------------------------------------------------------------------------
# Load Phase 2 full features and re-build with consolidated genres
# ---------------------------------------------------------------------------
logger.info("Loading full features from Phase 2...")
df = load_joblib(DATA_PROCESSED_DIR / "full_features.pkl")
logger.info(f"  {len(df):,} rows × {len(df.columns)} columns")

# Apply genre consolidation
df["main_genre"] = apply_genre_consolidation(df["main_genre"])
genre_counts = df["main_genre"].value_counts()
logger.info(f"  Consolidated genres: {len(genre_counts)}")
for genre, count in genre_counts.items():
    logger.info(f"    {genre:<25s} {count:>5d}  ({count/len(df)*100:5.1f}%)")

# ---------------------------------------------------------------------------
# Build feature matrix
# ---------------------------------------------------------------------------
# Collect all feature columns (exclude non-feature columns)
exclude = [
    "song_id", "song_name", "explicit", "song_type", "artist_id",
    "artist_name", "main_genre", "lyrics", "_clean_lyrics",
    "is_collaborative", "primary_artist_id",
]
feature_cols = [c for c in df.columns if c not in exclude]
# Also exclude any remaining object/text columns
for c in list(feature_cols):
    if df[c].dtype == object:
        feature_cols.remove(c)

logger.info(f"Feature columns: {len(feature_cols)}")

X = df[feature_cols].fillna(0).values
y_raw = df["main_genre"]

# Encode labels
label_enc = LabelEncoder()
y = label_enc.fit_transform(y_raw)
n_classes = len(label_enc.classes_)

logger.info(f"  X: {X.shape}, y: {y.shape}, classes: {n_classes}")

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED,
)
logger.info(f"  Train: {X_train.shape[0]:,}  Test: {X_test.shape[0]:,}")

# ---------------------------------------------------------------------------
# Define models with SMOTE pipelines
# ---------------------------------------------------------------------------
# SMOTE with k_neighbors adjusted for small classes
def _get_k_neighbors(y_train_subset):
    min_samples = min(np.bincount(y_train_subset))
    return max(1, min(5, min_samples - 1))

k_neigh = _get_k_neighbors(y_train)
logger.info(f"  SMOTE k_neighbors: {k_neigh}")

models = {
    "Logistic Regression": ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_SEED, k_neighbors=k_neigh)),
        ("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced",
            random_state=RANDOM_SEED, n_jobs=-1,
        )),
    ]),
    "Random Forest": ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_SEED, k_neighbors=k_neigh)),
        ("clf", RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, max_depth=20,
            min_samples_split=5, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1,
        )),
    ]),
    "XGBoost": ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_SEED, k_neighbors=k_neigh)),
        ("clf", XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=8,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            objective="multi:softmax", eval_metric="mlogloss",
            random_state=RANDOM_SEED, n_jobs=-1,
        )),
    ]),
    "MLP": ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_SEED, k_neighbors=k_neigh)),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), max_iter=300,
            early_stopping=True, random_state=RANDOM_SEED,
        )),
    ]),
}

# ---------------------------------------------------------------------------
# Train and evaluate each model
# ---------------------------------------------------------------------------
results = {}

for name, pipe in models.items():
    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {name}")
    logger.info(f"{'='*60}")

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_m = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    logger.info(f"  Test Accuracy:   {acc:.4f}")
    logger.info(f"  Test F1 (Macro):  {f1_m:.4f}")
    logger.info(f"  Test F1 (Weighted): {f1_w:.4f}")

    results[name] = {
        "model": pipe,
        "accuracy": acc,
        "f1_macro": f1_m,
        "f1_weighted": f1_w,
        "y_pred": y_pred,
    }

# ---------------------------------------------------------------------------
# Select best model and do detailed evaluation
# ---------------------------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["f1_macro"])
best_result = results[best_name]
logger.info(f"\n{'='*60}")
logger.info(f"BEST MODEL: {best_name}")
logger.info(f"{'='*60}")

# Classification report
report = classification_report(
    y_test, best_result["y_pred"],
    target_names=label_enc.classes_,
    zero_division=0,
)
logger.info(f"\nClassification Report:\n{report}")

# Confusion matrix
cm = confusion_matrix(y_test, best_result["y_pred"])

# ---------------------------------------------------------------------------
# Train deploy model on ALL data
# ---------------------------------------------------------------------------
logger.info(f"\nTraining final model on all data for deployment...")

# Re-instantiate the best model type with the same params
if best_name == "Logistic Regression":
    deploy_pipe = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_SEED, k_neighbors=k_neigh)),
        ("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced",
            random_state=RANDOM_SEED, n_jobs=-1,
        )),
    ])
elif best_name == "Random Forest":
    deploy_pipe = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_SEED, k_neighbors=k_neigh)),
        ("clf", RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, max_depth=20,
            min_samples_split=5, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1,
        )),
    ])
elif best_name == "XGBoost":
    deploy_pipe = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_SEED, k_neighbors=k_neigh)),
        ("clf", XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=8,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            objective="multi:softmax", eval_metric="mlogloss",
            random_state=RANDOM_SEED, n_jobs=-1,
        )),
    ])
else:
    deploy_pipe = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_SEED, k_neighbors=k_neigh)),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), max_iter=300,
            early_stopping=True, random_state=RANDOM_SEED,
        )),
    ])

deploy_pipe.fit(X, y)
final_acc = deploy_pipe.score(X, y)
logger.info(f"  Full-data accuracy: {final_acc:.4f}")

# ---------------------------------------------------------------------------
# Save all artifacts
# ---------------------------------------------------------------------------
MODELS_DIR.mkdir(parents=True, exist_ok=True)

save_joblib(deploy_pipe, MODELS_DIR / "best_model_pipeline.joblib")
save_joblib(scaler, MODELS_DIR / "feature_scaler.joblib")
save_joblib(label_enc, MODELS_DIR / "label_encoder.joblib")
save_joblib(feature_cols, MODELS_DIR / "feature_columns.joblib")
np.save(MODELS_DIR / "confusion_matrix.npy", cm)

save_json({
    "best_model": best_name,
    "metrics": {
        "accuracy": best_result["accuracy"],
        "f1_macro": best_result["f1_macro"],
        "f1_weighted": best_result["f1_weighted"],
        "final_train_accuracy": final_acc,
    },
    "n_classes": n_classes,
    "n_features": X.shape[1],
    "n_samples": X.shape[0],
    "genre_distribution": df["main_genre"].value_counts().to_dict(),
}, MODELS_DIR / "final_metrics.json")

# ---------------------------------------------------------------------------
# Confusion Matrix Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 12))
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
sns.heatmap(
    cm_norm, xticklabels=label_enc.classes_, yticklabels=label_enc.classes_,
    annot=True, fmt=".2f", cmap="Blues", ax=ax,
    cbar_kws={"label": "Proportion"},
)
ax.set_title(
    f"Confusion Matrix — {best_name} (Consolidated Genres)\n"
    f"F1 Macro: {best_result['f1_macro']:.4f}, Accuracy: {best_result['accuracy']:.4f}",
    fontsize=12,
)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
fig.savefig(MODELS_DIR / "confusion_matrix.png", dpi=150)
logger.info(f"  Confusion matrix → {MODELS_DIR / 'confusion_matrix.png'}")
plt.close()

# ---------------------------------------------------------------------------
# Model Comparison Bar Chart
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
names = list(results.keys())
f1_scores = [r["f1_macro"] for r in results.values()]
acc_scores = [r["accuracy"] for r in results.values()]

x = np.arange(len(names))
width = 0.35
bars1 = ax.bar(x - width/2, f1_scores, width, label="F1 Macro", color="steelblue")
bars2 = ax.bar(x + width/2, acc_scores, width, label="Accuracy", color="orange")
for bar, val in zip(bars1, f1_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", fontsize=10)
for bar, val in zip(bars2, acc_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", fontsize=10)

ax.set_ylabel("Score")
ax.set_title("Model Comparison — Consolidated Genres")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.legend(loc="lower right")
ax.set_ylim(0, max(max(f1_scores), max(acc_scores)) * 1.15)
plt.tight_layout()
fig.savefig(MODELS_DIR / "cv_comparison.png", dpi=150)
logger.info(f"  Comparison chart → {MODELS_DIR / 'cv_comparison.png'}")
plt.close()

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 3 CONSOLIDATED — COMPLETE")
print("=" * 60)
print(f"Best model: {best_name}")
print(f"Num genres: {n_classes}")
print(f"Test Accuracy:   {best_result['accuracy']:.4f}")
print(f"Test F1 (Macro):  {best_result['f1_macro']:.4f}")
print(f"Test F1 (Weighted): {best_result['f1_weighted']:.4f}")
print(f"Full-data Accuracy: {final_acc:.4f}")
print()
print("Genres:")
for g, c in genre_counts.items():
    print(f"  {g:<25s} {c:>5d}")
print(f"\nAll artifacts saved → {MODELS_DIR}")
print("Done.")