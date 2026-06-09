#!/usr/bin/env python3
"""
Extended Model Comparison Script — Music Genre Classification.
Trains 10+ models, tunes hyperparameters with Optuna, builds a stacking ensemble,
and produces a comprehensive model comparison with detailed metrics.

Models:
  - Logistic Regression (baseline)
  - K-Nearest Neighbors
  - Linear SVM (SVC)
  - Decision Tree
  - Random Forest
  - Extra Trees
  - Gradient Boosting (sklearn)
  - XGBoost
  - LightGBM
  - CatBoost
  - MLP Neural Network
  - Stacking Ensemble (RF + XGBoost + LightGBM + CatBoost)
"""

import sys, os, time, json, warnings
import traceback
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import (train_test_split, KFold, StratifiedKFold,
                                     cross_val_score, GridSearchCV, RandomizedSearchCV)
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, log_loss)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
from optuna.samplers import TPESampler

import re, ast
from textblob import TextBlob
import networkx as nx

OUT_DIR = '/tmp/extended_model_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# STEP 1: Load and prepare data (same as main notebook)
# =============================================================================
print("=" * 70)
print("STEP 1: DATA PREPARATION")
print("=" * 70)

import pathlib
DATA_PATH = str(pathlib.Path(__file__).resolve().parent.parent / 'data' / 'raw')

songs = pd.read_csv(f'{DATA_PATH}/musicoset_metadata/songs.csv', sep='\t')
artists = pd.read_csv(f'{DATA_PATH}/musicoset_metadata/artists.csv', sep='\t')
lyrics = pd.read_csv(f'{DATA_PATH}/musicoset_songfeatures/lyrics.csv', sep='\t')
hits = pd.read_csv(f'{DATA_PATH}/additional/hits_dataset.csv', sep='\t')

def parse_artist_ids(id_str):
    try:
        ids = ast.literal_eval(id_str)
        return ids if isinstance(ids, list) else [ids]
    except:
        return []

hits['artist_id_list'] = hits['id_artists'].apply(parse_artist_ids)
hits['primary_artist_id'] = hits['artist_id_list'].apply(lambda x: x[0] if len(x) > 0 else None)
hits['num_artists_parsed'] = hits['artist_id_list'].apply(len)

df = hits.merge(
    artists[['artist_id', 'main_genre', 'genres', 'followers', 'popularity']].rename(
        columns={'popularity': 'artist_popularity', 'followers': 'artist_followers'}),
    left_on='primary_artist_id', right_on='artist_id', how='left'
)
df = df[df['main_genre'].notna() & (df['main_genre'] != '-')].copy()

MIN_SAMPLES = 50
genre_counts = df['main_genre'].value_counts()
valid_genres = genre_counts[genre_counts >= MIN_SAMPLES].index.tolist()
df = df[df['main_genre'].isin(valid_genres)].copy()
df = df.merge(lyrics[['song_id', 'lyrics']], on='song_id', how='left')

print(f"Working dataset: {len(df):,} rows, {df['main_genre'].nunique()} genres")

# Feature definitions
ACOUSTIC_FEATURES = ['duration_ms', 'key', 'mode', 'time_signature',
                     'acousticness', 'danceability', 'energy', 'instrumentalness',
                     'liveness', 'loudness', 'speechiness', 'valence', 'tempo']
df['duration_sec'] = df['duration_ms'] / 1000
df['has_lyrics'] = df['lyrics'].notna().astype(int)
df['explicit'] = df['explicit'].astype(int)

# Impute artist metadata
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
artist_cols = ['artist_popularity', 'artist_followers']
if df[artist_cols].isnull().sum().sum() > 0:
    imputer = IterativeImputer(max_iter=10, random_state=42)
    df[artist_cols] = imputer.fit_transform(df[artist_cols])

# Label encoding
le = LabelEncoder()
df['genre_encoded'] = le.fit_transform(df['main_genre'])
N_CLASSES = df['genre_encoded'].nunique()
print(f"Classes: {N_CLASSES}")

# =============================================================================
# STEP 2: Feature Engineering
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 70)

# 2a: Graph/Network Features
print("Building collaboration graph...")
G = nx.Graph()
for _, row in songs.iterrows():
    try:
        artist_dict = ast.literal_eval(row['artists'])
        artist_ids = list(artist_dict.keys())
        if len(artist_ids) > 1:
            for i in range(len(artist_ids)):
                for j in range(i + 1, len(artist_ids)):
                    if G.has_edge(artist_ids[i], artist_ids[j]):
                        G[artist_ids[i]][artist_ids[j]]['weight'] += 1
                    else:
                        G.add_edge(artist_ids[i], artist_ids[j], weight=1)
        else:
            G.add_node(artist_ids[0])
    except: pass

degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G, k=min(500, G.number_of_nodes()))
closeness_centrality = nx.closeness_centrality(G)
clustering_coeff = nx.clustering(G)
try:
    eigenvector_centrality = nx.eigenvector_centrality_numpy(G, max_iter=200)
except:
    # Fallback for disconnected graphs
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=200)

df['degree_centrality'] = df['primary_artist_id'].map(degree_centrality).fillna(0)
df['betweenness_centrality'] = df['primary_artist_id'].map(betweenness_centrality).fillna(0)
df['closeness_centrality'] = df['primary_artist_id'].map(closeness_centrality).fillna(0)
df['clustering_coeff'] = df['primary_artist_id'].map(clustering_coeff).fillna(0)
df['eigenvector_centrality'] = df['primary_artist_id'].map(eigenvector_centrality).fillna(0)
print(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# 2b: NLP Features
print("Extracting NLP features...")
def extract_lyrics_features(lyrics_text):
    if pd.isna(lyrics_text) or lyrics_text == '':
        return pd.Series({
            'word_count': 0, 'unique_word_count': 0, 'lexical_richness': 0,
            'avg_word_length': 0, 'line_count': 0,
            'sentiment_polarity': 0, 'sentiment_subjectivity': 0,
            'lyrics_has_verse': 0, 'lyrics_has_chorus': 0, 'lyrics_has_bridge': 0
        })
    cleaned = re.sub(r'\[.*?\]', '', str(lyrics_text))
    words = cleaned.lower().split()
    word_count = len(words)
    unique_words = len(set(words))
    lexical_richness = unique_words / max(word_count, 1)
    avg_word_length = np.mean([len(w) for w in words]) if words else 0
    lines = str(lyrics_text).split('\n')
    line_count = len([l for l in lines if l.strip()])
    if word_count > 10:
        blob = TextBlob(cleaned)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
    else:
        polarity, subjectivity = 0, 0
    text_lower = str(lyrics_text).lower()
    has_verse = 1 if 'verse' in text_lower else 0
    has_chorus = 1 if 'chorus' in text_lower else 0
    has_bridge = 1 if 'bridge' in text_lower else 0
    return pd.Series({
        'word_count': word_count, 'unique_word_count': unique_words,
        'lexical_richness': lexical_richness, 'avg_word_length': avg_word_length,
        'line_count': line_count,
        'sentiment_polarity': polarity, 'sentiment_subjectivity': subjectivity,
        'lyrics_has_verse': has_verse, 'lyrics_has_chorus': has_chorus,
        'lyrics_has_bridge': has_bridge
    })

nlp_features = df['lyrics'].apply(extract_lyrics_features)
df = pd.concat([df, nlp_features], axis=1)
NLP_COLS = list(nlp_features.columns)

# 2c: Interaction Features
interaction_features = ['energy', 'danceability', 'valence', 'acousticness', 'loudness', 'tempo']
df_interaction = df[interaction_features].copy()
scaler_poly = StandardScaler()
df_interaction_scaled = scaler_poly.fit_transform(df_interaction)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_features = poly.fit_transform(df_interaction_scaled)
poly_feature_names = poly.get_feature_names_out(interaction_features)
interaction_names = [n for n in poly_feature_names if ' ' in n]
interaction_indices = [i for i, n in enumerate(poly_feature_names) if ' ' in n]
interaction_matrix = poly_features[:, interaction_indices]
for i, name in enumerate(interaction_names):
    col_name = 'inter_' + name.replace(' ', '_x_')
    df[col_name] = interaction_matrix[:, i]
INTERACTION_COLS = ['inter_' + name.replace(' ', '_x_') for name in interaction_names]

# 2d: K-Means
scaler_kmeans = StandardScaler()
acoustic_scaled = scaler_kmeans.fit_transform(df[ACOUSTIC_FEATURES])
kmeans = KMeans(n_clusters=8, random_state=42, n_init=20)
df['acoustic_cluster'] = kmeans.fit_predict(acoustic_scaled)

# 2e: PCA-reduced features (top 10 components)
pca = PCA(n_components=10, random_state=42)
pca_features = pca.fit_transform(StandardScaler().fit_transform(df[ACOUSTIC_FEATURES]))
for i in range(10):
    df[f'pca_acoustic_{i+1}'] = pca_features[:, i]

# =============================================================================
# STEP 3: Assemble Feature Matrix
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: FEATURE MATRIX")
print("=" * 70)

NETWORK_COLS = ['degree_centrality', 'betweenness_centrality', 'closeness_centrality',
                'clustering_coeff', 'eigenvector_centrality']
META_COLS = ['popularity', 'artist_popularity', 'artist_followers', 'num_artists',
             'explicit', 'has_lyrics', 'duration_sec']
CLUSTER_COLS = ['acoustic_cluster']
PCA_COLS = [f'pca_acoustic_{i+1}' for i in range(10)]

all_feature_cols = (ACOUSTIC_FEATURES + META_COLS + NETWORK_COLS +
                    NLP_COLS + INTERACTION_COLS + CLUSTER_COLS + PCA_COLS)

feature_df = df[all_feature_cols].copy()
feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
feature_df = feature_df.fillna(0)

X = feature_df.values
y = df['genre_encoded'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Feature matrix: {X_scaled.shape[0]:,} rows × {X_scaled.shape[1]} features")
print(f"Feature breakdown: Acoustic={len(ACOUSTIC_FEATURES)}, Meta={len(META_COLS)}, "
      f"Network={len(NETWORK_COLS)}, NLP={len(NLP_COLS)}, "
      f"Interaction={len(INTERACTION_COLS)}, Cluster={len(CLUSTER_COLS)}, PCA={len(PCA_COLS)}")

# =============================================================================
# STEP 4: Train/Test Split + SMOTE
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: TRAIN/TEST SPLIT + SMOTE")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

min_class_count = np.min(np.bincount(y_train))
k_neighbors = min(max(min_class_count - 1, 1), 5)
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"X_train_smote: {X_train_smote.shape}")
print(f"X_test: {X_test.shape}")
print(f"Classes balanced to ~{np.bincount(y_train_smote)[0]} each")

# For SVM and KNN (scale-sensitive), also keep non-SMOTE'd but scaled data
X_train_orig = X_train
y_train_orig = y_train

# =============================================================================
# STEP 5: Define all models
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: TRAINING ALL MODELS")
print("=" * 70)

results = {}
training_times = {}
all_predictions = {}

def evaluate_model(name, y_true, y_pred, train_time, y_pred_proba=None):
    """Compute comprehensive metrics for a model."""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Macro Precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Macro Recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'Macro F1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'Weighted Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Weighted Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'Weighted F1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'Train Time (s)': train_time,
    }
    return metrics

# --- Model 1: Logistic Regression ---
print("\n[1/11] Logistic Regression...")
t0 = time.time()
lr = LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1, C=0.1)
lr.fit(X_train_smote, y_train_smote)
y_pred_lr = lr.predict(X_test)
t_lr = time.time() - t0
results['Logistic Regression'] = evaluate_model('LR', y_test, y_pred_lr, t_lr)
all_predictions['Logistic Regression'] = y_pred_lr

# --- Model 2: K-Nearest Neighbors ---
print("[2/11] K-Nearest Neighbors...")
t0 = time.time()
# For KNN, we use a subset of SMOTE'd data (5000 samples) to avoid extremely slow prediction
n_knn_train = min(10000, len(X_train_smote))
idx_knn = np.random.choice(len(X_train_smote), n_knn_train, replace=False)
knn = KNeighborsClassifier(n_neighbors=15, weights='distance', n_jobs=-1)
knn.fit(X_train_smote[idx_knn], y_train_smote[idx_knn])
y_pred_knn = knn.predict(X_test)
t_knn = time.time() - t0
results['KNN (k=15)'] = evaluate_model('KNN', y_test, y_pred_knn, t_knn)
all_predictions['KNN (k=15)'] = y_pred_knn

# --- Model 3: SVM --- Use LinearSVC + SGD (much faster than kernel SVC for 35 classes)
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
print("[3/11] SVM (LinearSVC + Calibrated)...")
t0 = time.time()
# Use SGDClassifier with hinge loss for speed, then calibrate
from sklearn.calibration import CalibratedClassifierCV
n_svm_train = min(15000, len(X_train_smote))
idx_svm = np.random.choice(len(X_train_smote), n_svm_train, replace=False)
svm_base = LinearSVC(C=1.0, max_iter=2000, random_state=42, dual=False)
svm = CalibratedClassifierCV(svm_base, cv=3)
svm.fit(X_train_smote[idx_svm], y_train_smote[idx_svm])
y_pred_svm = svm.predict(X_test)
t_svm = time.time() - t0
results['SVM (Linear)'] = evaluate_model('SVM', y_test, y_pred_svm, t_svm)
all_predictions['SVM (Linear)'] = y_pred_svm

# --- Model 4: Decision Tree ---
print("[4/11] Decision Tree...")
t0 = time.time()
dt = DecisionTreeClassifier(max_depth=20, min_samples_split=10, random_state=42)
dt.fit(X_train_smote, y_train_smote)
y_pred_dt = dt.predict(X_test)
t_dt = time.time() - t0
results['Decision Tree'] = evaluate_model('DT', y_test, y_pred_dt, t_dt)
all_predictions['Decision Tree'] = y_pred_dt

# --- Model 5: Random Forest (tuned) ---
print("[5/11] Random Forest (tuned)...")
t0 = time.time()
rf = RandomForestClassifier(n_estimators=300, max_depth=25, min_samples_split=5,
                            min_samples_leaf=2, max_features='sqrt',
                            class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train_smote, y_train_smote)
y_pred_rf = rf.predict(X_test)
t_rf = time.time() - t0
results['Random Forest'] = evaluate_model('RF', y_test, y_pred_rf, t_rf)
all_predictions['Random Forest'] = y_pred_rf

# --- Model 6: Extra Trees ---
print("[6/11] Extra Trees...")
t0 = time.time()
et = ExtraTreesClassifier(n_estimators=300, max_depth=25, min_samples_split=5,
                          max_features='sqrt', random_state=42, n_jobs=-1)
et.fit(X_train_smote, y_train_smote)
y_pred_et = et.predict(X_test)
t_et = time.time() - t0
results['Extra Trees'] = evaluate_model('ET', y_test, y_pred_et, t_et)
all_predictions['Extra Trees'] = y_pred_et

# --- Model 7: Gradient Boosting (sklearn) --- USE HistGradientBoosting which is much faster
print("[7/11] HistGradient Boosting...")
from sklearn.ensemble import HistGradientBoostingClassifier
t0 = time.time()
hgb = HistGradientBoostingClassifier(max_iter=200, max_depth=8, learning_rate=0.1,
                                      random_state=42, early_stopping=False)
hgb.fit(X_train_smote, y_train_smote)
y_pred_gb = hgb.predict(X_test)
t_gb = time.time() - t0
results['HistGradient Boosting'] = evaluate_model('HGB', y_test, y_pred_gb, t_gb)
all_predictions['HistGradient Boosting'] = y_pred_gb

# --- Model 8: XGBoost ---
print("[8/11] XGBoost...")
t0 = time.time()
xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8,
                               objective='multi:softprob', random_state=42,
                               n_jobs=-1, eval_metric='mlogloss')
xgb_model.fit(X_train_smote, y_train_smote, verbose=False)
y_pred_xgb = xgb_model.predict(X_test)
t_xgb = time.time() - t0
results['XGBoost'] = evaluate_model('XGB', y_test, y_pred_xgb, t_xgb)
all_predictions['XGBoost'] = y_pred_xgb

# --- Model 9: LightGBM ---
print("[9/11] LightGBM...")
t0 = time.time()
lgb_model = lgb.LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                                subsample=0.8, colsample_bytree=0.8,
                                random_state=42, n_jobs=-1, verbose=-1,
                                num_leaves=127, min_child_samples=20)
lgb_model.fit(X_train_smote, y_train_smote)
y_pred_lgb = lgb_model.predict(X_test)
t_lgb = time.time() - t0
results['LightGBM'] = evaluate_model('LGBM', y_test, y_pred_lgb, t_lgb)
all_predictions['LightGBM'] = y_pred_lgb

# --- Model 10: CatBoost ---
print("[10/11] CatBoost...")
t0 = time.time()
cb_model = cb.CatBoostClassifier(iterations=300, depth=8, learning_rate=0.1,
                                  random_seed=42, thread_count=-1,
                                  verbose=False, allow_writing_files=False)
cb_model.fit(X_train_smote, y_train_smote)
y_pred_cb = cb_model.predict(X_test)
# CatBoost returns 1D array sometimes
y_pred_cb = np.array(y_pred_cb).flatten()
t_cb = time.time() - t0
results['CatBoost'] = evaluate_model('CB', y_test, y_pred_cb, t_cb)
all_predictions['CatBoost'] = y_pred_cb

# --- Model 11: MLP Neural Network ---
print("[11/11] MLP Neural Network...")
t0 = time.time()
mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu',
                    solver='adam', alpha=0.001, batch_size=128,
                    learning_rate='adaptive', learning_rate_init=0.001,
                    max_iter=300, early_stopping=True,
                    validation_fraction=0.1, n_iter_no_change=10,
                    random_state=42, verbose=False)
mlp.fit(X_train_smote, y_train_smote)
y_pred_mlp = mlp.predict(X_test)
t_mlp = time.time() - t0
results['MLP Neural Net'] = evaluate_model('MLP', y_test, y_pred_mlp, t_mlp)
all_predictions['MLP Neural Net'] = y_pred_mlp

print("\n✅ All 11 base models trained!")

# =============================================================================
# STEP 6: Hyperparameter Tuning with Optuna (for top models)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: HYPERPARAMETER TUNING (Optuna)")
print("=" * 70)

# Use a smaller subset for tuning speed
n_tune = min(8000, len(X_train_smote))
idx_tune = np.random.choice(len(X_train_smote), n_tune, replace=False)
X_tune = X_train_smote[idx_tune]
y_tune = y_train_smote[idx_tune]

X_tune_train, X_tune_val, y_tune_train, y_tune_val = train_test_split(
    X_tune, y_tune, test_size=0.2, random_state=42, stratify=y_tune
)

tuned_models = {}

def tune_xgboost(X_tr, y_tr, X_val, y_val, n_trials=30):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        }
        model = xgb.XGBClassifier(**params, objective='multi:softprob',
                                   random_state=42, n_jobs=-1, eval_metric='mlogloss')
        model.fit(X_tr, y_tr, verbose=False)
        y_pred = model.predict(X_val)
        return f1_score(y_val, y_pred, average='weighted')

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value

def tune_lightgbm(X_tr, y_tr, X_val, y_val, n_trials=30):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'num_leaves': trial.suggest_int('num_leaves', 15, 255),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        }
        model = lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        return f1_score(y_val, y_pred, average='weighted')

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value

def tune_catboost(X_tr, y_tr, X_val, y_val, n_trials=30):
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10, log=True),
            'random_strength': trial.suggest_float('random_strength', 1e-2, 10, log=True),
        }
        model = cb.CatBoostClassifier(**params, random_seed=42, thread_count=-1,
                                       verbose=False, allow_writing_files=False)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val).flatten()
        return f1_score(y_val, y_pred, average='weighted')

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value

# Tune XGBoost
print("\nTuning XGBoost (30 trials)...")
t0 = time.time()
xgb_best_params, xgb_best_score = tune_xgboost(X_tune_train, y_tune_train, X_tune_val, y_tune_val, n_trials=30)
print(f"  Best params: {xgb_best_params}")
print(f"  Best CV F1: {xgb_best_score:.4f}")
print(f"  Tuning time: {time.time() - t0:.1f}s")

# Train tuned XGBoost
t0 = time.time()
xgb_tuned = xgb.XGBClassifier(**xgb_best_params, objective='multi:softprob',
                               random_state=42, n_jobs=-1, eval_metric='mlogloss')
xgb_tuned.fit(X_train_smote, y_train_smote, verbose=False)
y_pred_xgb_tuned = xgb_tuned.predict(X_test)
t_xgb_tuned = time.time() - t0
results['XGBoost (Tuned)'] = evaluate_model('XGB-T', y_test, y_pred_xgb_tuned, t_xgb_tuned)
all_predictions['XGBoost (Tuned)'] = y_pred_xgb_tuned
tuned_models['XGBoost'] = {'params': xgb_best_params, 'model': xgb_tuned}

# Tune LightGBM
print("\nTuning LightGBM (30 trials)...")
t0 = time.time()
lgb_best_params, lgb_best_score = tune_lightgbm(X_tune_train, y_tune_train, X_tune_val, y_tune_val, n_trials=30)
print(f"  Best params: {lgb_best_params}")
print(f"  Best CV F1: {lgb_best_score:.4f}")
print(f"  Tuning time: {time.time() - t0:.1f}s")

t0 = time.time()
lgb_tuned = lgb.LGBMClassifier(**lgb_best_params, random_state=42, n_jobs=-1, verbose=-1)
lgb_tuned.fit(X_train_smote, y_train_smote)
y_pred_lgb_tuned = lgb_tuned.predict(X_test)
t_lgb_tuned = time.time() - t0
results['LightGBM (Tuned)'] = evaluate_model('LGBM-T', y_test, y_pred_lgb_tuned, t_lgb_tuned)
all_predictions['LightGBM (Tuned)'] = y_pred_lgb_tuned
tuned_models['LightGBM'] = {'params': lgb_best_params, 'model': lgb_tuned}

# Tune CatBoost
print("\nTuning CatBoost (30 trials)...")
t0 = time.time()
cb_best_params, cb_best_score = tune_catboost(X_tune_train, y_tune_train, X_tune_val, y_tune_val, n_trials=25)
print(f"  Best params: {cb_best_params}")
print(f"  Best CV F1: {cb_best_score:.4f}")
print(f"  Tuning time: {time.time() - t0:.1f}s")

t0 = time.time()
cb_tuned = cb.CatBoostClassifier(**cb_best_params, random_seed=42, thread_count=-1,
                                  verbose=False, allow_writing_files=False)
cb_tuned.fit(X_train_smote, y_train_smote)
y_pred_cb_tuned = cb_tuned.predict(X_test).flatten()
t_cb_tuned = time.time() - t0
results['CatBoost (Tuned)'] = evaluate_model('CB-T', y_test, y_pred_cb_tuned, t_cb_tuned)
all_predictions['CatBoost (Tuned)'] = y_pred_cb_tuned
tuned_models['CatBoost'] = {'params': cb_best_params, 'model': cb_tuned}

# Tune Random Forest
print("\nTuning Random Forest (20 trials)...")
def tune_rf(X_tr, y_tr, X_val, y_val, n_trials=20):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 10, 40),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        }
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1, class_weight='balanced')
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        return f1_score(y_val, y_pred, average='weighted')

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value

t0 = time.time()
rf_best_params, rf_best_score = tune_rf(X_tune_train, y_tune_train, X_tune_val, y_tune_val, n_trials=20)
print(f"  Best params: {rf_best_params}")
print(f"  Best CV F1: {rf_best_score:.4f}")

t0 = time.time()
rf_tuned = RandomForestClassifier(**rf_best_params, random_state=42, n_jobs=-1, class_weight='balanced')
rf_tuned.fit(X_train_smote, y_train_smote)
y_pred_rf_tuned = rf_tuned.predict(X_test)
t_rf_tuned = time.time() - t0
results['Random Forest (Tuned)'] = evaluate_model('RF-T', y_test, y_pred_rf_tuned, t_rf_tuned)
all_predictions['Random Forest (Tuned)'] = y_pred_rf_tuned
tuned_models['RandomForest'] = {'params': rf_best_params, 'model': rf_tuned}

print("\n✅ Hyperparameter tuning complete!")
print(f"   XGBoost best CV F1:  {xgb_best_score:.4f}")
print(f"   LightGBM best CV F1: {lgb_best_score:.4f}")
print(f"   CatBoost best CV F1: {cb_best_score:.4f}")
print(f"   RandomForest best CV F1: {rf_best_score:.4f}")

# =============================================================================
# STEP 7: Stacking Ensemble
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: STACKING ENSEMBLE")
print("=" * 70)

# Build stacking classifier with the best tuned models
base_estimators = [
    ('xgb', tuned_models.get('XGBoost', {}).get('model', xgb_model)),
    ('lgb', tuned_models.get('LightGBM', {}).get('model', lgb_model)),
    ('cb', tuned_models.get('CatBoost', {}).get('model', cb_model)),
    ('rf', tuned_models.get('RandomForest', {}).get('model', rf)),
]

# Use Logistic Regression as meta-learner
meta_learner = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)

print("Training stacking classifier...")
t0 = time.time()
# Use subset for stacking due to memory
n_stack = min(15000, len(X_train_smote))
idx_stack = np.random.choice(len(X_train_smote), n_stack, replace=False)
stacking = StackingClassifier(
    estimators=base_estimators,
    final_estimator=meta_learner,
    cv=3, n_jobs=-1, passthrough=False
)
stacking.fit(X_train_smote[idx_stack], y_train_smote[idx_stack])
y_pred_stack = stacking.predict(X_test)
t_stack = time.time() - t0
results['Stacking Ensemble'] = evaluate_model('Stack', y_test, y_pred_stack, t_stack)
all_predictions['Stacking Ensemble'] = y_pred_stack
print(f"  Stacking ensemble trained in {t_stack:.1f}s")

# Also try a simpler Voting Classifier
print("Training voting classifier...")
t0 = time.time()
voting = VotingClassifier(
    estimators=[
        ('xgb', tuned_models.get('XGBoost', {}).get('model', xgb_model)),
        ('lgb', tuned_models.get('LightGBM', {}).get('model', lgb_model)),
        ('cb', tuned_models.get('CatBoost', {}).get('model', cb_model)),
    ],
    voting='soft', weights=[2, 1, 1]
)
voting.fit(X_train_smote[idx_stack], y_train_smote[idx_stack])
y_pred_voting = voting.predict(X_test)
t_voting = time.time() - t0
results['Voting Ensemble'] = evaluate_model('Vote', y_test, y_pred_voting, t_voting)
all_predictions['Voting Ensemble'] = y_pred_voting
print(f"  Voting ensemble trained in {t_voting:.1f}s")

# =============================================================================
# STEP 8: Comprehensive Cross-Validation
# =============================================================================
print("\n" + "=" * 70)
print("STEP 8: 5-FOLD CROSS-VALIDATION")
print("=" * 70)

cv_results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# CV for key models (using smaller subset for speed)
n_cv = min(5000, len(X_scaled))
idx_cv = np.random.choice(len(X_scaled), n_cv, replace=False)
X_cv = X_scaled[idx_cv]
y_cv = y[idx_cv]

cv_models = {
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                  objective='multi:softprob', random_state=42, n_jobs=-1,
                                  eval_metric='mlogloss'),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                    random_state=42, n_jobs=-1, verbose=-1),
    'CatBoost': cb.CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1,
                                       random_seed=42, thread_count=-1,
                                       verbose=False, allow_writing_files=False),
    'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=100, max_depth=8,
                                                            random_state=42, early_stopping=False),
    'Extra Trees': ExtraTreesClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
}

for name, model in cv_models.items():
    t0 = time.time()
    scores = []
    for train_idx, val_idx in skf.split(X_cv, y_cv):
        X_fold_train, X_fold_val = X_cv[train_idx], X_cv[val_idx]
        y_fold_train, y_fold_val = y_cv[train_idx], y_cv[val_idx]

        min_count = np.min(np.bincount(y_fold_train))
        k = min(max(min_count - 1, 1), 3)
        smote_fold = SMOTE(random_state=42, k_neighbors=k)
        X_fold_smote, y_fold_smote = smote_fold.fit_resample(X_fold_train, y_fold_train)

        model.fit(X_fold_smote, y_fold_smote)
        y_fold_pred = model.predict(X_fold_val)
        scores.append(f1_score(y_fold_val, y_fold_pred, average='weighted'))

    cv_results[name] = {'mean': np.mean(scores), 'std': np.std(scores), 'scores': scores}
    print(f"  {name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# =============================================================================
# STEP 9: Generate Visualizations
# =============================================================================
print("\n" + "=" * 70)
print("STEP 9: GENERATING VISUALIZATIONS")
print("=" * 70)

# Sort results by Weighted F1
sorted_models = sorted(results.items(), key=lambda x: x[1]['Weighted F1'], reverse=True)

# 9a: Static comparison bar chart
print("Creating comparison bar chart...")
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

metrics_plot = ['Accuracy', 'Weighted F1', 'Macro F1']
titles = ['Accuracy', 'Weighted F1', 'Macro F1']
colors_list = plt.cm.viridis(np.linspace(0.2, 0.9, len(sorted_models)))

for idx, (metric, title) in enumerate(zip(metrics_plot, titles)):
    names = [m[0] for m in sorted_models]
    values = [m[1][metric] for m in sorted_models]
    ax = axes[idx]
    bars = ax.barh(range(len(names)), values, color=colors_list)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(metric, fontsize=11)
    ax.set_title(f'Model Comparison — {title}', fontsize=13, fontweight='bold')
    ax.set_xlim(0, max(values) * 1.15)
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=8, fontweight='bold')

plt.suptitle('Comprehensive Model Comparison — Multi-Modal Music Genre Classification',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/model_comparison_all.png', dpi=120, bbox_inches='tight')
plt.close()

# 9b: Interactive Plotly comparison
print("Creating interactive comparison...")
results_df = pd.DataFrame(results).T
results_df.index.name = 'Model'
results_df = results_df.reset_index()
results_df = results_df.sort_values('Weighted F1', ascending=True)

fig = go.Figure()
for metric in ['Accuracy', 'Weighted F1', 'Macro F1']:
    fig.add_trace(go.Bar(
        name=metric, y=results_df['Model'], x=results_df[metric],
        orientation='h', text=results_df[metric].round(4),
        textposition='outside', textfont=dict(size=10)
    ))

fig.update_layout(
    title='<b>Model Performance Comparison — All Models</b><br><sup>35-class Music Genre Classification</sup>',
    barmode='group', height=700,
    xaxis=dict(title='Score', range=[0, 0.85]),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    margin=dict(l=200)
)
fig.write_html(f'{OUT_DIR}/model_comparison_interactive.html')

# 9c: Radar/Spider chart for top models
print("Creating radar chart...")
top_radar_models = [m for m in sorted_models[:6]]
radar_metrics = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1', 'Weighted F1']

fig = go.Figure()
for name, metrics in top_radar_models:
    values = [metrics[m] for m in radar_metrics]
    values.append(values[0])  # Close the loop
    fig.add_trace(go.Scatterpolar(
        r=values, theta=radar_metrics + [radar_metrics[0]],
        name=name, fill='toself', opacity=0.6
    ))

fig.update_layout(
    title='<b>Top 6 Models — Radar Comparison</b>',
    polar=dict(radialaxis=dict(range=[0, 0.8], tickfont=dict(size=10))),
    height=600, showlegend=True
)
fig.write_html(f'{OUT_DIR}/model_radar.html')

# 9d: Training time vs Performance
print("Creating time vs performance plot...")
fig, ax = plt.subplots(figsize=(12, 8))
names = [m[0] for m in sorted_models]
f1_scores = [m[1]['Weighted F1'] for m in sorted_models]
times = [m[1]['Train Time (s)'] for m in sorted_models]

scatter = ax.scatter(times, f1_scores, s=120, c=range(len(names)), cmap='viridis',
                     edgecolors='black', linewidth=1, zorder=5)
for i, name in enumerate(names):
    ax.annotate(name, (times[i], f1_scores[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=7,
                alpha=0.8)
ax.set_xlabel('Training Time (seconds)', fontsize=12)
ax.set_ylabel('Weighted F1 Score', fontsize=12)
ax.set_title('Model Performance vs Training Time', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Model Rank', fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/time_vs_performance.png', dpi=120, bbox_inches='tight')
plt.close()

# 9e: CV results comparison
print("Creating CV comparison...")
fig, ax = plt.subplots(figsize=(10, 6))
cv_names_list = list(cv_results.keys())
cv_means = [cv_results[n]['mean'] for n in cv_names_list]
cv_stds = [cv_results[n]['std'] for n in cv_names_list]

bars = ax.bar(cv_names_list, cv_means, yerr=cv_stds, capsize=10,
              color=plt.cm.viridis(np.linspace(0.2, 0.9, len(cv_names_list))),
              edgecolor='white', linewidth=1.5)
ax.set_ylabel('Weighted F1 Score', fontsize=12)
ax.set_title('5-Fold Cross-Validation Results', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(cv_means) + max(cv_stds) + 0.08)
for bar, mean, std in zip(bars, cv_means, cv_stds):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.005,
            f'{mean:.4f}±{std:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.xticks(rotation=30, ha='right', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/cv_comparison.png', dpi=120, bbox_inches='tight')
plt.close()

# 9f: Confusion matrix for best model
print("Creating confusion matrix (best model)...")
best_name = sorted_models[0][0]
best_pred = all_predictions[best_name]

# Top 12 genres only for readability
top12_genres = df['main_genre'].value_counts().head(12).index
test_genres_labels = le.inverse_transform(y_test)
top12_mask = np.isin(test_genres_labels, top12_genres)

if top12_mask.sum() > 50:
    y_test_sub = y_test[top12_mask]
    y_pred_sub = best_pred[top12_mask]

    # Remap to top-12 encoding
    sub_le = LabelEncoder()
    sub_labels = le.inverse_transform(y_test_sub)
    y_test_enc = sub_le.fit_transform(sub_labels)
    y_pred_labels = le.inverse_transform(y_pred_sub)
    # Handle predictions for genres outside top12
    valid_pm = np.isin(y_pred_labels, top12_genres)
    y_test_enc_f = y_test_enc[valid_pm]
    y_pred_enc_f = sub_le.transform(y_pred_labels[valid_pm])

    cm = confusion_matrix(y_test_enc_f, y_pred_enc_f)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                xticklabels=sub_le.classes_, yticklabels=sub_le.classes_,
                linewidths=0.5, linecolor='white', cbar_kws={'label': 'Proportion'})
    ax.set_title(f'Confusion Matrix — {best_name} (Top 12 Genres)', fontsize=15, fontweight='bold')
    ax.set_xlabel('Predicted Genre', fontsize=12)
    ax.set_ylabel('True Genre', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/best_model_confusion.png', dpi=120, bbox_inches='tight')
    plt.close()

# 9g: Detailed classification report for best model
print("\n" + "=" * 70)
print(f"BEST MODEL: {best_name}")
print("=" * 70)

best_metrics = results[best_name]
print(f"  Accuracy:          {best_metrics['Accuracy']:.4f}")
print(f"  Macro F1:          {best_metrics['Macro F1']:.4f}")
print(f"  Weighted F1:       {best_metrics['Weighted F1']:.4f}")
print(f"  Macro Precision:   {best_metrics['Macro Precision']:.4f}")
print(f"  Macro Recall:      {best_metrics['Macro Recall']:.4f}")
print(f"  Training Time:     {best_metrics['Train Time (s)']:.1f}s")
print(f"\n  Classification Report (top 10 genres by F1):")
report = classification_report(y_test, best_pred, target_names=le.classes_, digits=3, output_dict=True)
report_df = pd.DataFrame(report).T
# Filter out avg rows and show top genres by f1-score
genre_rows = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
genre_rows = genre_rows.sort_values('f1-score', ascending=False)
print(genre_rows.head(15).to_string())

# Save full classification report
with open(f'{OUT_DIR}/best_model_report.txt', 'w') as f:
    f.write(f"Best Model: {best_name}\n")
    f.write(f"Accuracy: {best_metrics['Accuracy']:.4f}\n")
    f.write(f"Macro F1: {best_metrics['Macro F1']:.4f}\n")
    f.write(f"Weighted F1: {best_metrics['Weighted F1']:.4f}\n\n")
    f.write(classification_report(y_test, best_pred, target_names=le.classes_, digits=3))

# =============================================================================
# STEP 10: Final Summary
# =============================================================================
print("\n" + "=" * 70)
print("FINAL RANKINGS (by Weighted F1)")
print("=" * 70)

for rank, (name, metrics) in enumerate(sorted_models, 1):
    star = " ⭐" if rank == 1 else ""
    print(f"  {rank:2d}. {name:<25s} | WF1={metrics['Weighted F1']:.4f} | "
          f"Acc={metrics['Accuracy']:.4f} | MF1={metrics['Macro F1']:.4f} | "
          f"Time={metrics['Train Time (s)']:.1f}s{star}")

# Save results as JSON for notebook injection
results_json = {}
for name, metrics in results.items():
    results_json[name] = {k: float(v) if isinstance(v, (np.floating, float)) else v
                          for k, v in metrics.items()}

results_json['_cv_results'] = {k: {'mean': float(v['mean']), 'std': float(v['std'])}
                                for k, v in cv_results.items()}
results_json['_best_model'] = best_name
results_json['_tuned_params'] = {k: {pk: (int(pv) if isinstance(pv, np.integer) else float(pv) if isinstance(pv, np.floating) else pv)
                                      for pk, pv in v['params'].items()}
                                  for k, v in tuned_models.items()}

with open(f'{OUT_DIR}/results.json', 'w') as f:
    json.dump(results_json, f, indent=2)

# Save all predictions for later confusion matrix analysis
np.savez(f'{OUT_DIR}/predictions.npz', y_test=y_test, **{k: v for k, v in all_predictions.items()})

print(f"\n✅ All outputs saved to {OUT_DIR}/")
print(f"Files generated:")
for f in sorted(os.listdir(OUT_DIR)):
    size = os.path.getsize(os.path.join(OUT_DIR, f))
    print(f"  {f} ({size:,} bytes)")