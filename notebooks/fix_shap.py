#!/usr/bin/env python3
"""Fix the SHAP cells by running just the problematic cells with corrected code."""
import sys, os
import json

out_dir = '/tmp/notebook_outputs'

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

# Reload data from checkpoint
DATA_PATH = '../data/raw'
import ast
from textblob import TextBlob
import re
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import networkx as nx
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, precision_score, recall_score)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from imblearn.over_sampling import SMOTE

# Quick reload pipeline to get shap_values and required variables
print("Reloading data pipeline...")

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
        columns={'popularity': 'artist_popularity', 'followers': 'artist_followers'}
    ), left_on='primary_artist_id', right_on='artist_id', how='left'
)
df = df[df['main_genre'].notna() & (df['main_genre'] != '-')].copy()

MIN_SAMPLES = 50
genre_counts = df['main_genre'].value_counts()
valid_genres = genre_counts[genre_counts >= MIN_SAMPLES].index.tolist()
df = df[df['main_genre'].isin(valid_genres)].copy()
df = df.merge(lyrics[['song_id', 'lyrics']], on='song_id', how='left')

ACOUSTIC_FEATURES = ['duration_ms', 'key', 'mode', 'time_signature',
                     'acousticness', 'danceability', 'energy', 'instrumentalness',
                     'liveness', 'loudness', 'speechiness', 'valence', 'tempo']
df['duration_sec'] = df['duration_ms'] / 1000
df['has_lyrics'] = df['lyrics'].notna().astype(int)

# Impute
artist_cols = ['artist_popularity', 'artist_followers']
if df[artist_cols].isnull().sum().sum() > 0:
    imputer = IterativeImputer(max_iter=10, random_state=42)
    df[artist_cols] = imputer.fit_transform(df[artist_cols])

le = LabelEncoder()
df['genre_encoded'] = le.fit_transform(df['main_genre'])

# Build graph
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
    except:
        pass

degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G, k=min(500, G.number_of_nodes()))
closeness_centrality = nx.closeness_centrality(G)
clustering_coeff = nx.clustering(G)

df['degree_centrality'] = df['primary_artist_id'].map(degree_centrality).fillna(0)
df['betweenness_centrality'] = df['primary_artist_id'].map(betweenness_centrality).fillna(0)
df['closeness_centrality'] = df['primary_artist_id'].map(closeness_centrality).fillna(0)
df['clustering_coeff'] = df['primary_artist_id'].map(clustering_coeff).fillna(0)

# NLP
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

print("  Extracting NLP features...")
nlp_features = df['lyrics'].apply(extract_lyrics_features)
df = pd.concat([df, nlp_features], axis=1)

NLP_COLS = ['word_count', 'unique_word_count', 'lexical_richness', 'avg_word_length',
            'line_count', 'sentiment_polarity', 'sentiment_subjectivity',
            'lyrics_has_verse', 'lyrics_has_chorus', 'lyrics_has_bridge']

# Interaction features
interaction_features = ['energy', 'danceability', 'valence', 'acousticness', 'loudness', 'tempo']
df_interaction = df[interaction_features].copy()
scaler_poly = StandardScaler()
df_interaction_scaled = scaler_poly.fit_transform(df_interaction)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_features = poly.fit_transform(df_interaction_scaled)
poly_feature_names = poly.get_feature_names_out(interaction_features)
interaction_names = [name for name in poly_feature_names if ' ' in name]
interaction_indices = [i for i, name in enumerate(poly_feature_names) if ' ' in name]
interaction_matrix = poly_features[:, interaction_indices]
for i, name in enumerate(interaction_names):
    col_name = 'inter_' + name.replace(' ', '_x_')
    df[col_name] = interaction_matrix[:, i]
INTERACTION_COLS = ['inter_' + name.replace(' ', '_x_') for name in interaction_names]

# K-Means
scaler_kmeans = StandardScaler()
acoustic_scaled = scaler_kmeans.fit_transform(df[ACOUSTIC_FEATURES])
kmeans = KMeans(n_clusters=5, random_state=42, n_init=20)
df['acoustic_cluster'] = kmeans.fit_predict(acoustic_scaled)

# Full feature set
NETWORK_COLS = ['degree_centrality', 'betweenness_centrality', 'closeness_centrality', 'clustering_coeff']
META_COLS = ['popularity', 'artist_popularity', 'artist_followers', 'num_artists',
             'explicit', 'has_lyrics', 'duration_sec']
CLUSTER_COLS = ['acoustic_cluster']
all_feature_cols = (ACOUSTIC_FEATURES + META_COLS + NETWORK_COLS +
                   NLP_COLS + INTERACTION_COLS + CLUSTER_COLS)
df['explicit'] = df['explicit'].astype(int)

feature_df = df[all_feature_cols].copy()
feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
feature_df = feature_df.fillna(0)
X = feature_df.values
y = df['genre_encoded'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

min_class_count = np.min(np.bincount(y_train))
k_neighbors = min(max(min_class_count - 1, 1), 5)
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Models
rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1)
rf.fit(X_train_smote, y_train_smote)
y_pred_rf = rf.predict(X_test)

xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8,
                               objective='multi:softprob', random_state=42, n_jobs=-1, eval_metric='mlogloss')
xgb_model.fit(X_train_smote, y_train_smote, verbose=False)
y_pred_xgb = xgb_model.predict(X_test)

models = ['Logistic Regression', 'Random Forest', 'XGBoost', 'MLP Neural Network']

# Best model
best_model_name = 'XGBoost'
best_pred = y_pred_xgb
shap_model = xgb_model

# SHAP
print("Computing SHAP values...")
n_background = min(200, len(X_train_smote))
X_background = X_train_smote[np.random.choice(len(X_train_smote), n_background, replace=False)]
explainer = shap.TreeExplainer(shap_model)
n_shap = min(500, len(X_test))
X_shap = X_test[np.random.choice(len(X_test), n_shap, replace=False)]
shap_values = explainer.shap_values(X_shap)

print(f"SHAP values shape: {np.array(shap_values).shape}")  # (n_classes, n_samples, n_features)

# ======== FIXED CELL 92: Force/Waterfall Plot ========
print("\n=== SHAP Force/Waterfall Plot ===")
sample_idx = 0
pred_class = best_pred[sample_idx]
# Select SHAP for the predicted class: shap_values[pred_class][sample_idx]
shap_vals_sample = shap_values[pred_class][sample_idx]
base_value = explainer.expected_value[pred_class]  # scalar for this class

true_genre = le.inverse_transform([y_test[sample_idx]])[0]
pred_genre = le.inverse_transform([pred_class])[0]

print(f"True Genre: {true_genre}")
print(f"Predicted:  {pred_genre}")
print(f"Base value (expected output): {base_value:.4f}")

# Waterfall plot (reliable alternative to force plot)
try:
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_vals_sample,
            base_values=base_value,
            data=X_shap[sample_idx],
            feature_names=all_feature_cols
        ),
        max_display=15, show=False
    )
    plt.title(f'SHAP Waterfall Plot — Predicted: {pred_genre} | True: {true_genre}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/shap_force_plot.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("SHAP waterfall (force alternative) saved.")
except Exception as e:
    print(f"Waterfall error: {e}")
    # Fallback: bar chart of top contributions
    fig, ax = plt.subplots(figsize=(10, 8))
    top_idx = np.argsort(np.abs(shap_vals_sample))[-15:]
    ax.barh(range(15), shap_vals_sample[top_idx],
            color=['#ff0051' if v < 0 else '#008bfb' for v in shap_vals_sample[top_idx]])
    ax.set_yticks(range(15))
    ax.set_yticklabels([all_feature_cols[i] for i in top_idx])
    ax.set_xlabel('SHAP Value')
    ax.set_title(f'SHAP Contributions — Predicted: {pred_genre}', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/shap_force_plot.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("SHAP bar chart (fallback) saved.")

# ======== FIXED CELL 93: Waterfall with text output ========
print("\n=== SHAP Waterfall Detailed ===")
shap_vals_wf = shap_values[pred_class][sample_idx]
print(f'SHAP Waterfall for sample {sample_idx}:')
print(f'   Base value (expected output for class {pred_genre}): {base_value:.4f}')
print(f'   Model output (prediction): {base_value + shap_vals_wf.sum():.4f}')
print(f'   Top 5 positive contributions:')
top_pos_idx = np.argsort(shap_vals_wf)[-5:][::-1]
for i in top_pos_idx:
    if shap_vals_wf[i] > 0.001:
        print(f'      + {all_feature_cols[i]}: {shap_vals_wf[i]:.4f}')
print(f'   Top 5 negative contributions:')
top_neg_idx = np.argsort(shap_vals_wf)[:5]
for i in top_neg_idx:
    if shap_vals_wf[i] < -0.001:
        print(f'      - {all_feature_cols[i]}: {shap_vals_wf[i]:.4f}')

try:
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_vals_wf, base_values=base_value,
            data=X_shap[sample_idx], feature_names=all_feature_cols
        ),
        max_display=15, show=False
    )
    plt.title(f'SHAP Waterfall Plot — Predicted: {pred_genre}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/shap_waterfall.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Waterfall plot saved.")
except Exception as e:
    print(f"Waterfall error: {e}")
    # Fallback bar chart
    fig, ax = plt.subplots(figsize=(10, 8))
    top_idx = np.argsort(np.abs(shap_vals_wf))[-15:]
    ax.barh(range(15), shap_vals_wf[top_idx],
            color=['#ff0051' if v < 0 else '#008bfb' for v in shap_vals_wf[top_idx]])
    ax.set_yticks(range(15))
    ax.set_yticklabels([all_feature_cols[i] for i in top_idx])
    ax.set_xlabel('SHAP Value')
    ax.set_title(f'SHAP Contributions — Predicted: {pred_genre}', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/shap_waterfall.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Fallback bar chart saved.")

# ======== FIXED CELL 94: Interactive SHAP ========
print("\n=== Interactive SHAP ===")
# Global importance: average absolute SHAP across all classes, all samples
# shap_values shape: (n_samples, n_features, n_classes)
shap_global = np.abs(np.array(shap_values)).mean(axis=(0, 2))  # mean across (sample, class) → (n_features,)
print(f"Global SHAP shape: {shap_global.shape}")

shap_importance_df = pd.DataFrame({
    'Feature': all_feature_cols,
    'SHAP Importance': shap_global
}).sort_values('SHAP Importance', ascending=True).tail(25)

fig = px.bar(
    shap_importance_df, x='SHAP Importance', y='Feature', orientation='h',
    title='Top 25 Features by SHAP Importance',
    color='SHAP Importance', color_continuous_scale='Viridis'
)
fig.update_layout(height=600)
fig.write_html(f'{out_dir}/shap_importance_interactive.html')
print("Interactive SHAP plot saved.")

print("\nAll SHAP cells fixed and saved!")