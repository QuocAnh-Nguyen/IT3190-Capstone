#!/usr/bin/env python3
"""Execute the music genre classification pipeline and save outputs to JSON for notebook injection.
This script executes ALL the code cells from the notebook in sequence, capturing outputs.
"""

import sys, os, json, traceback, io
import base64
from contextlib import redirect_stdout, redirect_stderr

# We'll capture cell outputs as structured data
out_dir = '/tmp/notebook_outputs'
os.makedirs(out_dir, exist_ok=True)

# Check if we're resuming from a checkpoint
RESUME_FROM = int(sys.argv[1]) if len(sys.argv) > 1 else 0

print(f"Starting execution from cell {RESUME_FROM}")

###############################################################################
# Cell 0-1: Markdown cells (skip)
###############################################################################

###############################################################################
# Cell 2: Environment Setup & Imports
###############################################################################
if RESUME_FROM <= 2:
    print("=== Cell 2: Imports ===")
    try:
        import pandas as pd
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio

        from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
        from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import (classification_report, confusion_matrix,
                                     accuracy_score, f1_score, precision_score, recall_score)
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from imblearn.over_sampling import SMOTE

        import xgboost as xgb
        import shap
        import networkx as nx
        import re
        import ast
        from textblob import TextBlob

        import warnings
        warnings.filterwarnings('ignore')

        sns.set_style('whitegrid')
        sns.set_palette('husl')
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['figure.dpi'] = 100

        GENRE_COLORS = px.colors.qualitative.Bold

        print('All libraries loaded successfully!')
        print(f'pandas {pd.__version__} | numpy {np.__version__}')

        with open(f'{out_dir}/imports_done', 'w') as f:
            f.write('ok')
    except Exception as e:
        print(f"ERROR in cell 2: {e}")
        traceback.print_exc()
        sys.exit(1)

###############################################################################
# Cell 4: Data Loading
###############################################################################
if RESUME_FROM <= 4:
    print("\n=== Cell 4: Data Loading ===")
    try:
        DATA_PATH = '../data/raw'

        songs = pd.read_csv(f'{DATA_PATH}/musicoset_metadata/songs.csv', sep='\t')
        artists = pd.read_csv(f'{DATA_PATH}/musicoset_metadata/artists.csv', sep='\t')
        acoustic = pd.read_csv(f'{DATA_PATH}/musicoset_songfeatures/acoustic_features.csv', sep='\t')
        lyrics = pd.read_csv(f'{DATA_PATH}/musicoset_songfeatures/lyrics.csv', sep='\t')
        tracks = pd.read_csv(f'{DATA_PATH}/musicoset_metadata/tracks.csv', sep='\t')
        hits = pd.read_csv(f'{DATA_PATH}/additional/hits_dataset.csv', sep='\t')

        print(f'Dataset Shapes:')
        print(f'   songs:     {songs.shape[0]:>6,} rows × {songs.shape[1]:>2} cols')
        print(f'   artists:   {artists.shape[0]:>6,} rows × {artists.shape[1]:>2} cols')
        print(f'   acoustic:  {acoustic.shape[0]:>6,} rows × {acoustic.shape[1]:>2} cols')
        print(f'   lyrics:    {lyrics.shape[0]:>6,} rows × {lyrics.shape[1]:>2} cols')
        print(f'   hits:      {hits.shape[0]:>6,} rows × {hits.shape[1]:>2} cols')
    except Exception as e:
        print(f"ERROR in cell 4: {e}")
        traceback.print_exc()
        sys.exit(1)

###############################################################################
# Cell 6: Parse Artist IDs
###############################################################################
if RESUME_FROM <= 6:
    print("\n=== Cell 6: Parse Artist IDs ===")
    try:
        def parse_artist_ids(id_str):
            try:
                ids = ast.literal_eval(id_str)
                return ids if isinstance(ids, list) else [ids]
            except:
                return []

        hits['artist_id_list'] = hits['id_artists'].apply(parse_artist_ids)
        hits['primary_artist_id'] = hits['artist_id_list'].apply(lambda x: x[0] if len(x) > 0 else None)
        hits['num_artists_parsed'] = hits['artist_id_list'].apply(len)

        print(f'Songs with valid primary artist: {hits["primary_artist_id"].notna().sum():,}')
        print(f'Distribution of number of artists per song:')
        print(hits['num_artists_parsed'].value_counts().head(10))
    except Exception as e:
        print(f"ERROR in cell 6: {e}")
        traceback.print_exc()
        sys.exit(1)

###############################################################################
# Cell 7: Join with artists for genre
###############################################################################
if RESUME_FROM <= 7:
    print("\n=== Cell 7: Join with Artists ===")
    try:
        df = hits.merge(
            artists[['artist_id', 'main_genre', 'genres', 'followers', 'popularity']].rename(
                columns={'popularity': 'artist_popularity', 'followers': 'artist_followers'}
            ),
            left_on='primary_artist_id',
            right_on='artist_id',
            how='left'
        )

        print(f'Before genre filter: {len(df):,} rows')
        df = df[df['main_genre'].notna() & (df['main_genre'] != '-')].copy()
        print(f'After genre filter:  {len(df):,} rows')
        print(f'\nUnique genres: {df["main_genre"].nunique()}')
        print(f'\nTop 20 Genres:')
        print(df['main_genre'].value_counts().head(20))
    except Exception as e:
        print(f"ERROR in cell 7: {e}")
        traceback.print_exc()
        sys.exit(1)

###############################################################################
# Cell 8: Filter genres
###############################################################################
if RESUME_FROM <= 8:
    print("\n=== Cell 8: Filter Genres ===")
    try:
        MIN_SAMPLES = 50
        genre_counts = df['main_genre'].value_counts()
        valid_genres = genre_counts[genre_counts >= MIN_SAMPLES].index.tolist()

        df_filtered = df[df['main_genre'].isin(valid_genres)].copy()
        print(f'After filtering (>= {MIN_SAMPLES} samples/genre):')
        print(f'   Rows: {len(df_filtered):,} | Genres: {len(valid_genres)}')
        print(f'\nGenre Distribution:')
        print(df_filtered['main_genre'].value_counts())
        df = df_filtered
    except Exception as e:
        print(f"ERROR in cell 8: {e}")
        traceback.print_exc()
        sys.exit(1)

###############################################################################
# Cell 10: Merge lyrics
###############################################################################
if RESUME_FROM <= 10:
    print("\n=== Cell 10: Merge Lyrics ===")
    try:
        df = df.merge(lyrics[['song_id', 'lyrics']], on='song_id', how='left')
        print(f'After lyrics merge: {len(df):,} rows')
        print(f'Songs with lyrics: {df["lyrics"].notna().sum():,}')
        print(f'Songs without lyrics: {df["lyrics"].isna().sum():,}')
    except Exception as e:
        print(f"ERROR in cell 10: {e}")
        traceback.print_exc()
        sys.exit(1)

###############################################################################
# Cell 12: Final dataset summary
###############################################################################
if RESUME_FROM <= 12:
    print("\n=== Cell 12: Final Dataset Summary ===")
    print(f'Final Working Dataset: {df.shape[0]:,} rows × {df.shape[1]:,} columns')
    print(f'Target classes (genres): {df["main_genre"].nunique()}')
    print(f'Lyrics available: {df["lyrics"].notna().sum():,} / {len(df):,}')

###############################################################################
# Cell 14: Genre Distribution Viz (Seaborn)
###############################################################################
if RESUME_FROM <= 14:
    print("\n=== Cell 14: Genre Distribution Viz ===")
    try:
        genre_counts = df['main_genre'].value_counts()
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        colors = sns.color_palette('viridis', len(genre_counts))
        axes[0].barh(range(len(genre_counts)), genre_counts.values, color=colors)
        axes[0].set_yticks(range(len(genre_counts)))
        axes[0].set_yticklabels(genre_counts.index, fontsize=9)
        axes[0].set_xlabel('Number of Songs', fontsize=12)
        axes[0].set_title('Genre Distribution Across Dataset', fontsize=14, fontweight='bold')
        axes[0].invert_yaxis()

        top_n = 12
        top_genres_pie = genre_counts.head(top_n)
        other_count = genre_counts.iloc[top_n:].sum()
        pie_data = pd.concat([top_genres_pie, pd.Series({'Other': other_count})])
        axes[1].pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%',
                    colors=sns.color_palette('viridis', len(pie_data)), textprops={'fontsize': 8})
        axes[1].set_title(f'Top {top_n} Genres + Other', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{out_dir}/genre_distribution.png', dpi=100, bbox_inches='tight')
        plt.close()
        print('Genre distribution plot saved.')
    except Exception as e:
        print(f"ERROR in cell 14: {e}")
        traceback.print_exc()

###############################################################################
# Cell 16: Interactive genre distribution (Plotly) - save as HTML
###############################################################################
if RESUME_FROM <= 16:
    print("\n=== Cell 16: Interactive Genre Distribution ===")
    try:
        genre_counts = df['main_genre'].value_counts()
        fig = px.bar(
            x=genre_counts.values, y=genre_counts.index,
            orientation='h',
            title='Interactive Genre Distribution',
            labels={'x': 'Number of Songs', 'y': 'Genre'},
            color=genre_counts.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=600, showlegend=False)
        fig.update_yaxes(categoryorder='total ascending')
        fig.write_html(f'{out_dir}/genre_distribution_interactive.html')
        print('Interactive genre distribution saved.')
    except Exception as e:
        print(f"ERROR in cell 16: {e}")
        traceback.print_exc()

###############################################################################
# Cell 18: Acoustic Feature Distributions
###############################################################################
if RESUME_FROM <= 18:
    print("\n=== Cell 18: Acoustic Feature Distributions ===")
    try:
        ACOUSTIC_FEATURES = ['duration_ms', 'key', 'mode', 'time_signature',
                             'acousticness', 'danceability', 'energy', 'instrumentalness',
                             'liveness', 'loudness', 'speechiness', 'valence', 'tempo']
        df['duration_sec'] = df['duration_ms'] / 1000

        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.flatten()

        all_features = ['duration_sec', 'key', 'mode', 'time_signature',
                        'acousticness', 'danceability', 'energy', 'instrumentalness',
                        'liveness', 'loudness', 'speechiness', 'valence', 'tempo']

        for i, feat in enumerate(all_features):
            if i < len(axes):
                axes[i].hist(df[feat].dropna(), bins=50, color='steelblue', edgecolor='white', alpha=0.8)
                axes[i].set_title(feat, fontsize=12, fontweight='bold')
                axes[i].set_xlabel('')
                axes[i].set_ylabel('Count')
                mean_val = df[feat].mean()
                axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean={mean_val:.2f}')
                axes[i].legend(fontsize=7)

        for j in range(len(all_features), len(axes)):
            axes[j].set_visible(False)

        plt.suptitle('Distribution of Acoustic Features', fontsize=18, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/acoustic_distributions.png', dpi=100, bbox_inches='tight')
        plt.close()
        print('Acoustic distribution plot saved.')
    except Exception as e:
        print(f"ERROR in cell 18: {e}")
        traceback.print_exc()

###############################################################################
# Cell 21: Genre-specific acoustic profiles
###############################################################################
if RESUME_FROM <= 21:
    print("\n=== Cell 21: Genre Acoustic Profiles ===")
    try:
        top_genres = df['main_genre'].value_counts().head(15).index
        genre_acoustic = df[df['main_genre'].isin(top_genres)].groupby('main_genre')[ACOUSTIC_FEATURES].mean()

        from sklearn.preprocessing import MinMaxScaler
        scaler_mm = MinMaxScaler()
        genre_acoustic_scaled = pd.DataFrame(
            scaler_mm.fit_transform(genre_acoustic),
            index=genre_acoustic.index,
            columns=genre_acoustic.columns
        )

        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(genre_acoustic_scaled, annot=genre_acoustic.round(2), fmt='',
                    cmap='RdYlBu_r', center=0.5, ax=ax,
                    cbar_kws={'label': 'Normalized Value'},
                    linewidths=0.5, linecolor='white')
        ax.set_title('Acoustic Feature Profiles by Genre (Top 15)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Acoustic Feature', fontsize=12)
        ax.set_ylabel('Genre', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{out_dir}/genre_acoustic_profiles.png', dpi=100, bbox_inches='tight')
        plt.close()
        print('Genre acoustic profiles saved.')
    except Exception as e:
        print(f"ERROR in cell 21: {e}")
        traceback.print_exc()

###############################################################################
# Cell 24: Correlation matrix
###############################################################################
if RESUME_FROM <= 24:
    print("\n=== Cell 24: Correlation Matrix ===")
    try:
        corr_features = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                         'liveness', 'loudness', 'speechiness', 'valence', 'tempo', 'duration_sec']
        corr_matrix = df[corr_features].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, square=True, ax=ax,
                    linewidths=0.5, linecolor='white',
                    cbar_kws={'shrink': 0.8, 'label': 'Pearson Correlation'})
        ax.set_title('Acoustic Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{out_dir}/correlation_matrix.png', dpi=100, bbox_inches='tight')
        plt.close()
        print('Correlation matrix saved.')
    except Exception as e:
        print(f"ERROR in cell 24: {e}")
        traceback.print_exc()

###############################################################################
# Cell 25: Interactive correlation matrix
###############################################################################
if RESUME_FROM <= 25:
    print("\n=== Cell 25: Interactive Correlation Matrix ===")
    try:
        fig = px.imshow(corr_matrix, text_auto='.2f', color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1, title='Interactive Correlation Matrix', aspect='auto')
        fig.update_layout(height=700, width=800)
        fig.write_html(f'{out_dir}/correlation_matrix_interactive.html')
        print('Interactive correlation matrix saved.')
    except Exception as e:
        print(f"ERROR in cell 25: {e}")
        traceback.print_exc()

###############################################################################
# Cell 28: 3D scatter
###############################################################################
if RESUME_FROM <= 28:
    print("\n=== Cell 28: 3D Acoustic Space ===")
    try:
        top_genres_3d = df['main_genre'].value_counts().head(8).index
        df_3d = df[df['main_genre'].isin(top_genres_3d)]

        fig = px.scatter_3d(
            df_3d.sample(min(2000, len(df_3d)), random_state=42),
            x='energy', y='danceability', z='valence',
            color='main_genre', size='popularity',
            hover_name='song_name', opacity=0.7,
            title='3D Acoustic Space: Energy × Danceability × Valence',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig.update_layout(height=700, scene=dict(
            xaxis_title='Energy', yaxis_title='Danceability', zaxis_title='Valence'
        ))
        fig.write_html(f'{out_dir}/3d_acoustic_space.html')
        print('3D scatter saved.')
    except Exception as e:
        print(f"ERROR in cell 28: {e}")
        traceback.print_exc()

###############################################################################
# Cell 31: Missing data
###############################################################################
if RESUME_FROM <= 31:
    print("\n=== Cell 31: Missing Data Summary ===")
    try:
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({'Count': missing, 'Percentage': missing_pct})
        missing_df = missing_df[missing_df['Count'] > 0].sort_values('Count', ascending=False)
        print(missing_df)
    except Exception as e:
        print(f"ERROR in cell 31: {e}")
        traceback.print_exc()

###############################################################################
# Cell 32: Imputation
###############################################################################
if RESUME_FROM <= 32:
    print("\n=== Cell 32: Imputation ===")
    try:
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer

        df['has_lyrics'] = df['lyrics'].notna().astype(int)

        artist_cols = ['artist_popularity', 'artist_followers']
        if df[artist_cols].isnull().sum().sum() > 0:
            imputer = IterativeImputer(max_iter=10, random_state=42)
            df[artist_cols] = imputer.fit_transform(df[artist_cols])
            print('Artist metadata imputed using MICE (IterativeImputer)')
        else:
            print('No artist metadata imputation needed')

        critical_cols = ACOUSTIC_FEATURES + ['popularity', 'artist_popularity', 'artist_followers']
        remaining_missing = df[critical_cols].isnull().sum().sum()
        print(f'Remaining missing values in critical columns: {remaining_missing}')
    except Exception as e:
        print(f"ERROR in cell 32: {e}")
        traceback.print_exc()

###############################################################################
# Cell 34: Label encoding and pre-SMOTE summary
###############################################################################
if RESUME_FROM <= 34:
    print("\n=== Cell 34: Label Encoding ===")
    try:
        le = LabelEncoder()
        df['genre_encoded'] = le.fit_transform(df['main_genre'])

        base_features = ACOUSTIC_FEATURES + ['popularity', 'artist_popularity', 'artist_followers',
                                             'num_artists', 'explicit', 'has_lyrics']

        print(f'Before SMOTE:')
        print(f'   Samples: {len(df):,}')
        print(f'   Genres: {df["genre_encoded"].nunique()}')

        class_counts_before = df['main_genre'].value_counts()
        print(f'   Min class: {class_counts_before.min()} ({class_counts_before.idxmin()})')
        print(f'   Max class: {class_counts_before.max()} ({class_counts_before.idxmax()})')
        print(f'   Imbalance ratio: {class_counts_before.max() / class_counts_before.min():.1f}:1')
    except Exception as e:
        print(f"ERROR in cell 34: {e}")
        traceback.print_exc()

###############################################################################
# Cell 35: Class imbalance visualization
###############################################################################
if RESUME_FROM <= 35:
    print("\n=== Cell 35: Class Imbalance Viz ===")
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        genre_counts_sorted = class_counts_before.sort_values()
        axes[0].barh(range(len(genre_counts_sorted)), genre_counts_sorted.values, color='coral')
        axes[0].set_yticks(range(len(genre_counts_sorted)))
        axes[0].set_yticklabels(genre_counts_sorted.index, fontsize=7)
        axes[0].set_xlabel('Count', fontsize=12)
        axes[0].set_title('Genre Distribution — BEFORE SMOTE', fontsize=14, fontweight='bold')
        axes[0].invert_yaxis()

        axes[1].barh(range(len(genre_counts_sorted)), genre_counts_sorted.values, color='coral')
        axes[1].set_yticks(range(len(genre_counts_sorted)))
        axes[1].set_yticklabels(genre_counts_sorted.index, fontsize=7)
        axes[1].set_xscale('log')
        axes[1].set_xlabel('Count (log scale)', fontsize=12)
        axes[1].set_title('Genre Distribution — Log Scale', fontsize=14, fontweight='bold')
        axes[1].invert_yaxis()

        plt.tight_layout()
        plt.savefig(f'{out_dir}/class_imbalance_before.png', dpi=100, bbox_inches='tight')
        plt.close()
        print('Class imbalance plot saved.')
    except Exception as e:
        print(f"ERROR in cell 35: {e}")
        traceback.print_exc()

###############################################################################
# Cell 38: Build collaboration graph
###############################################################################
if RESUME_FROM <= 38:
    print("\n=== Cell 38: Build Collaboration Graph ===")
    try:
        G = nx.Graph()
        collab_count = 0
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
                            collab_count += 1
                else:
                    G.add_node(artist_ids[0])
            except:
                pass

        print(f'Collaboration Graph:')
        print(f'   Nodes (artists): {G.number_of_nodes():,}')
        print(f'   Edges (collaborations): {G.number_of_edges():,}')
        print(f'   Total collaboration links: {collab_count:,}')
        print(f'   Density: {nx.density(G):.4f}')
    except Exception as e:
        print(f"ERROR in cell 38: {e}")
        traceback.print_exc()

###############################################################################
# Cell 39: Network centrality
###############################################################################
if RESUME_FROM <= 39:
    print("\n=== Cell 39: Network Centrality ===")
    try:
        print('Computing centrality metrics...')

        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G, k=min(500, G.number_of_nodes()))
        closeness_centrality = nx.closeness_centrality(G)

        df['degree_centrality'] = df['primary_artist_id'].map(degree_centrality).fillna(0)
        df['betweenness_centrality'] = df['primary_artist_id'].map(betweenness_centrality).fillna(0)
        df['closeness_centrality'] = df['primary_artist_id'].map(closeness_centrality).fillna(0)

        clustering_coeff = nx.clustering(G)
        df['clustering_coeff'] = df['primary_artist_id'].map(clustering_coeff).fillna(0)

        print(f'Network features added:')
        print(f'   degree_centrality: mean={df["degree_centrality"].mean():.5f}, max={df["degree_centrality"].max():.5f}')
        print(f'   betweenness_centrality: mean={df["betweenness_centrality"].mean():.5f}')
        print(f'   closeness_centrality: mean={df["closeness_centrality"].mean():.5f}')
        print(f'   clustering_coeff: mean={df["clustering_coeff"].mean():.5f}')
    except Exception as e:
        print(f"ERROR in cell 39: {e}")
        traceback.print_exc()

###############################################################################
# Cell 40: Centrality visualization
###############################################################################
if RESUME_FROM <= 40:
    print("\n=== Cell 40: Centrality Visualization ===")
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].hist(df['degree_centrality'], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        axes[0].set_title('Degree Centrality Distribution', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Degree Centrality')
        axes[0].set_ylabel('Count')

        axes[1].scatter(df['degree_centrality'], df['popularity'], alpha=0.3, s=5, c='steelblue')
        axes[1].set_title('Degree Centrality vs Popularity', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Degree Centrality')
        axes[1].set_ylabel('Song Popularity')

        top_genres_net = df['main_genre'].value_counts().head(15).index
        genre_centrality = df[df['main_genre'].isin(top_genres_net)].groupby('main_genre')['degree_centrality'].mean().sort_values()
        axes[2].barh(range(len(genre_centrality)), genre_centrality.values, color='steelblue')
        axes[2].set_yticks(range(len(genre_centrality)))
        axes[2].set_yticklabels(genre_centrality.index, fontsize=8)
        axes[2].set_title('Mean Degree Centrality by Genre', fontsize=13, fontweight='bold')
        axes[2].set_xlabel('Mean Degree Centrality')

        plt.tight_layout()
        plt.savefig(f'{out_dir}/centrality_viz.png', dpi=100, bbox_inches='tight')
        plt.close()
        print('Centrality visualization saved.')
    except Exception as e:
        print(f"ERROR in cell 40: {e}")
        traceback.print_exc()

###############################################################################
# Cell 41: Interactive network viz - save as HTML
###############################################################################
if RESUME_FROM <= 41:
    print("\n=== Cell 41: Interactive Network Viz ===")
    try:
        if G.number_of_nodes() > 500:
            top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:200]
            subgraph_nodes = [n for n, _ in top_nodes]
            for node in subgraph_nodes[:50]:
                neighbors = list(G.neighbors(node))[:10]
                subgraph_nodes.extend(neighbors)
            subgraph_nodes = list(set(subgraph_nodes))[:300]
            G_sub = G.subgraph(subgraph_nodes)
        else:
            G_sub = G

        pos = nx.spring_layout(G_sub, k=0.5, iterations=30, seed=42)

        edge_x, edge_y = [], []
        for edge in G_sub.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        node_x = [pos[node][0] for node in G_sub.nodes()]
        node_y = [pos[node][1] for node in G_sub.nodes()]
        node_sizes = [degree_centrality.get(node, 0) * 100 + 5 for node in G_sub.nodes()]

        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.3, color='#888'),
                                hoverinfo='none')
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers',
                                marker=dict(size=node_sizes, color=node_sizes,
                                            colorscale='Viridis', showscale=True,
                                            colorbar=dict(title='Centrality')),
                                text=[f'Node: {n[:12]}...' for n in G_sub.nodes()],
                                hoverinfo='text')

        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(title='Interactive Artist Collaboration Network',
                                        showlegend=False, hovermode='closest',
                                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                        height=700))
        fig.write_html(f'{out_dir}/network_graph.html')
        print('Network visualization saved.')
    except Exception as e:
        print(f"ERROR in cell 41: {e}")
        traceback.print_exc()

###############################################################################
# Cell 44: NLP feature extraction function
###############################################################################
if RESUME_FROM <= 44:
    print("\n=== Cell 44: NLP Feature Extractor ===")
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

###############################################################################
# Cell 45: Extract NLP features
###############################################################################
if RESUME_FROM <= 45:
    print("\n=== Cell 45: Extract NLP Features ===")
    try:
        print('Extracting NLP features from lyrics...')
        nlp_features = df['lyrics'].apply(extract_lyrics_features)
        df = pd.concat([df, nlp_features], axis=1)

        NLP_COLS = ['word_count', 'unique_word_count', 'lexical_richness', 'avg_word_length',
                    'line_count', 'sentiment_polarity', 'sentiment_subjectivity',
                    'lyrics_has_verse', 'lyrics_has_chorus', 'lyrics_has_bridge']

        print(f'NLP features extracted: {len(NLP_COLS)} features')
        print(f'\nNLP Feature Summary:')
        print(df[NLP_COLS].describe().to_string())
    except Exception as e:
        print(f"ERROR in cell 45: {e}")
        traceback.print_exc()

###############################################################################
# Cell 46: NLP feature viz
###############################################################################
if RESUME_FROM <= 46:
    print("\n=== Cell 46: NLP Feature Viz ===")
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        nlp_vis_features = ['lexical_richness', 'sentiment_polarity', 'sentiment_subjectivity',
                            'word_count', 'avg_word_length', 'line_count']

        for i, feat in enumerate(nlp_vis_features):
            ax = axes[i // 3, i % 3]
            data = df[df['has_lyrics'] == 1][feat] if feat != 'word_count' else df[df['has_lyrics'] == 1][feat].clip(upper=1000)
            ax.hist(data.dropna(), bins=50, color='mediumseagreen', edgecolor='white', alpha=0.8)
            ax.set_title(feat, fontsize=12, fontweight='bold')
            ax.axvline(data.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean={data.mean():.3f}')
            ax.legend(fontsize=8)

        plt.suptitle('NLP Feature Distributions (Songs with Lyrics)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{out_dir}/nlp_distributions.png', dpi=100, bbox_inches='tight')
        plt.close()
        print('NLP distribution plot saved.')
    except Exception as e:
        print(f"ERROR in cell 46: {e}")
        traceback.print_exc()

###############################################################################
# Cell 47: Sentiment by genre (Plotly)
###############################################################################
if RESUME_FROM <= 47:
    print("\n=== Cell 47: Sentiment by Genre ===")
    try:
        top_genres_nlp = df['main_genre'].value_counts().head(10).index
        df_nlp_viz = df[df['main_genre'].isin(top_genres_nlp) & (df['has_lyrics'] == 1)]

        fig = px.box(
            df_nlp_viz, x='main_genre', y='sentiment_polarity', color='main_genre',
            title='Sentiment Polarity by Genre',
            labels={'sentiment_polarity': 'Sentiment Polarity', 'main_genre': 'Genre'},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig.update_layout(height=500, showlegend=False, xaxis_tickangle=-45)
        fig.write_html(f'{out_dir}/sentiment_by_genre.html')
        print('Sentiment by genre plot saved.')
    except Exception as e:
        print(f"ERROR in cell 47: {e}")
        traceback.print_exc()

###############################################################################
# Cell 50: Interaction features
###############################################################################
if RESUME_FROM <= 50:
    print("\n=== Cell 50: Interaction Features ===")
    try:
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

        print(f'Created {len(INTERACTION_COLS)} polynomial interaction features:')
        for name in INTERACTION_COLS:
            print(f'   {name}')
    except Exception as e:
        print(f"ERROR in cell 50: {e}")
        traceback.print_exc()

###############################################################################
# Cell 53: K-Means elbow method
###############################################################################
if RESUME_FROM <= 53:
    print("\n=== Cell 53: K-Means Elbow ===")
    try:
        scaler_kmeans = StandardScaler()
        acoustic_scaled = scaler_kmeans.fit_transform(df[ACOUSTIC_FEATURES])

        inertias = []
        K_range = range(1, 16)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(acoustic_scaled)
            inertias.append(km.inertia_)

        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)
        elbow_k = np.argmax(diffs2) + 2

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(K_range, inertias, 'bo-', markersize=8, linewidth=2)
        ax.set_xlabel('Number of Clusters (K)', fontsize=12)
        ax.set_ylabel('Inertia', fontsize=12)
        ax.set_title('K-Means Elbow Method', fontsize=14, fontweight='bold')
        ax.axvline(elbow_k, color='red', linestyle='--', linewidth=2, label=f'Suggested K={elbow_k}')
        ax.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/kmeans_elbow.png', dpi=100, bbox_inches='tight')
        plt.close()

        print(f'Suggested optimal K: {elbow_k}')
    except Exception as e:
        print(f"ERROR in cell 53: {e}")
        traceback.print_exc()

###############################################################################
# Cell 54: Fit K-Means
###############################################################################
if RESUME_FROM <= 54:
    print("\n=== Cell 54: Fit K-Means ===")
    try:
        OPTIMAL_K = max(elbow_k, 5)
        kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=20)
        df['acoustic_cluster'] = kmeans.fit_predict(acoustic_scaled)

        print(f'K-Means clustering complete with K={OPTIMAL_K}')
        print(f'\nCluster Distribution:')
        print(df['acoustic_cluster'].value_counts().sort_index())
    except Exception as e:
        print(f"ERROR in cell 54: {e}")
        traceback.print_exc()

###############################################################################
# Cell 55: 3D K-Means viz
###############################################################################
if RESUME_FROM <= 55:
    print("\n=== Cell 55: 3D K-Means Viz ===")
    try:
        pca_3d = PCA(n_components=3, random_state=42)
        acoustic_pca_3d = pca_3d.fit_transform(acoustic_scaled)

        df_viz = df.sample(min(2000, len(df)), random_state=42)
        viz_indices = df_viz.index

        # Use boolean indexing properly
        mask = df.index.isin(viz_indices)

        fig = px.scatter_3d(
            x=acoustic_pca_3d[mask, 0],
            y=acoustic_pca_3d[mask, 1],
            z=acoustic_pca_3d[mask, 2],
            color=df.loc[mask, 'acoustic_cluster'].astype(str),
            opacity=0.7,
            title=f'K-Means Acoustic Clusters (K={OPTIMAL_K})',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig.update_layout(height=700)
        fig.write_html(f'{out_dir}/kmeans_3d.html')
        print('3D K-Means plot saved.')
    except Exception as e:
        print(f"ERROR in cell 55: {e}")
        traceback.print_exc()

###############################################################################
# Cell 56: Cluster-genre heatmap
###############################################################################
if RESUME_FROM <= 56:
    print("\n=== Cell 56: Cluster-Genre Heatmap ===")
    try:
        cluster_genre = pd.crosstab(df['acoustic_cluster'], df['main_genre'])
        cluster_genre_pct = cluster_genre.div(cluster_genre.sum(axis=1), axis=0)

        fig, ax = plt.subplots(figsize=(16, 8))
        sns.heatmap(cluster_genre_pct.T.head(20), annot=True, fmt='.2f', cmap='YlOrRd',
                    ax=ax, linewidths=0.5, linecolor='white',
                    cbar_kws={'label': 'Proportion within Cluster'})
        ax.set_title('Genre Composition of Acoustic Clusters', fontsize=16, fontweight='bold')
        ax.set_xlabel('K-Means Cluster', fontsize=12)
        ax.set_ylabel('Genre', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/cluster_genre_heatmap.png', dpi=100, bbox_inches='tight')
        plt.close()
        print('Cluster-genre heatmap saved.')
    except Exception as e:
        print(f"ERROR in cell 56: {e}")
        traceback.print_exc()

###############################################################################
# Cell 59: Assemble feature set
###############################################################################
if RESUME_FROM <= 59:
    print("\n=== Cell 59: Assemble Feature Set ===")
    try:
        NETWORK_COLS = ['degree_centrality', 'betweenness_centrality', 'closeness_centrality', 'clustering_coeff']
        META_COLS = ['popularity', 'artist_popularity', 'artist_followers', 'num_artists',
                     'explicit', 'has_lyrics', 'duration_sec']
        CLUSTER_COLS = ['acoustic_cluster']

        all_feature_cols = (ACOUSTIC_FEATURES + META_COLS + NETWORK_COLS +
                           NLP_COLS + INTERACTION_COLS + CLUSTER_COLS)

        df['explicit'] = df['explicit'].astype(int)

        print(f'Complete Feature Matrix: {len(all_feature_cols)} features')
        print(f'   Acoustic: {len(ACOUSTIC_FEATURES)}')
        print(f'   Metadata: {len(META_COLS)}')
        print(f'   Network:  {len(NETWORK_COLS)}')
        print(f'   NLP:      {len(NLP_COLS)}')
        print(f'   Interaction: {len(INTERACTION_COLS)}')
        print(f'   Cluster:  {len(CLUSTER_COLS)}')
    except Exception as e:
        print(f"ERROR in cell 59: {e}")
        traceback.print_exc()

###############################################################################
# Cell 60: Prepare X and y
###############################################################################
if RESUME_FROM <= 60:
    print("\n=== Cell 60: Prepare X and y ===")
    try:
        feature_df = df[all_feature_cols].copy()
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.fillna(0)

        X = feature_df.values
        y = df['genre_encoded'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        print(f'X shape: {X_scaled.shape}')
        print(f'y shape: {y.shape}')
        print(f'Classes: {len(np.unique(y))}')
    except Exception as e:
        print(f"ERROR in cell 60: {e}")
        traceback.print_exc()

###############################################################################
# Cell 61: RF Feature Importance
###############################################################################
if RESUME_FROM <= 61:
    print("\n=== Cell 61: RF Feature Importance ===")
    try:
        rf_importance = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_importance.fit(X_scaled, y)

        importances = rf_importance.feature_importances_
        indices = np.argsort(importances)[::-1]

        fig, ax = plt.subplots(figsize=(12, 10))
        top_n_features = min(30, len(all_feature_cols))
        ax.barh(range(top_n_features), importances[indices][:top_n_features][::-1], color='steelblue')
        ax.set_yticks(range(top_n_features))
        ax.set_yticklabels([all_feature_cols[i] for i in indices[:top_n_features]][::-1], fontsize=10)
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title('Random Forest Feature Importance (Top 30)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{out_dir}/rf_feature_importance.png', dpi=100, bbox_inches='tight')
        plt.close()
        print('RF feature importance plot saved.')
    except Exception as e:
        print(f"ERROR in cell 61: {e}")
        traceback.print_exc()

###############################################################################
# Cell 62: Interactive feature importance
###############################################################################
if RESUME_FROM <= 62:
    print("\n=== Cell 62: Interactive Feature Importance ===")
    try:
        importance_df = pd.DataFrame({
            'Feature': [all_feature_cols[i] for i in indices[:top_n_features]],
            'Importance': importances[indices][:top_n_features]
        })

        fig = px.bar(
            importance_df, x='Importance', y='Feature', orientation='h',
            title='Interactive Feature Importance (Random Forest)',
            color='Importance', color_continuous_scale='Blues'
        )
        fig.update_layout(height=600)
        fig.write_html(f'{out_dir}/feature_importance_interactive.html')
        print('Interactive feature importance saved.')
    except Exception as e:
        print(f"ERROR in cell 62: {e}")
        traceback.print_exc()

###############################################################################
# Cell 63: PCA
###############################################################################
if RESUME_FROM <= 63:
    print("\n=== Cell 63: PCA Analysis ===")
    try:
        pca = PCA(random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        axes[0].plot(range(1, len(pca.explained_variance_ratio_) + 1),
                     np.cumsum(pca.explained_variance_ratio_), 'b-', linewidth=2)
        axes[0].axhline(0.90, color='r', linestyle='--', label='90% Variance')
        axes[0].axhline(0.95, color='g', linestyle='--', label='95% Variance')
        n90 = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.90) + 1
        n95 = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
        axes[0].axvline(n90, color='r', linestyle=':', alpha=0.5)
        axes[0].axvline(n95, color='g', linestyle=':', alpha=0.5)
        axes[0].set_xlabel('Number of Components', fontsize=12)
        axes[0].set_ylabel('Cumulative Explained Variance', fontsize=12)
        axes[0].set_title('PCA — Cumulative Explained Variance', fontsize=13, fontweight='bold')
        axes[0].legend()

        top_genres_pca = df['main_genre'].value_counts().head(8).index
        df_pca = df[df['main_genre'].isin(top_genres_pca)]

        for genre in top_genres_pca:
            mask = df_pca['main_genre'] == genre
            genre_idx = df_pca[mask].index
            pos = [i for i, idx in enumerate(df.index) if idx in genre_idx]
            axes[1].scatter(X_pca[pos, 0], X_pca[pos, 1], alpha=0.4, s=8, label=genre)

        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
        axes[1].set_title('PCA Projection — First 2 Components by Genre', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=7, loc='upper right')

        plt.tight_layout()
        plt.savefig(f'{out_dir}/pca_analysis.png', dpi=100, bbox_inches='tight')
        plt.close()

        print(f'PCA Summary:')
        print(f'   Components for 90% variance: {n90}')
        print(f'   Components for 95% variance: {n95}')
    except Exception as e:
        print(f"ERROR in cell 63: {e}")
        traceback.print_exc()

###############################################################################
# Cell 67: Train/test split
###############################################################################
if RESUME_FROM <= 67:
    print("\n=== Cell 67: Train/Test Split ===")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f'Train/Test Split:')
        print(f'   X_train: {X_train.shape}')
        print(f'   X_test:  {X_test.shape}')
        print(f'   y_train: {y_train.shape}')
        print(f'   y_test:  {y_test.shape}')
    except Exception as e:
        print(f"ERROR in cell 67: {e}")
        traceback.print_exc()

###############################################################################
# Cell 68: SMOTE
###############################################################################
if RESUME_FROM <= 68:
    print("\n=== Cell 68: SMOTE ===")
    try:
        min_class_count = np.min(np.bincount(y_train))
        k_neighbors = min(min_class_count - 1, 5)
        k_neighbors = max(k_neighbors, 1)

        print(f'Applying SMOTE with k_neighbors={k_neighbors}...')
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        print(f'\nAfter SMOTE:')
        print(f'   X_train: {X_train_smote.shape} (was {X_train.shape[0]:,})')
        print(f'   y_train: {y_train_smote.shape}')
        print(f'   Class distribution after SMOTE:')
        unique_smote, counts_smote = np.unique(y_train_smote, return_counts=True)
        for cls, cnt in zip(unique_smote, counts_smote):
            genre_name = le.inverse_transform([cls])[0]
            print(f'      {genre_name}: {cnt}')
    except Exception as e:
        print(f"ERROR in cell 68: {e}")
        traceback.print_exc()

###############################################################################
# Cell 70: Logistic Regression
###############################################################################
if RESUME_FROM <= 70:
    print("\n=== Cell 70: Logistic Regression ===")
    try:
        print('Training Logistic Regression...')
        lr = LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1)
        lr.fit(X_train_smote, y_train_smote)

        y_pred_lr = lr.predict(X_test)

        print('Logistic Regression Results:')
        print(f'   Test Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}')
        print(f'   Macro F1:      {f1_score(y_test, y_pred_lr, average="macro"):.4f}')
        print(f'   Weighted F1:   {f1_score(y_test, y_pred_lr, average="weighted"):.4f}')
    except Exception as e:
        print(f"ERROR in cell 70: {e}")
        traceback.print_exc()

###############################################################################
# Cell 72: Random Forest
###############################################################################
if RESUME_FROM <= 72:
    print("\n=== Cell 72: Random Forest ===")
    try:
        print('Training Random Forest...')
        rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5,
                                    random_state=42, n_jobs=-1)
        rf.fit(X_train_smote, y_train_smote)

        y_pred_rf = rf.predict(X_test)

        print('Random Forest Results:')
        print(f'   Test Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}')
        print(f'   Macro F1:      {f1_score(y_test, y_pred_rf, average="macro"):.4f}')
        print(f'   Weighted F1:   {f1_score(y_test, y_pred_rf, average="weighted"):.4f}')
    except Exception as e:
        print(f"ERROR in cell 72: {e}")
        traceback.print_exc()

###############################################################################
# Cell 74: XGBoost
###############################################################################
if RESUME_FROM <= 74:
    print("\n=== Cell 74: XGBoost ===")
    try:
        print('Training XGBoost...')
        xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            objective='multi:softprob', random_state=42, n_jobs=-1,
            eval_metric='mlogloss'
        )
        xgb_model.fit(X_train_smote, y_train_smote, verbose=False)

        y_pred_xgb = xgb_model.predict(X_test)

        print('XGBoost Results:')
        print(f'   Test Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}')
        print(f'   Macro F1:      {f1_score(y_test, y_pred_xgb, average="macro"):.4f}')
        print(f'   Weighted F1:   {f1_score(y_test, y_pred_xgb, average="weighted"):.4f}')
    except Exception as e:
        print(f"ERROR in cell 74: {e}")
        traceback.print_exc()

###############################################################################
# Cell 76: MLP
###############################################################################
if RESUME_FROM <= 76:
    print("\n=== Cell 76: MLP Neural Network ===")
    try:
        print('Training MLP Neural Network...')
        mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu', solver='adam',
            alpha=0.001, batch_size=128,
            learning_rate='adaptive', learning_rate_init=0.001,
            max_iter=300, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=10,
            random_state=42, verbose=False
        )
        mlp.fit(X_train_smote, y_train_smote)

        y_pred_mlp = mlp.predict(X_test)

        print('MLP Neural Network Results:')
        print(f'   Test Accuracy: {accuracy_score(y_test, y_pred_mlp):.4f}')
        print(f'   Macro F1:      {f1_score(y_test, y_pred_mlp, average="macro"):.4f}')
        print(f'   Weighted F1:   {f1_score(y_test, y_pred_mlp, average="weighted"):.4f}')
        print(f'   Iterations:    {mlp.n_iter_}')
        print(f'   Final loss:    {mlp.loss_:.4f}')
    except Exception as e:
        print(f"ERROR in cell 76: {e}")
        traceback.print_exc()

###############################################################################
# Cell 78: Model comparison table
###############################################################################
if RESUME_FROM <= 78:
    print("\n=== Cell 78: Model Comparison ===")
    try:
        models = ['Logistic Regression', 'Random Forest', 'XGBoost', 'MLP Neural Network']
        predictions = [y_pred_lr, y_pred_rf, y_pred_xgb, y_pred_mlp]

        results = []
        for name, yp in zip(models, predictions):
            results.append({
                'Model': name,
                'Accuracy': accuracy_score(y_test, yp),
                'Macro F1': f1_score(y_test, yp, average='macro'),
                'Weighted F1': f1_score(y_test, yp, average='weighted'),
                'Macro Precision': precision_score(y_test, yp, average='macro'),
                'Macro Recall': recall_score(y_test, yp, average='macro')
            })

        results_df = pd.DataFrame(results).set_index('Model')
        print('Model Performance Comparison:')
        print(results_df.to_string())
    except Exception as e:
        print(f"ERROR in cell 78: {e}")
        traceback.print_exc()

###############################################################################
# Cell 79: Interactive model comparison
###############################################################################
if RESUME_FROM <= 79:
    print("\n=== Cell 79: Interactive Model Comparison ===")
    try:
        fig = go.Figure()
        for metric in ['Accuracy', 'Macro F1', 'Weighted F1']:
            fig.add_trace(go.Bar(
                name=metric, x=results_df.index, y=results_df[metric],
                text=results_df[metric].round(4), textposition='outside'
            ))
        fig.update_layout(
            title='Model Performance Comparison',
            barmode='group',
            yaxis=dict(range=[0, 1.0], title='Score'),
            height=500,
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        fig.write_html(f'{out_dir}/model_comparison.html')
        print('Model comparison plot saved.')
    except Exception as e:
        print(f"ERROR in cell 79: {e}")
        traceback.print_exc()

###############################################################################
# Cell 80: Best model classification report
###############################################################################
if RESUME_FROM <= 80:
    print("\n=== Cell 80: Best Model Report ===")
    try:
        best_model_name = results_df['Weighted F1'].idxmax()
        best_pred_idx = models.index(best_model_name)
        best_pred = predictions[best_pred_idx]

        print(f'Best Model: {best_model_name}')
        print(f'\nClassification Report:')
        print(classification_report(y_test, best_pred, target_names=le.classes_, digits=3))
    except Exception as e:
        print(f"ERROR in cell 80: {e}")
        traceback.print_exc()

###############################################################################
# Cell 81: K-Fold CV (optimized - 3 folds, fewer estimators)
###############################################################################
if RESUME_FROM <= 81:
    print("\n=== Cell 81: K-Fold CV ===")
    try:
        print('Running 3-Fold Cross-Validation...')
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        cv_results = {}

        cv_models = [
            ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)),
            ('Random Forest', RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)),
            ('XGBoost', xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                           objective='multi:softprob', random_state=42, n_jobs=-1, eval_metric='mlogloss')),
            ('MLP', MLPClassifier(hidden_layer_sizes=(128, 64), alpha=0.001, max_iter=100,
                                  early_stopping=True, random_state=42))
        ]

        for name, model in cv_models:
            scores = []
            for train_idx, val_idx in kf.split(X_scaled, y):
                X_fold_train, X_fold_val = X_scaled[train_idx], X_scaled[val_idx]
                y_fold_train, y_fold_val = y[train_idx], y[val_idx]

                min_count = np.min(np.bincount(y_fold_train))
                k = min(max(min_count - 1, 1), 3)
                smote_fold = SMOTE(random_state=42, k_neighbors=k)
                X_fold_smote, y_fold_smote = smote_fold.fit_resample(X_fold_train, y_fold_train)

                model.fit(X_fold_smote, y_fold_smote)
                y_fold_pred = model.predict(X_fold_val)
                scores.append(f1_score(y_fold_val, y_fold_pred, average='weighted'))

            cv_results[name] = scores
            print(f'   {name}: {np.mean(scores):.4f} (+/-{np.std(scores):.4f})')

        print('Cross-validation complete!')
    except Exception as e:
        print(f"ERROR in cell 81: {e}")
        traceback.print_exc()

###############################################################################
# Cell 82: CV results plot
###############################################################################
if RESUME_FROM <= 82:
    print("\n=== Cell 82: CV Results Plot ===")
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        cv_names = list(cv_results.keys())
        cv_means = [np.mean(cv_results[n]) for n in cv_names]
        cv_stds = [np.std(cv_results[n]) for n in cv_names]

        bars = ax.bar(cv_names, cv_means, yerr=cv_stds, capsize=10,
                      color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'],
                      edgecolor='white', linewidth=1.5)
        ax.set_ylabel('Weighted F1 Score', fontsize=12)
        ax.set_title('3-Fold Cross-Validation Results', fontsize=14, fontweight='bold')

        for bar, mean, std in zip(bars, cv_means, cv_stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.005,
                    f'{mean:.4f} (+/-{std:.4f})', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylim(0, max(cv_means) + max(cv_stds) + 0.08)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/cv_results.png', dpi=100, bbox_inches='tight')
        plt.close()
        print('CV results plot saved.')
    except Exception as e:
        print(f"ERROR in cell 82: {e}")
        traceback.print_exc()

###############################################################################
# Cell 84: Confusion matrices
###############################################################################
if RESUME_FROM <= 84:
    print("\n=== Cell 84: Confusion Matrices ===")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(20, 18))
        axes = axes.flatten()

        for i, (name, yp) in enumerate(zip(models, predictions)):
            cm = confusion_matrix(y_test, yp)
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            sns.heatmap(cm_norm, ax=axes[i], cmap='Blues',
                        xticklabels=False, yticklabels=False,
                        cbar_kws={'label': 'Proportion'})
            axes[i].set_title(f'{name}\nAccuracy: {accuracy_score(y_test, yp):.4f}',
                              fontsize=13, fontweight='bold')
            axes[i].set_xlabel('Predicted', fontsize=11)
            axes[i].set_ylabel('True', fontsize=11)

        plt.suptitle('Confusion Matrices — All Models', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{out_dir}/confusion_matrices.png', dpi=100, bbox_inches='tight')
        plt.close()
        print('Confusion matrices saved.')
    except Exception as e:
        print(f"ERROR in cell 84: {e}")
        traceback.print_exc()

###############################################################################
# Cell 85: Top-15 confusion matrix
###############################################################################
if RESUME_FROM <= 85:
    print("\n=== Cell 85: Top-15 Confusion Matrix ===")
    try:
        top15_genres = df['main_genre'].value_counts().head(15).index
        test_genres = le.inverse_transform(y_test)
        top15_test_mask = np.isin(test_genres, top15_genres)

        if top15_test_mask.sum() > 50:
            y_test_top15 = y_test[top15_test_mask]
            y_pred_top15 = best_pred[top15_test_mask]

            top15_le = LabelEncoder()
            top15_labels = le.inverse_transform(y_test_top15)
            top15_encoded = top15_le.fit_transform(top15_labels)
            top15_pred_labels = le.inverse_transform(y_pred_top15)

            valid_pred_mask = np.isin(top15_pred_labels, top15_genres)
            top15_encoded_filtered = top15_encoded[valid_pred_mask]
            top15_pred_encoded = top15_le.transform(top15_pred_labels[valid_pred_mask])

            cm_top15 = confusion_matrix(top15_encoded_filtered, top15_pred_encoded)
            cm_top15_norm = cm_top15.astype('float') / cm_top15.sum(axis=1)[:, np.newaxis]

            fig, ax = plt.subplots(figsize=(14, 12))
            sns.heatmap(cm_top15_norm, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                        xticklabels=top15_le.classes_, yticklabels=top15_le.classes_,
                        linewidths=0.5, linecolor='white', cbar_kws={'label': 'Proportion'})
            ax.set_title(f'Confusion Matrix — Top 15 Genres ({best_model_name})', fontsize=15, fontweight='bold')
            ax.set_xlabel('Predicted Genre', fontsize=12)
            ax.set_ylabel('True Genre', fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=9)
            plt.yticks(rotation=0, fontsize=9)
            plt.tight_layout()
            plt.savefig(f'{out_dir}/top15_confusion.png', dpi=100, bbox_inches='tight')
            plt.close()
            print('Top-15 confusion matrix saved.')
        else:
            print('Not enough top-15 genre samples in test set.')
    except Exception as e:
        print(f"ERROR in cell 85: {e}")
        traceback.print_exc()

###############################################################################
# Cell 88: Select model for SHAP
###############################################################################
if RESUME_FROM <= 88:
    print("\n=== Cell 88: Select SHAP Model ===")
    print(f'Best model for SHAP: {best_model_name}')

    if 'XGBoost' in best_model_name:
        shap_model = xgb_model
    elif 'Random Forest' in best_model_name:
        shap_model = rf
    elif 'MLP' in best_model_name:
        shap_model = mlp
    else:
        shap_model = rf
        print('   Falling back to Random Forest for SHAP explainability')

###############################################################################
# Cell 89: SHAP explainer
###############################################################################
if RESUME_FROM <= 89:
    print("\n=== Cell 89: SHAP Explainer ===")
    try:
        n_background = min(200, len(X_train_smote))
        X_background = X_train_smote[np.random.choice(len(X_train_smote), n_background, replace=False)]

        print(f'Creating SHAP explainer with {n_background} background samples...')

        if hasattr(shap_model, 'predict_proba'):
            explainer = shap.TreeExplainer(shap_model)
            print('   Using TreeExplainer')
        else:
            explainer = shap.KernelExplainer(shap_model.predict_proba, X_background)
            print('   Using KernelExplainer')

        n_shap = min(500, len(X_test))
        X_shap = X_test[np.random.choice(len(X_test), n_shap, replace=False)]

        print(f'Computing SHAP values for {n_shap} test samples...')
        shap_values = explainer.shap_values(X_shap)
        print('SHAP values computed successfully!')
    except Exception as e:
        print(f"ERROR in cell 89: {e}")
        traceback.print_exc()

###############################################################################
# Cell 90: SHAP summary bar plot
###############################################################################
if RESUME_FROM <= 90:
    print("\n=== Cell 90: SHAP Summary Bar ===")
    try:
        fig, ax = plt.subplots(figsize=(10, 12))
        shap.summary_plot(
            shap_values, X_shap, feature_names=all_feature_cols,
            plot_type='bar', max_display=30, show=False
        )
        plt.title('SHAP Feature Importance (Global)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{out_dir}/shap_summary_bar.png', dpi=100, bbox_inches='tight')
        plt.close()
        print('SHAP summary bar plot saved.')
    except Exception as e:
        print(f"ERROR in cell 90: {e}")
        traceback.print_exc()

###############################################################################
# Cell 91: SHAP beeswarm
###############################################################################
if RESUME_FROM <= 91:
    print("\n=== Cell 91: SHAP Beeswarm ===")
    try:
        fig, ax = plt.subplots(figsize=(10, 12))
        shap.summary_plot(
            shap_values, X_shap, feature_names=all_feature_cols,
            max_display=20, show=False
        )
        plt.title('SHAP Beeswarm Summary Plot', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{out_dir}/shap_beeswarm.png', dpi=100, bbox_inches='tight')
        plt.close()
        print('SHAP beeswarm plot saved.')
    except Exception as e:
        print(f"ERROR in cell 91: {e}")
        traceback.print_exc()

###############################################################################
# Cell 92: SHAP Force Plot
###############################################################################
if RESUME_FROM <= 92:
    print("\n=== Cell 92: SHAP Force Plot ===")
    try:
        sample_idx = 0

        if isinstance(shap_values, list):
            pred_class = best_pred[sample_idx] if sample_idx < len(best_pred) else 0
            shap_vals_sample = shap_values[pred_class][sample_idx]
            base_value = explainer.expected_value[pred_class]
        else:
            shap_vals_sample = shap_values[sample_idx]
            base_value = explainer.expected_value

        true_genre = le.inverse_transform([y_test[sample_idx]])[0] if sample_idx < len(y_test) else 'N/A'
        pred_genre_idx = best_pred[sample_idx] if sample_idx < len(best_pred) else 0
        pred_genre = le.inverse_transform([pred_genre_idx])[0]

        print(f'Force Plot Explanation:')
        print(f'   True Genre:     {true_genre}')
        print(f'   Predicted:      {pred_genre}')
        print(f'   Correct:        {true_genre == pred_genre}')

        # Use waterfall as a robust alternative to force plot
        try:
            fig = shap.plots.force(base_value, shap_vals_sample, feature_names=all_feature_cols, matplotlib=True, show=False)
            plt.title(f'SHAP Force Plot — Predicted: {pred_genre} | True: {true_genre}', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{out_dir}/shap_force.png', dpi=100, bbox_inches='tight')
            plt.close()
            print('SHAP force plot saved.')
        except Exception as fe:
            print(f'Force plot issue: {fe}')
            # Fallback: waterfall
            shap.plots.waterfall(
                shap.Explanation(values=shap_vals_sample, base_values=base_value,
                                data=X_shap[sample_idx], feature_names=all_feature_cols),
                max_display=15, show=False
            )
            plt.title(f'SHAP Waterfall — Predicted: {pred_genre}', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{out_dir}/shap_force.png', dpi=100, bbox_inches='tight')
            plt.close()
            print('SHAP waterfall (fallback) saved.')
    except Exception as e:
        print(f"ERROR in cell 92: {e}")
        traceback.print_exc()

###############################################################################
# Cell 93: SHAP Waterfall
###############################################################################
if RESUME_FROM <= 93:
    print("\n=== Cell 93: SHAP Waterfall ===")
    try:
        if isinstance(shap_values, list):
            base_val = explainer.expected_value[pred_class]
            shap_vals_wf = shap_values[pred_class][sample_idx]
        else:
            base_val = explainer.expected_value
            shap_vals_wf = shap_values[sample_idx]

        print(f'SHAP Waterfall Plot for sample {sample_idx}:')
        print(f'   Base value: {base_val:.4f}')
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
                shap.Explanation(values=shap_vals_wf, base_values=base_val,
                                data=X_shap[sample_idx], feature_names=all_feature_cols),
                max_display=15, show=False
            )
            plt.title(f'SHAP Waterfall Plot — Predicted: {pred_genre}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{out_dir}/shap_waterfall.png', dpi=100, bbox_inches='tight')
            plt.close()
            print('SHAP waterfall plot saved.')
        except Exception as we:
            print(f'Waterfall issue: {we}')
            # Fallback: bar plot
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
            print('SHAP bar chart (fallback) saved.')
    except Exception as e:
        print(f"ERROR in cell 93: {e}")
        traceback.print_exc()

###############################################################################
# Cell 94: Interactive SHAP
###############################################################################
if RESUME_FROM <= 94:
    print("\n=== Cell 94: Interactive SHAP ===")
    try:
        if isinstance(shap_values, list):
            shap_global = np.abs(np.array(shap_values)).mean(axis=(0, 2))
        else:
            shap_global = np.abs(shap_values).mean(axis=0)

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
        print('Interactive SHAP plot saved.')
    except Exception as e:
        print(f"ERROR in cell 94: {e}")
        traceback.print_exc()

print("\n\n" + "="*60)
print("EXECUTION COMPLETE!")
print("="*60)
print(f"All outputs saved to: {out_dir}/")
# List all files
for f in sorted(os.listdir(out_dir)):
    size = os.path.getsize(os.path.join(out_dir, f))
    print(f"  {f} ({size:,} bytes)")