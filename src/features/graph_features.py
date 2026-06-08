"""Graph/Network feature extraction using NetworkX.

Builds an artist-artist collaboration network from the song metadata
and computes centrality metrics to capture the "homophily" effect
(the tendency of similar artists to collaborate within specific genres).

Features computed:
- Per-song: number of collaborating artists, is_collaborative
- Per-artist: degree centrality, betweenness centrality, clustering coefficient
- Mapped to songs as mean of participating artists' metrics
"""

from pathlib import Path
from typing import Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from src.utils.helpers import get_logger, save_joblib

logger = get_logger(__name__)


def build_artist_graph(
    df: pd.DataFrame,
    artist_id_col: str = "artist_id",
    song_id_col: str = "song_id",
) -> nx.Graph:
    """Build an artist-artist collaboration graph.

    Two artists are connected if they appear on the same song.
    Edge weight = number of collaborations.
    """
    logger.info("Building artist-artist collaboration graph...")

    # Group: song_id → list of artist_ids
    song_artists = df.groupby(song_id_col)[artist_id_col].apply(list)

    G = nx.Graph()
    edge_weights = {}

    for song_id, artists in song_artists.items():
        artists = list(set(artists))  # deduplicate
        for i in range(len(artists)):
            for j in range(i + 1, len(artists)):
                pair = tuple(sorted([artists[i], artists[j]]))
                edge_weights[pair] = edge_weights.get(pair, 0) + 1

    for (a1, a2), weight in edge_weights.items():
        G.add_edge(a1, a2, weight=weight)

    # Add isolated nodes (artists with no collaborations)
    all_artists = df[artist_id_col].unique()
    for a in all_artists:
        if a not in G:
            G.add_node(a)

    logger.info(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    if G.number_of_edges() > 0:
        density = nx.density(G)
        logger.info(f"  Density: {density:.6f}")

    return G


def compute_artist_metrics(G: nx.Graph) -> pd.DataFrame:
    """Compute centrality and clustering metrics for each artist.

    Returns DataFrame indexed by artist_id with columns:
        degree_centrality, betweenness_centrality, clustering_coefficient
    """
    logger.info("Computing artist-level graph metrics...")

    metrics = pd.DataFrame(index=list(G.nodes()))

    # Degree Centrality: how connected each artist is
    metrics["degree_centrality"] = pd.Series(nx.degree_centrality(G))

    # Betweenness Centrality: how often an artist lies on shortest paths
    # (computationally expensive — approximate for large graphs)
    k = min(500, len(G))
    metrics["betweenness_centrality"] = pd.Series(
        nx.betweenness_centrality(G, k=k, normalized=True, seed=42)
    )

    # Clustering Coefficient: how tightly an artist's collaborators
    # collaborate with each other
    metrics["clustering_coefficient"] = pd.Series(nx.clustering(G))

    # Eigenvector Centrality: influence of connected nodes
    try:
        metrics["eigenvector_centrality"] = pd.Series(
            nx.eigenvector_centrality_numpy(G)
        )
    except Exception:
        logger.warning("  Eigenvector centrality failed — using zeros")
        metrics["eigenvector_centrality"] = 0.0

    metrics.index.name = "artist_id"

    logger.info(f"  Computed metrics for {len(metrics):,} artists")
    return metrics


def extract_graph_features(
    df: pd.DataFrame,
    artist_metrics: Optional[pd.DataFrame] = None,
    build_graph: bool = True,
    save_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Optional[nx.Graph], Optional[pd.DataFrame]]:
    """Extract graph-based features for each song.

    If build_graph=True, constructs the artist graph from df and computes metrics.
    Otherwise, uses the provided artist_metrics DataFrame.

    Song-level features:
        - graph_num_artists: number of artists on the track
        - graph_is_collaborative: 0/1 flag
        - graph_degree_centrality: mean degree centrality of artists
        - graph_betweenness_centrality: mean betweenness centrality
        - graph_clustering_coefficient: mean clustering coefficient
        - graph_eigenvector_centrality: mean eigenvector centrality

    Returns:
        (df with graph feature columns, graph, artist_metrics_df)
    """
    logger.info("Extracting graph/network features...")
    df = df.copy()

    G = None

    if build_graph:
        G = build_artist_graph(df)
        artist_metrics = compute_artist_metrics(G)

    if artist_metrics is None:
        raise ValueError("artist_metrics must be provided if build_graph=False")

    # --- Song-level features ---
    # Number of artists on the track (already in df if from loader)
    if "num_artists" in df.columns:
        df["graph_num_artists"] = df["num_artists"]
    else:
        df["graph_num_artists"] = 1

    # Collaborative flag
    if "is_collaborative" in df.columns:
        df["graph_is_collaborative"] = df["is_collaborative"].astype(int)
    else:
        df["graph_is_collaborative"] = (df["graph_num_artists"] > 1).astype(int)

    # Map artist-level metrics to songs
    # For songs with multiple artists, take the mean
    metrics_cols = [
        "degree_centrality",
        "betweenness_centrality",
        "clustering_coefficient",
        "eigenvector_centrality",
    ]

    # Build a lookup from artist_id → metric values
    lookup = artist_metrics[metrics_cols]

    for col in metrics_cols:
        graph_col = f"graph_{col}"
        df[graph_col] = df["artist_id"].map(lookup[col])
        # Fill missing artists (not in graph) with 0
        df[graph_col] = df[graph_col].fillna(0.0)

    graph_feature_cols = (
        ["graph_num_artists", "graph_is_collaborative"]
        + [f"graph_{c}" for c in metrics_cols]
    )

    logger.info(f"  Total graph features: {len(graph_feature_cols)}")

    if save_dir and G is not None:
        save_dir = Path(save_dir)
        save_joblib(G, save_dir / "artist_graph.joblib")
        save_joblib(artist_metrics, save_dir / "artist_metrics.pkl")

    return df, G, artist_metrics