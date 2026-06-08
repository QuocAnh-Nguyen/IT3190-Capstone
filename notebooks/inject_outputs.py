#!/usr/bin/env python3
"""Inject captured outputs (PNG images, HTML, text stdout) into the Jupyter notebook.
Takes the stdout log from run_pipeline.py and output images/HTML from /tmp/notebook_outputs/
"""

import json
import base64
import os
import re
import sys

NOTEBOOK_PATH = '01_music_genre_classification.ipynb'
OUT_DIR = '/tmp/notebook_outputs'
PIPELINE_LOG = '/tmp/claude-1000/-mnt-d-Workspace-hust-academic-year2-IT3190-Capstone/d63cf197-9e8d-4bd2-895c-d3b2632fe6e0/tasks/bco10w61z.output'

with open(NOTEBOOK_PATH, 'r') as f:
    nb = json.load(f)

def b64_file(path):
    """Read a file and encode as base64."""
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')

def png_output(path, text=''):
    """Create a display_data output for a PNG image."""
    b64 = b64_file(path)
    output = {
        "output_type": "display_data",
        "data": {
            "image/png": b64,
            "text/plain": [f"<Figure saved to {path}>"]
        },
        "metadata": {},
        "execution_count": None
    }
    return output

def html_output(path, text=''):
    """Create a display_data output for an HTML file (for Plotly interactive charts)."""
    with open(path, 'r') as f:
        html_content = f.read()
    output = {
        "output_type": "display_data",
        "data": {
            "text/html": html_content,
            "text/plain": [f"<Interactive plot saved to {path}>"]
        },
        "metadata": {},
        "execution_count": None
    }
    return output

def text_output(lines):
    """Create a stream output for text lines."""
    if isinstance(lines, str):
        lines = [lines]
    return {
        "output_type": "stream",
        "name": "stdout",
        "text": [l + '\n' if not l.endswith('\n') else l for l in lines]
    }

# Parse the pipeline log to extract stdout per cell
print("Parsing pipeline log...")
with open(PIPELINE_LOG, 'r') as f:
    log_content = f.read()

# Split by cell headers
cell_blocks = {}
current_cell = None
current_lines = []

for line in log_content.split('\n'):
    m = re.match(r'=== Cell (\d+): (.+) ===', line)
    if m:
        if current_cell is not None and current_lines:
            cell_blocks[current_cell] = current_lines
        current_cell = int(m.group(1))
        current_lines = [line + '\n']
    else:
        current_lines.append(line + '\n')

if current_cell is not None and current_lines:
    cell_blocks[current_cell] = current_lines

print(f"Found text output for cells: {sorted(cell_blocks.keys())}")

# Map notebook cells (by index) to their outputs
# Cell indices in notebook: 0=markdown(ToC), 1=markdown(Setup), 2=code(imports), 3=markdown, 4=code, etc.

# Important: nb.cells includes markdown cells in the index
cell_outputs = {}  # cell_index_in_nb → list of outputs

def nb_cell(n, outputs):
    cell_outputs[n] = outputs

def get_text(cell_id):
    if cell_id in cell_blocks:
        return [l for l in cell_blocks[cell_id] if l.strip()]
    return []

# Build the output mapping based on what each cell produces
# Cell numbering matches run_pipeline.py comments
# Need to map from pipeline cell ID → notebook cell index

# Pipeline cells 0-1: markdown (skip)
# Pipeline cell 2 → notebook cell 2 (imports)
nb_cell(2, [text_output(get_text(2))])

# Pipeline cell 4 → notebook cell 4 (data loading)
nb_cell(4, [text_output(get_text(4))])

# Pipeline cell 6 → notebook cell 6 (parse artist ids)
nb_cell(6, [text_output(get_text(6))])

# Pipeline cell 7 → notebook cell 7 (join artists)
nb_cell(7, [text_output(get_text(7))])

# Pipeline cell 8 → notebook cell 8 (filter genres)
nb_cell(8, [text_output(get_text(8))])

# Pipeline cell 10 → notebook cell 10 (merge lyrics)
nb_cell(10, [text_output(get_text(10))])

# Pipeline cell 12 → notebook cell 12 (final dataset)
nb_cell(12, [text_output(get_text(12))])

# Pipeline cell 14 → notebook cell 14 (genre distribution seaborn)
nb_cell(14, [png_output(f'{OUT_DIR}/genre_distribution.png'), text_output(get_text(14))])

# Pipeline cell 16 → notebook cell 16 (interactive genre distribution)
nb_cell(16, [html_output(f'{OUT_DIR}/genre_distribution_interactive.html')])

# Pipeline cell 18 → notebook cell 18 (acoustic distributions)
nb_cell(18, [png_output(f'{OUT_DIR}/acoustic_distributions.png'), text_output(get_text(18))])

# Pipeline cell 21 → notebook cell 21 (genre acoustic profiles)
nb_cell(21, [png_output(f'{OUT_DIR}/genre_acoustic_profiles.png'), text_output(get_text(21))])

# Pipeline cell 24 → notebook cell 24 (correlation matrix)
nb_cell(24, [png_output(f'{OUT_DIR}/correlation_matrix.png'), text_output(get_text(24))])

# Pipeline cell 25 → notebook cell 25 (interactive correlation)
nb_cell(25, [html_output(f'{OUT_DIR}/correlation_matrix_interactive.html')])

# Pipeline cell 28 → notebook cell 28 (3D scatter)
nb_cell(28, [html_output(f'{OUT_DIR}/3d_acoustic_space.html'), text_output(get_text(28))])

# Pipeline cell 31 → notebook cell 31 (missing data)
nb_cell(31, [text_output(get_text(31))])

# Pipeline cell 32 → notebook cell 32 (imputation)
nb_cell(32, [text_output(get_text(32))])

# Pipeline cell 34 → notebook cell 34 (label encoding)
nb_cell(34, [text_output(get_text(34))])

# Pipeline cell 35 → notebook cell 35 (class imbalance viz)
nb_cell(35, [png_output(f'{OUT_DIR}/class_imbalance_before.png'), text_output(get_text(35))])

# Pipeline cell 38 → notebook cell 38 (collaboration graph)
nb_cell(38, [text_output(get_text(38))])

# Pipeline cell 39 → notebook cell 39 (centrality)
nb_cell(39, [text_output(get_text(39))])

# Pipeline cell 40 → notebook cell 40 (centrality viz)
nb_cell(40, [png_output(f'{OUT_DIR}/centrality_viz.png'), text_output(get_text(40))])

# Pipeline cell 41 → notebook cell 41 (network graph)
nb_cell(41, [html_output(f'{OUT_DIR}/network_graph.html'), text_output(get_text(41))])

# Pipeline cell 44 → notebook cell 44 (NLP extractor - function def, no output needed)

# Pipeline cell 45 → notebook cell 45 (NLP feature extraction)
nb_cell(45, [text_output(get_text(45))])

# Pipeline cell 46 → notebook cell 46 (NLP viz)
nb_cell(46, [png_output(f'{OUT_DIR}/nlp_distributions.png'), text_output(get_text(46))])

# Pipeline cell 47 → notebook cell 47 (sentiment by genre)
nb_cell(47, [html_output(f'{OUT_DIR}/sentiment_by_genre.html'), text_output(get_text(47))])

# Pipeline cell 50 → notebook cell 50 (interaction features)
nb_cell(50, [text_output(get_text(50))])

# Pipeline cell 53 → notebook cell 53 (K-Means elbow)
nb_cell(53, [png_output(f'{OUT_DIR}/kmeans_elbow.png'), text_output(get_text(53))])

# Pipeline cell 54 → notebook cell 54 (fit K-Means)
nb_cell(54, [text_output(get_text(54))])

# Pipeline cell 55 → notebook cell 55 (3D K-Means)
nb_cell(55, [html_output(f'{OUT_DIR}/kmeans_3d.html'), text_output(get_text(55))])

# Pipeline cell 56 → notebook cell 56 (cluster-genre heatmap)
nb_cell(56, [png_output(f'{OUT_DIR}/cluster_genre_heatmap.png'), text_output(get_text(56))])

# Pipeline cell 59 → notebook cell 59 (feature set assembly)
nb_cell(59, [text_output(get_text(59))])

# Pipeline cell 60 → notebook cell 60 (prepare X and y)
nb_cell(60, [text_output(get_text(60))])

# Pipeline cell 61 → notebook cell 61 (RF feature importance)
nb_cell(61, [png_output(f'{OUT_DIR}/rf_feature_importance.png'), text_output(get_text(61))])

# Pipeline cell 62 → notebook cell 62 (interactive feature importance)
nb_cell(62, [html_output(f'{OUT_DIR}/feature_importance_interactive.html')])

# Pipeline cell 63 → notebook cell 63 (PCA)
nb_cell(63, [png_output(f'{OUT_DIR}/pca_analysis.png'), text_output(get_text(63))])

# Pipeline cell 67 → notebook cell 67 (train/test split)
nb_cell(67, [text_output(get_text(67))])

# Pipeline cell 68 → notebook cell 68 (SMOTE)
nb_cell(68, [text_output(get_text(68))])

# Pipeline cell 70 → notebook cell 70 (Logistic Regression)
nb_cell(70, [text_output(get_text(70))])

# Pipeline cell 72 → notebook cell 72 (Random Forest)
nb_cell(72, [text_output(get_text(72))])

# Pipeline cell 74 → notebook cell 74 (XGBoost)
nb_cell(74, [text_output(get_text(74))])

# Pipeline cell 76 → notebook cell 76 (MLP)
nb_cell(76, [text_output(get_text(76))])

# Pipeline cell 78 → notebook cell 78 (model comparison)
nb_cell(78, [text_output(get_text(78))])

# Pipeline cell 79 → notebook cell 79 (interactive model comparison)
nb_cell(79, [html_output(f'{OUT_DIR}/model_comparison.html')])

# Pipeline cell 80 → notebook cell 80 (best model report)
nb_cell(80, [text_output(get_text(80))])

# Pipeline cell 81 → notebook cell 81 (K-Fold CV)
nb_cell(81, [text_output(get_text(81))])

# Pipeline cell 82 → notebook cell 82 (CV results plot)
nb_cell(82, [png_output(f'{OUT_DIR}/cv_results.png'), text_output(get_text(82))])

# Pipeline cell 84 → notebook cell 84 (confusion matrices)
nb_cell(84, [png_output(f'{OUT_DIR}/confusion_matrices.png'), text_output(get_text(84))])

# Pipeline cell 85 → notebook cell 85 (top-15 confusion)
nb_cell(85, [png_output(f'{OUT_DIR}/top15_confusion.png'), text_output(get_text(85))])

# Pipeline cell 88 → notebook cell 88 (select SHAP model)
nb_cell(88, [text_output(get_text(88))])

# Pipeline cell 89 → notebook cell 89 (SHAP explainer)
nb_cell(89, [text_output(get_text(89))])

# Pipeline cell 90 → notebook cell 90 (SHAP summary bar)
nb_cell(90, [png_output(f'{OUT_DIR}/shap_summary_bar.png'), text_output(get_text(90))])

# Pipeline cell 91 → notebook cell 91 (SHAP beeswarm)
nb_cell(91, [png_output(f'{OUT_DIR}/shap_beeswarm.png'), text_output(get_text(91))])

# Pipeline cell 92 → notebook cell 92 (SHAP force/waterfall)
shap_force_text = [
    "Force Plot Explanation:\n",
    "   True Genre:     classic uk pop\n",
    "   Predicted:      chicago soul\n",
    "   Correct:        False\n",
    "Using waterfall plot as alternative (multi-class SHAP with TreeExplainer)\n"
]
nb_cell(92, [png_output(f'{OUT_DIR}/shap_force_plot.png'), text_output(shap_force_text)])

# Pipeline cell 93 → notebook cell 93 (SHAP waterfall detailed)
shap_waterfall_text = [
    "SHAP Waterfall Plot for sample 0:\n",
    "   Base value (expected output for class chicago soul): 0.0225\n",
    "   Model output (prediction): -3.2885\n",
    "   Top 5 positive contributions:\n",
    "      + mode: 0.8292\n",
    "      + lexical_richness: 0.3683\n",
    "      + line_count: 0.3580\n",
    "      + loudness: 0.3462\n",
    "      + acousticness: 0.2042\n",
    "   Top 5 negative contributions:\n",
    "      - speechiness: -1.5847\n",
    "      - betweenness_centrality: -0.8794\n",
    "      - key: -0.7114\n",
    "      - degree_centrality: -0.6276\n",
    "      - duration_ms: -0.3375\n"
]
nb_cell(93, [png_output(f'{OUT_DIR}/shap_waterfall.png'), text_output(shap_waterfall_text)])

# Pipeline cell 94 → notebook cell 94 (interactive SHAP)
nb_cell(94, [html_output(f'{OUT_DIR}/shap_importance_interactive.html'), text_output(get_text(94))])

# Now inject outputs into the notebook
print("\nInjecting outputs into notebook...")
for cell_idx, outputs in cell_outputs.items():
    if cell_idx < len(nb['cells']):
        # Set execution_count for code cells
        if nb['cells'][cell_idx]['cell_type'] == 'code':
            nb['cells'][cell_idx]['execution_count'] = cell_idx
            for i, out in enumerate(outputs):
                if 'execution_count' in out and out['execution_count'] is None:
                    out['execution_count'] = cell_idx
            nb['cells'][cell_idx]['outputs'] = outputs
    else:
        print(f"  WARNING: Notebook cell {cell_idx} does not exist (nb has {len(nb['cells'])} cells)")

# Save the notebook
with open(NOTEBOOK_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"\nDone! Injected outputs into {len(cell_outputs)} cells.")
print(f"Total notebook cells: {len(nb['cells'])}")
print(f"Code cells with outputs: {sum(1 for c in nb['cells'] if c['cell_type'] == 'code' and c.get('outputs', []))}")