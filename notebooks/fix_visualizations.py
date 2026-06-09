#!/usr/bin/env python3
"""Quick fix: reload results from JSON and generate all remaining visualizations + outputs."""
import os, json, numpy as np, pandas as pd, gc
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; import seaborn as sns
import plotly.graph_objects as go
from sklearn.metrics import *

OUT_DIR = '/tmp/extended_model_outputs'

# Load saved results
with open(f'{OUT_DIR}/results.json') as f:
    rj = json.load(f)

# Also load predictions
preds_data = np.load(f'{OUT_DIR}/predictions.npz', allow_pickle=True)
Yte = preds_data['y_test']
preds = {k: preds_data[k] for k in preds_data.files if k != 'y_test'}
feature_names = pd.read_csv(f'{OUT_DIR}/feature_names.csv', header=None).iloc[:,0].tolist()

# Reconstruct results dict
results = {}
for k, v in rj.items():
    if k.startswith('_'):
        continue
    results[k] = v

best_name = rj['_best']
best_pred = preds[best_name]
cv_r = rj['_cv']

print(f"Loaded {len(results)} models, best={best_name}")
print(f"CV results for {len(cv_r)} models")

# Sort
sorted_m = sorted(results.items(), key=lambda x: x[1]['WF1'], reverse=True)

# ====== 2. Interactive Plotly comparison (FIXED: avoid duplicate 'Model' column) ======
print("Generating interactive comparison...")
# Build DataFrame manually to avoid duplicate column names
rdf_data = {}
for name, met in results.items():
    for mk, mv in met.items():
        if mk not in rdf_data:
            rdf_data[mk] = []
        rdf_data[mk].append(mv)
rdf = pd.DataFrame(rdf_data)
rdf = rdf.sort_values('WF1', ascending=True)

fig = go.Figure()
for m, color in [('Acc', '#3498db'), ('WF1', '#e74c3c'), ('MF1', '#2ecc71')]:
    fig.add_trace(go.Bar(
        name=m, y=rdf['Model'].tolist(), x=rdf[m].tolist(), orientation='h',
        text=[f'{v:.4f}' for v in rdf[m]], textposition='outside',
        textfont=dict(size=9), marker_color=color))
fig.update_layout(
    title='<b>Model Performance Comparison — All 16 Models</b><br><sup>35-Class Music Genre Classification | Multi-Modal Features</sup>',
    barmode='group', height=850, xaxis=dict(title='Score', range=[0, 0.85]),
    legend=dict(orientation='h', yanchor='bottom', y=1.02), margin=dict(l=220))
fig.write_html(f'{OUT_DIR}/model_comparison_interactive.html')
print("  [2/7] Interactive comparison ✔")

# ====== 3. Radar chart (top 7) ======
print("Generating radar chart...")
top_radar = sorted_m[:7]
radar_m = ['Acc', 'MP', 'MR', 'MF1', 'WF1']
fig = go.Figure()
for name, met in top_radar:
    vals = [met[m] for m in radar_m]
    vals.append(vals[0])
    fig.add_trace(go.Scatterpolar(r=vals, theta=radar_m + [radar_m[0]], name=name, fill='toself', opacity=0.45))
fig.update_layout(
    title='<b>Top 7 Models — Radar Comparison</b>',
    polar=dict(radialaxis=dict(range=[0, 0.75], tickfont=dict(size=9))), height=650,
    legend=dict(orientation='h', y=-0.1))
fig.write_html(f'{OUT_DIR}/model_radar.html')
print("  [3/7] Radar chart ✔")

# ====== 4. Time vs Performance scatter ======
print("Generating time vs perf...")
fig, ax = plt.subplots(figsize=(12, 8))
names = [m[0] for m in sorted_m]
f1s = [m[1]['WF1'] for m in sorted_m]
times = [max(m[1]['Time'], 0.1) for m in sorted_m]
sc = ax.scatter(times, f1s, s=150, c=range(len(names)), cmap='viridis_r', edgecolors='black', linewidth=1, zorder=5)
for i, name in enumerate(names):
    ax.annotate(name.split('(')[0].strip(), (times[i], f1s[i]), xytext=(8, 5),
                textcoords='offset points', fontsize=7.5, alpha=0.85)
ax.set_xlabel('Training Time (seconds, log scale)', fontsize=12)
ax.set_ylabel('Weighted F1', fontsize=12)
ax.set_xscale('log')
ax.set_title('Model Performance vs Training Time', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.colorbar(sc, ax=ax, label='Rank (1=best)')
# Pareto frontier
pareto_t = []; pareto_f = []; best_f1 = 0
for m in sorted_m:
    t_ = max(m[1]['Time'], 0.1); f_ = m[1]['WF1']
    if f_ > best_f1: best_f1 = f_; pareto_t.append(t_); pareto_f.append(f_)
if pareto_t:
    ax.plot(pareto_t, pareto_f, 'r--', linewidth=2, alpha=0.6, label='Pareto frontier')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/time_vs_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [4/7] Time vs Performance ✔")

# ====== 5. CV results ======
print("Generating CV comparison...")
fig, ax = plt.subplots(figsize=(14, 6))
cn = list(cv_r.keys())
cm = [cv_r[n]['mean'] for n in cn]
cs = [cv_r[n]['std'] for n in cn]
colors_cv = plt.cm.viridis(np.linspace(0.15, 0.95, len(cn)))
bars = ax.barh(range(len(cn)), cm, xerr=cs, capsize=8, color=colors_cv, edgecolor='white', linewidth=1.5)
ax.set_yticks(range(len(cn)))
ax.set_yticklabels(cn, fontsize=10)
ax.set_xlabel('Weighted F1 Score', fontsize=12)
ax.invert_yaxis()
ax.set_title('5-Fold Cross-Validation — Weighted F1 Scores', fontsize=14, fontweight='bold')
for bar, mean, std in zip(bars, cm, cs):
    ax.text(mean + std + 0.005, bar.get_y() + bar.get_height() / 2,
            f'{mean:.4f} ±{std:.4f}', va='center', fontsize=9, fontweight='bold')
ax.set_xlim(0, max(cm) + max(cs) + 0.08)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/cv_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [5/7] CV comparison ✔")

# ====== 6. Confusion matrix (best model, top 12 genres) ======
print("Generating confusion matrix...")
import pathlib, re, ast
from sklearn.preprocessing import LabelEncoder
DATA_PATH = str(pathlib.Path(__file__).resolve().parent.parent / 'data' / 'raw')
artists = pd.read_csv(f'{DATA_PATH}/musicoset_metadata/artists.csv', sep='\t')
hits = pd.read_csv(f'{DATA_PATH}/additional/hits_dataset.csv', sep='\t')

def parse_artist_ids(id_str):
    try: ids = ast.literal_eval(id_str); return ids if isinstance(ids, list) else [ids]
    except: return []
hits['artist_id_list'] = hits['id_artists'].apply(parse_artist_ids)
hits['primary_artist_id'] = hits['artist_id_list'].apply(lambda x: x[0] if len(x) > 0 else None)
hits['num_artists_parsed'] = hits['artist_id_list'].apply(len)
df = hits.merge(artists[['artist_id','main_genre','genres','followers','popularity']].rename(
    columns={'popularity':'artist_popularity','followers':'artist_followers'}),
    left_on='primary_artist_id', right_on='artist_id', how='left')
df = df[df['main_genre'].notna() & (df['main_genre'] != '-')].copy()
valid_genres = df['main_genre'].value_counts()
valid_genres = valid_genres[valid_genres >= 50].index.tolist()
df = df[df['main_genre'].isin(valid_genres)].copy()

le = LabelEncoder(); le.fit(df['main_genre'])
Yte_labels = le.inverse_transform(Yte.astype(int))

top12 = df['main_genre'].value_counts().head(12).index
t12m = np.isin(Yte_labels, top12)

if t12m.sum() > 50:
    yts = Yte[t12m]; yps = best_pred[t12m]
    sle = LabelEncoder()
    sbl = le.inverse_transform(yts); yte_enc = sle.fit_transform(sbl)
    ypl = le.inverse_transform(yps); vpm = np.isin(ypl, list(sle.classes_))
    yte_f = yte_enc[vpm]; ype_f = sle.transform(ypl[vpm])
    cm = confusion_matrix(yte_f, ype_f)
    cmn = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(15, 13))
    sns.heatmap(cmn, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                xticklabels=sle.classes_, yticklabels=sle.classes_,
                linewidths=0.5, linecolor='white', cbar_kws={'label': 'Proportion'},
                vmin=0, vmax=1, annot_kws={'fontsize': 8})
    ax.set_title(f'Confusion Matrix — {best_name} (Top 12 Genres)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicted Genre', fontsize=12)
    ax.set_ylabel('True Genre', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/best_model_confusion.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [6/7] Confusion matrix ✔")
else:
    print("  [6/7] Confusion matrix skipped (too few samples)")

# ====== 7. Feature importance (manual from best_model_obj) ======
print("Generating feature importance...")
# We need to find which model was best and extract its feature_importances_
# Since we saved predictions, we can identify the best model name
if 'XGBoost' in best_name or 'LightGBM' in best_name:
    # Feature importance isn't directly available from JSON, but we can approximate
    # For now, use a text-based approach from the model type info
    print(f"  Best model is {best_name} - feature importance requires model object")
    print("  [7/7] Feature importance skipped (needs model object, not in JSON)")

# ====== 8. Category comparison ======
print("Generating category comparison...")
categories = {
    'Linear/Simple': ['Logistic Regression', 'SVM (Linear)'],
    'Tree-based': ['Decision Tree', 'Random Forest', 'Extra Trees'],
    'Gradient Boosting': ['HistGradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost',
                          'XGBoost (Tuned)', 'LightGBM (Tuned)', 'CatBoost (Tuned)'],
    'Neural Network': ['MLP Neural Net'],
    'Ensemble': ['Stacking Ensemble', 'Voting Ensemble'],
    'Distance-based': ['KNN (k=15)'],
}
cat_results = {}
for cat, model_names in categories.items():
    cat_vals = {n: results[n] for n in model_names if n in results}
    if cat_vals:
        best_in_cat = max(cat_vals.items(), key=lambda x: x[1]['WF1'])
        cat_results[cat] = {'best_model': best_in_cat[0], 'WF1': best_in_cat[1]['WF1'],
                            'Acc': best_in_cat[1]['Acc'], 'count': len(cat_vals)}

fig, ax = plt.subplots(figsize=(12, 6))
cats = list(cat_results.keys())
wf1s = [cat_results[c]['WF1'] for c in cats]
accs = [cat_results[c]['Acc'] for c in cats]
x = np.arange(len(cats))
w = 0.35
b1 = ax.bar(x - w/2, wf1s, w, label='Weighted F1', color='#e74c3c', edgecolor='white')
b2 = ax.bar(x + w/2, accs, w, label='Accuracy', color='#3498db', edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(cats, fontsize=10)
for bar, val in zip(b1, wf1s):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
            ha='center', fontsize=9, fontweight='bold')
for bar, val in zip(b2, accs):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
            ha='center', fontsize=9)
ax.set_ylabel('Score', fontsize=12)
ax.legend()
ax.set_ylim(0, 0.85)
ax.set_title('Best Model Performance by Category', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/category_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [8/8] Category comparison ✔")

# ====== FINAL REPORT ======
print("\n" + "=" * 70)
print("FINAL MODEL RANKINGS (by Weighted F1)")
print("=" * 70)
for rank, (name, met) in enumerate(sorted_m, 1):
    star = " ⭐" if rank == 1 else "  "
    bar_str = "█" * int(met['WF1'] * 40)
    print(f"{star} {rank:2d}. {name:<28s} | WF1={met['WF1']:.4f} Acc={met['Acc']:.4f} "
          f"MF1={met['MF1']:.4f} Time={met['Time']:.0f}s {bar_str}")

print(f"\n{'='*70}")
print(f"BEST MODEL: {best_name}")
print(f"{'='*70}")
print(f"  Accuracy:          {results[best_name]['Acc']:.4f}")
print(f"  Macro F1:          {results[best_name]['MF1']:.4f}")
print(f"  Weighted F1:       {results[best_name]['WF1']:.4f}")
print(f"  Macro Precision:   {results[best_name]['MP']:.4f}")
print(f"  Macro Recall:      {results[best_name]['MR']:.4f}")
print(f"  Training Time:     {results[best_name]['Time']:.1f}s")

report = classification_report(Yte.astype(int), best_pred.astype(int),
                               target_names=le.classes_, digits=3, output_dict=True)
rdf_report = pd.DataFrame(report).T
genre_rows = rdf_report.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore').sort_values('f1-score', ascending=False)
print("\n  BEST 10 GENRES:")
for idx, row in genre_rows.head(10).iterrows():
    print(f"    {idx:<25s} F1={row['f1-score']:.3f} P={row['precision']:.3f} R={row['recall']:.3f} sup={int(row['support'])}")
print("\n  WORST 10 GENRES:")
for idx, row in genre_rows.tail(10).iterrows():
    print(f"    {idx:<25s} F1={row['f1-score']:.3f} P={row['precision']:.3f} R={row['recall']:.3f} sup={int(row['support'])}")

# Save full classification report
with open(f'{OUT_DIR}/best_model_report.txt', 'w') as f:
    f.write(f"BEST MODEL: {best_name}\n{'='*60}\n")
    for k, v in results[best_name].items():
        f.write(f"{k}: {v}\n")
    f.write(f"\nCV RESULTS:\n")
    for k, v in cv_r.items():
        f.write(f"  {k}: {v['mean']:.4f} +/- {v['std']:.4f}\n")
    f.write(f"\nFULL CLASSIFICATION REPORT:\n")
    f.write(classification_report(Yte.astype(int), best_pred.astype(int),
                                  target_names=le.classes_, digits=3))

print(f"\n{'='*70}")
print(f"ALL DONE! Outputs in {OUT_DIR}/")
print(f"{'='*70}")
for f in sorted(os.listdir(OUT_DIR)):
    sz = os.path.getsize(os.path.join(OUT_DIR, f))
    print(f"  {f} ({sz:>10,} bytes)")