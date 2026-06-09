#!/usr/bin/env python3
"""
FINAL ROBUST COMPARISON PIPELINE - All bugs fixed.
Saves intermediate results before visualizations so crashes don't lose everything.
"""
import sys, os, time, json, warnings, gc, pathlib, re, ast
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; import seaborn as sns
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, StackingClassifier, VotingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTE
import xgboost as xgb; import lightgbm as lgb; import catboost as cb
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
from optuna.samplers import TPESampler
from textblob import TextBlob; import networkx as nx

OUT_DIR = str(pathlib.Path(__file__).resolve().parent.parent / 'outputs' / 'model_comparison')
os.makedirs(OUT_DIR, exist_ok=True)
print("=" * 60)
print("FINAL ROBUST COMPARISON PIPELINE")
print("=" * 60)

# ====== PHASE 1: DATA PIPELINE ======
DATA_PATH = str(pathlib.Path(__file__).resolve().parent.parent / 'data' / 'raw')
songs = pd.read_csv(f'{DATA_PATH}/musicoset_metadata/songs.csv', sep='\t')
artists = pd.read_csv(f'{DATA_PATH}/musicoset_metadata/artists.csv', sep='\t')
lyrics = pd.read_csv(f'{DATA_PATH}/musicoset_songfeatures/lyrics.csv', sep='\t')
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
valid_genres = valid_genres[valid_genres>=50].index.tolist()
df = df[df['main_genre'].isin(valid_genres)].copy()
df = df.merge(lyrics[['song_id','lyrics']], on='song_id', how='left')
print(f"Dataset: {len(df):,} rows, {df['main_genre'].nunique()} genres")

ACOUSTIC_FEATURES = ['duration_ms','key','mode','time_signature',
                     'acousticness','danceability','energy','instrumentalness',
                     'liveness','loudness','speechiness','valence','tempo']
df['duration_sec']=df['duration_ms']/1000; df['has_lyrics']=df['lyrics'].notna().astype(int)
df['explicit']=df['explicit'].astype(int)
artist_cols=['artist_popularity','artist_followers']
if df[artist_cols].isnull().sum().sum()>0:
    imp=IterativeImputer(max_iter=10,random_state=42); df[artist_cols]=imp.fit_transform(df[artist_cols])
le=LabelEncoder(); df['genre_encoded']=le.fit_transform(df['main_genre'])

# Graph
print("Building graph...")
G=nx.Graph()
for _,row in songs.iterrows():
    try:
        ad=ast.literal_eval(row['artists']); aids=list(ad.keys())
        if len(aids)>1:
            for i in range(len(aids)):
                for j in range(i+1,len(aids)):
                    if G.has_edge(aids[i],aids[j]): G[aids[i]][aids[j]]['weight']+=1
                    else: G.add_edge(aids[i],aids[j],weight=1)
        else: G.add_node(aids[0])
    except: pass
dc=nx.degree_centrality(G); bc=nx.betweenness_centrality(G,k=min(500,G.number_of_nodes()))
cc=nx.closeness_centrality(G); clust=nx.clustering(G)
try: ec=nx.eigenvector_centrality_numpy(G,max_iter=200)
except: ec=nx.eigenvector_centrality(G,max_iter=200)
df['degree_centrality']=df['primary_artist_id'].map(dc).fillna(0)
df['betweenness_centrality']=df['primary_artist_id'].map(bc).fillna(0)
df['closeness_centrality']=df['primary_artist_id'].map(cc).fillna(0)
df['clustering_coeff']=df['primary_artist_id'].map(clust).fillna(0)
df['eigenvector_centrality']=df['primary_artist_id'].map(ec).fillna(0)

# NLP
print("NLP...")
def extract_nlp(lt):
    if pd.isna(lt) or lt=='': return [0]*10
    cl=re.sub(r'\[.*?\]','',str(lt)); ws=cl.lower().split(); wc=len(ws); uwc=len(set(ws))
    lr=uwc/max(wc,1); awl=np.mean([len(w) for w in ws]) if ws else 0
    ls=str(lt).split('\n'); lc=len([l for l in ls if l.strip()])
    if wc>10: b=TextBlob(cl); sp=b.sentiment.polarity; ss=b.sentiment.subjectivity
    else: sp,ss=0,0
    tl=str(lt).lower(); hv=1 if 'verse' in tl else 0; hc=1 if 'chorus' in tl else 0; hb=1 if 'bridge' in tl else 0
    return [wc,uwc,lr,awl,lc,sp,ss,hv,hc,hb]
NLP_COLS=['word_count','unique_word_count','lexical_richness','avg_word_length','line_count',
          'sentiment_polarity','sentiment_subjectivity','lyrics_has_verse','lyrics_has_chorus','lyrics_has_bridge']
nlp_arr=np.array(df['lyrics'].apply(extract_nlp).tolist())
for i,c in enumerate(NLP_COLS): df[c]=nlp_arr[:,i]

# Interaction
int_feats=['energy','danceability','valence','acousticness','loudness','tempo']
sp_scaler=StandardScaler(); df_int_s=sp_scaler.fit_transform(df[int_feats])
poly=PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)
pf=poly.fit_transform(df_int_s); pfn=poly.get_feature_names_out(int_feats)
int_names=[n for n in pfn if ' ' in n]; int_idx=[i for i,n in enumerate(pfn) if ' ' in n]
int_mat=pf[:,int_idx]
INTERACTION_COLS=[]
for i,name in enumerate(int_names):
    cn='inter_'+name.replace(' ','_x_'); df[cn]=int_mat[:,i]; INTERACTION_COLS.append(cn)

# K-Means + PCA
kms=StandardScaler(); ac_s=kms.fit_transform(df[ACOUSTIC_FEATURES])
df['acoustic_cluster']=KMeans(n_clusters=8,random_state=42,n_init=20).fit_predict(ac_s)
pca=PCA(n_components=10,random_state=42); pca_f=pca.fit_transform(StandardScaler().fit_transform(df[ACOUSTIC_FEATURES]))
PCA_COLS=[]
for i in range(10): cn=f'pca_acoustic_{i+1}'; df[cn]=pca_f[:,i]; PCA_COLS.append(cn)

# Feature matrix
NETWORK_COLS=['degree_centrality','betweenness_centrality','closeness_centrality','clustering_coeff','eigenvector_centrality']
META_COLS=['popularity','artist_popularity','artist_followers','num_artists','explicit','has_lyrics','duration_sec']
all_fc = ACOUSTIC_FEATURES + META_COLS + NETWORK_COLS + NLP_COLS + INTERACTION_COLS + ['acoustic_cluster'] + PCA_COLS
fdf=df[all_fc].copy(); fdf=fdf.replace([np.inf,-np.inf],np.nan).fillna(0)
X,Y = fdf.values, df['genre_encoded'].values
scaler = StandardScaler(); Xs = scaler.fit_transform(X)
print(f"Feature matrix: {Xs.shape} | {len(all_fc)} features")

Xtr,Xte,Ytr,Yte = train_test_split(Xs,Y,test_size=0.2,random_state=42,stratify=Y)
mc = np.min(np.bincount(Ytr)); kn = min(max(mc-1,1),5)
smt = SMOTE(random_state=42,k_neighbors=kn); Xtr_s,Ytr_s = smt.fit_resample(Xtr,Ytr)
print(f"Train (SMOTE): {Xtr_s.shape}, Test: {Xte.shape}")

# ====== PHASE 2: TRAIN 11 BASE MODELS ======
print("\n" + "=" * 60)
print("TRAINING 11 BASE MODELS")
print("=" * 60)
results={}; preds={}; base_models={}

def ev(name,yp,tt):
    return {'Model':name, 'Acc':accuracy_score(Yte,yp),
            'MP':precision_score(Yte,yp,average='macro',zero_division=0),
            'MR':recall_score(Yte,yp,average='macro',zero_division=0),
            'MF1':f1_score(Yte,yp,average='macro',zero_division=0),
            'WF1':f1_score(Yte,yp,average='weighted',zero_division=0),
            'WP':precision_score(Yte,yp,average='weighted',zero_division=0),
            'WR':recall_score(Yte,yp,average='weighted',zero_division=0),
            'Time':tt}

models_config = [
    ('Logistic Regression', LogisticRegression(max_iter=2000,random_state=42,n_jobs=-1)),
    ('KNN (k=15)', KNeighborsClassifier(n_neighbors=15,weights='distance',n_jobs=-1)),
    ('SVM (Linear)', None),
    ('Decision Tree', DecisionTreeClassifier(max_depth=20,min_samples_split=10,random_state=42)),
    ('Random Forest', RandomForestClassifier(n_estimators=300,max_depth=25,min_samples_split=5,
                                              max_features='sqrt',class_weight='balanced',random_state=42,n_jobs=-1)),
    ('Extra Trees', ExtraTreesClassifier(n_estimators=300,max_depth=25,min_samples_split=5,
                                          max_features='sqrt',random_state=42,n_jobs=-1)),
    ('HistGradient Boosting', HistGradientBoostingClassifier(max_iter=200,max_depth=8,learning_rate=0.1,
                                                              random_state=42,early_stopping=False)),
    ('XGBoost', xgb.XGBClassifier(n_estimators=300,max_depth=8,learning_rate=0.1,subsample=0.8,
                                   colsample_bytree=0.8,objective='multi:softprob',random_state=42,
                                   n_jobs=-1,eval_metric='mlogloss')),
    ('LightGBM', lgb.LGBMClassifier(n_estimators=300,max_depth=8,learning_rate=0.1,subsample=0.8,
                                     colsample_bytree=0.8,random_state=42,n_jobs=-1,verbose=-1,
                                     num_leaves=127,min_child_samples=20)),
    ('CatBoost', cb.CatBoostClassifier(iterations=300,depth=8,learning_rate=0.1,random_seed=42,
                                        thread_count=-1,verbose=False,allow_writing_files=False)),
    ('MLP Neural Net', MLPClassifier(hidden_layer_sizes=(256,128,64),activation='relu',solver='adam',
                                      alpha=0.001,batch_size=128,learning_rate='adaptive',
                                      learning_rate_init=0.001,max_iter=300,early_stopping=True,
                                      validation_fraction=0.1,n_iter_no_change=10,random_state=42)),
]

for name,model in models_config:
    print(f"  {name}...",end=' ',flush=True)
    if name=='SVM (Linear)':
        t0=time.time()
        n_svm=min(15000,len(Xtr_s)); idx=np.random.choice(len(Xtr_s),n_svm,replace=False)
        svm_b=LinearSVC(C=1.0,max_iter=2000,random_state=42,dual=False)
        svm_c=CalibratedClassifierCV(svm_b,cv=3); svm_c.fit(Xtr_s[idx],Ytr_s[idx])
        yp=svm_c.predict(Xte); tt=time.time()-t0; base_models[name]=svm_c
    elif name=='KNN (k=15)':
        t0=time.time()
        n_knn=min(10000,len(Xtr_s)); idx=np.random.choice(len(Xtr_s),n_knn,replace=False)
        model.fit(Xtr_s[idx],Ytr_s[idx]); yp=model.predict(Xte); tt=time.time()-t0; base_models[name]=model
    elif name=='CatBoost':
        t0=time.time(); model.fit(Xtr_s,Ytr_s); yp=model.predict(Xte).flatten(); tt=time.time()-t0; base_models[name]=model
    else:
        t0=time.time(); model.fit(Xtr_s,Ytr_s); yp=model.predict(Xte); tt=time.time()-t0; base_models[name]=model
    results[name]=ev(name,yp,tt); preds[name]=yp
    print(f"WF1={results[name]['WF1']:.4f} ({tt:.1f}s)")
    gc.collect()
print("✓ 11 base models done")

# ====== SAVE CHECKPOINT 1 ======
print("\nSaving checkpoint 1...")
np.savez(f'{OUT_DIR}/predictions.npz', y_test=Yte, **{k:v for k,v in preds.items()})
pd.Series(all_fc).to_csv(f'{OUT_DIR}/feature_names.csv',index=False)
print("✓ Checkpoint 1 saved")

# ====== PHASE 3: OPTUNA TUNING ======
print("\n" + "=" * 60)
print("OPTUNA TUNING (top 3, 5 trials each)")
print("=" * 60)
n_tune=min(6000,len(Xtr_s)); idx_t=np.random.choice(len(Xtr_s),n_tune,replace=False)
Xtu=Xtr_s[idx_t]; Ytu=Ytr_s[idx_t]
Xtt,Xtv,Ytt,Ytv = train_test_split(Xtu,Ytu,test_size=0.2,random_state=42,stratify=Ytu)
tuned_params = {}

for tune_name, obj_fn, factory_fn in [
    ('XGBoost',
     lambda t: {'n_estimators':t.suggest_int('n_estimators',100,500),'max_depth':t.suggest_int('max_depth',3,12),
                'learning_rate':t.suggest_float('learning_rate',0.01,0.3,log=True),
                'subsample':t.suggest_float('subsample',0.6,1.0),'colsample_bytree':t.suggest_float('colsample_bytree',0.6,1.0),
                'min_child_weight':t.suggest_int('min_child_weight',1,10),
                'reg_alpha':t.suggest_float('reg_alpha',1e-8,1.0,log=True),
                'reg_lambda':t.suggest_float('reg_lambda',1e-8,1.0,log=True)},
     lambda p: xgb.XGBClassifier(**p,objective='multi:softprob',random_state=42,n_jobs=-1,eval_metric='mlogloss')),
    ('LightGBM',
     lambda t: {'n_estimators':t.suggest_int('n_estimators',100,500),'max_depth':t.suggest_int('max_depth',3,12),
                'learning_rate':t.suggest_float('learning_rate',0.01,0.3,log=True),
                'subsample':t.suggest_float('subsample',0.6,1.0),'colsample_bytree':t.suggest_float('colsample_bytree',0.6,1.0),
                'num_leaves':t.suggest_int('num_leaves',15,255),'min_child_samples':t.suggest_int('min_child_samples',5,50),
                'reg_alpha':t.suggest_float('reg_alpha',1e-8,1.0,log=True),
                'reg_lambda':t.suggest_float('reg_lambda',1e-8,1.0,log=True)},
     lambda p: lgb.LGBMClassifier(**p,random_state=42,n_jobs=-1,verbose=-1)),
    ('CatBoost',
     lambda t: {'iterations':t.suggest_int('iterations',100,500),'depth':t.suggest_int('depth',3,10),
                'learning_rate':t.suggest_float('learning_rate',0.01,0.3,log=True),
                'l2_leaf_reg':t.suggest_float('l2_leaf_reg',1e-2,10,log=True)},
     lambda p: cb.CatBoostClassifier(**p,random_seed=42,thread_count=-1,verbose=False,allow_writing_files=False)),
]:
    print(f"  Tuning {tune_name}...",end=' ',flush=True)
    def make_obj(fn, factory):
        def objective(trial):
            p=fn(trial); m=factory(p)
            if 'CatBoost' in str(type(m)):
                m.fit(Xtt,Ytt); return f1_score(Ytv,m.predict(Xtv).flatten(),average='weighted')
            else:
                m.fit(Xtt,Ytt); return f1_score(Ytv,m.predict(Xtv),average='weighted')
        return objective
    st=optuna.create_study(direction='maximize',sampler=TPESampler(seed=42))
    st.optimize(make_obj(obj_fn,factory_fn),n_trials=5,show_progress_bar=False)
    bp=st.best_params; bs=st.best_value
    t0=time.time()
    fm = factory_fn(bp)
    if 'CatBoost' in str(type(fm)):
        fm.fit(Xtr_s,Ytr_s); yp=fm.predict(Xte).flatten()
    else:
        fm.fit(Xtr_s,Ytr_s); yp=fm.predict(Xte)
    tt=time.time()-t0
    tuned_name = f'{tune_name} (Tuned)'
    results[tuned_name]=ev(tuned_name,yp,tt); preds[tuned_name]=yp
    tuned_params[tune_name]=bp
    print(f"CV={bs:.4f} → test WF1={results[tuned_name]['WF1']:.4f}")
    gc.collect()
print("✓ Tuning done")

# ====== SAVE CHECKPOINT 2 ======
print("\nSaving checkpoint 2...")
rj = {}
for name,met in results.items():
    rj[name] = {k:float(v) if isinstance(v,(np.floating,float,np.integer,int)) else v for k,v in met.items()}
rj['_tuned'] = {k:{pk:(int(pv) if isinstance(pv,np.integer) else float(pv) if isinstance(pv,np.floating) else pv)
                    for pk,pv in v.items()} for k,v in tuned_params.items()}
with open(f'{OUT_DIR}/results.json','w') as f: json.dump(rj,f,indent=2)
np.savez(f'{OUT_DIR}/predictions.npz', y_test=Yte, **{k:v for k,v in preds.items()})
print("✓ Checkpoint 2 saved")

# ====== PHASE 4: ENSEMBLES ======
print("\n" + "=" * 60)
print("ENSEMBLES")
print("=" * 60)
# Use base models for ensemble (trained earlier in PHASE 2)
base_est = [('xgb', base_models['XGBoost']), ('lgb', base_models['LightGBM']),
            ('cb', base_models['CatBoost'])]

print("  Stacking ensemble...",end=' ',flush=True)
n_st = min(12000,len(Xtr_s)); idx_st=np.random.choice(len(Xtr_s),n_st,replace=False)
t0=time.time()
stacking = StackingClassifier(estimators=base_est,
                              final_estimator=LogisticRegression(max_iter=1000,random_state=42),
                              cv=3,n_jobs=-1,passthrough=False)
stacking.fit(Xtr_s[idx_st],Ytr_s[idx_st]); yp_s=stacking.predict(Xte); tt_s=time.time()-t0
results['Stacking Ensemble']=ev('Stacking Ensemble',yp_s,tt_s); preds['Stacking Ensemble']=yp_s
print(f"WF1={results['Stacking Ensemble']['WF1']:.4f} ({tt_s:.1f}s)")

print("  Voting ensemble...",end=' ',flush=True)
t0=time.time()
voting = VotingClassifier(estimators=base_est,voting='soft',weights=[2,2,1])
voting.fit(Xtr_s[idx_st],Ytr_s[idx_st]); yp_v=voting.predict(Xte); tt_v=time.time()-t0
results['Voting Ensemble']=ev('Voting Ensemble',yp_v,tt_v); preds['Voting Ensemble']=yp_v
print(f"WF1={results['Voting Ensemble']['WF1']:.4f} ({tt_v:.1f}s)")
gc.collect()

# ====== PHASE 5: 5-FOLD CV ======
print("\n" + "=" * 60)
print("5-FOLD CV (7 key models)")
print("=" * 60)
skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
n_cv=min(5000,len(Xs)); idx_cv=np.random.choice(len(Xs),n_cv,replace=False)
Xcv=Xs[idx_cv]; Ycv=Y[idx_cv]
cv_r={}
cv_models={
    'Logistic Regression':LogisticRegression(max_iter=2000,random_state=42,n_jobs=-1),
    'Random Forest':RandomForestClassifier(n_estimators=100,max_depth=15,random_state=42,n_jobs=-1),
    'XGBoost':xgb.XGBClassifier(n_estimators=100,max_depth=6,learning_rate=0.1,
                                 objective='multi:softprob',random_state=42,n_jobs=-1,eval_metric='mlogloss'),
    'LightGBM':lgb.LGBMClassifier(n_estimators=100,max_depth=6,learning_rate=0.1,
                                   random_state=42,n_jobs=-1,verbose=-1),
    'CatBoost':cb.CatBoostClassifier(iterations=100,depth=6,learning_rate=0.1,
                                       random_seed=42,thread_count=-1,verbose=False,allow_writing_files=False),
    'HistGradientBoosting':HistGradientBoostingClassifier(max_iter=100,max_depth=8,random_state=42),
    'Extra Trees':ExtraTreesClassifier(n_estimators=100,max_depth=15,random_state=42,n_jobs=-1),
}
for name,model in cv_models.items():
    scores=[]
    for ti,vi in skf.split(Xcv,Ycv):
        xft,xfv=Xcv[ti],Xcv[vi]; yft,yfv=Ycv[ti],Ycv[vi]
        mc=np.min(np.bincount(yft)); k=min(max(mc-1,1),3)
        sf=SMOTE(random_state=42,k_neighbors=k); xfs,yfs=sf.fit_resample(xft,yft)
        if 'CatBoost' in name:
            model.fit(xfs,yfs); yfp=model.predict(xfv).flatten()
        else:
            model.fit(xfs,yfs); yfp=model.predict(xfv)
        scores.append(f1_score(yfv,yfp,average='weighted'))
    cv_r[name]={'mean':np.mean(scores),'std':np.std(scores),'scores':scores}
    print(f"  {name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    gc.collect()

# ====== SAVE CHECKPOINT 3 (EVERYTHING before viz) ======
print("\nSaving checkpoint 3 (all results)...")
rj = {}
for name,met in results.items():
    rj[name] = {k:float(v) if isinstance(v,(np.floating,float,np.integer,int)) else v for k,v in met.items()}
rj['_cv'] = {k:{'mean':float(v['mean']),'std':float(v['std'])} for k,v in cv_r.items()}
rj['_best'] = None  # will be filled
rj['_tuned'] = {k:{pk:(int(pv) if isinstance(pv,np.integer) else float(pv) if isinstance(pv,np.floating) else pv)
                    for pk,pv in v.items()} for k,v in tuned_params.items()}
with open(f'{OUT_DIR}/results.json','w') as f: json.dump(rj,f,indent=2)
np.savez(f'{OUT_DIR}/predictions.npz', y_test=Yte, **{k:v for k,v in preds.items()})
pd.Series(all_fc).to_csv(f'{OUT_DIR}/feature_names.csv',index=False)
print("✓ Checkpoint 3 saved")

# ====== PHASE 6: ALL VISUALIZATIONS ======
print("\n" + "=" * 60)
print("VISUALIZATIONS")
print("=" * 60)

sorted_m = sorted(results.items(), key=lambda x: x[1]['WF1'], reverse=True)
best_name = sorted_m[0][0]; best_pred = preds[best_name]
rj['_best'] = best_name
with open(f'{OUT_DIR}/results.json','w') as f: json.dump(rj,f,indent=2)

# 1. Static comparison (3-panel)
print("  [1/8] Static comparison...")
fig,axes=plt.subplots(1,3,figsize=(26,9))
cl=plt.cm.viridis(np.linspace(0.15,0.95,len(sorted_m)))
for idx,(metric,title) in enumerate(zip(['Acc','WF1','MF1'],['Accuracy','Weighted F1','Macro F1'])):
    names=[m[0] for m in sorted_m]; vals=[m[1][metric] for m in sorted_m]
    ax=axes[idx]; bars=ax.barh(range(len(names)),vals,color=cl)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names,fontsize=8)
    ax.set_xlabel(metric,fontsize=11); ax.set_title(title,fontsize=13,fontweight='bold')
    ax.set_xlim(0,max(vals)*1.15)
    for i,(bar,val) in enumerate(zip(bars,vals)):
        ax.text(val+0.005,bar.get_y()+bar.get_height()/2,f'{val:.4f}',va='center',fontsize=8,fontweight='bold')
plt.suptitle('Comprehensive Model Comparison — 35-Class Music Genre Classification\n(16 models across 3 key metrics)',fontsize=16,fontweight='bold',y=1.02)
plt.tight_layout(); plt.savefig(f'{OUT_DIR}/model_comparison_all.png',dpi=150,bbox_inches='tight'); plt.close()
print("    ✓")

# 2. Interactive Plotly (FIXED: no duplicate columns)
print("  [2/8] Interactive Plotly...")
rdf = pd.DataFrame({'Model': [m[0] for m in sorted_m],
                     'Acc': [m[1]['Acc'] for m in sorted_m],
                     'WF1': [m[1]['WF1'] for m in sorted_m],
                     'MF1': [m[1]['MF1'] for m in sorted_m]})
rdf = rdf.sort_values('WF1', ascending=True)
fig=go.Figure()
for m,color in [('Acc','#3498db'),('WF1','#e74c3c'),('MF1','#2ecc71')]:
    fig.add_trace(go.Bar(name=m, y=rdf['Model'].tolist(), x=rdf[m].tolist(), orientation='h',
                         text=[f'{v:.4f}' for v in rdf[m]], textposition='outside',
                         textfont=dict(size=9), marker_color=color))
fig.update_layout(title='<b>Model Performance Comparison — All 16 Models</b><br><sup>35-Class Music Genre Classification | Multi-Modal Features</sup>',
                  barmode='group', height=850, xaxis=dict(title='Score', range=[0,0.85]),
                  legend=dict(orientation='h', yanchor='bottom', y=1.02), margin=dict(l=220))
fig.write_html(f'{OUT_DIR}/model_comparison_interactive.html')
print("    ✓")

# 3. Radar chart
print("  [3/8] Radar chart...")
top_radar=sorted_m[:7]; radar_m=['Acc','MP','MR','MF1','WF1']
fig=go.Figure()
for name,met in top_radar:
    vals=[met[m] for m in radar_m]; vals.append(vals[0])
    fig.add_trace(go.Scatterpolar(r=vals, theta=radar_m+[radar_m[0]], name=name, fill='toself', opacity=0.45))
fig.update_layout(title='<b>Top 7 Models — Radar Comparison</b>',
                  polar=dict(radialaxis=dict(range=[0,0.75], tickfont=dict(size=9))), height=650,
                  legend=dict(orientation='h', y=-0.1))
fig.write_html(f'{OUT_DIR}/model_radar.html')
print("    ✓")

# 4. Time vs Performance
print("  [4/8] Time vs Performance...")
fig,ax=plt.subplots(figsize=(12,8))
names=[m[0] for m in sorted_m]; f1s=[m[1]['WF1'] for m in sorted_m]
times=[max(m[1]['Time'],0.1) for m in sorted_m]
sc=ax.scatter(times,f1s,s=150,c=range(len(names)),cmap='viridis_r',edgecolors='black',linewidth=1,zorder=5)
for i,name in enumerate(names):
    ax.annotate(name.split('(')[0].strip(),(times[i],f1s[i]),xytext=(8,5),
                textcoords='offset points',fontsize=7.5,alpha=0.85)
ax.set_xlabel('Training Time (seconds, log scale)',fontsize=12); ax.set_ylabel('Weighted F1',fontsize=12)
ax.set_xscale('log'); ax.set_title('Model Performance vs Training Time',fontsize=14,fontweight='bold')
ax.grid(True,alpha=0.3); plt.colorbar(sc,ax=ax,label='Rank (1=best)')
pareto_t=[]; pareto_f=[]; best_f1=0
for m in sorted_m:
    t_=max(m[1]['Time'],0.1); f_=m[1]['WF1']
    if f_>best_f1: best_f1=f_; pareto_t.append(t_); pareto_f.append(f_)
if pareto_t: ax.plot(pareto_t,pareto_f,'r--',linewidth=2,alpha=0.6,label='Pareto frontier')
ax.legend(); plt.tight_layout()
plt.savefig(f'{OUT_DIR}/time_vs_performance.png',dpi=150,bbox_inches='tight'); plt.close()
print("    ✓")

# 5. CV comparison
print("  [5/8] CV comparison...")
fig,ax=plt.subplots(figsize=(14,6))
cn=list(cv_r.keys()); cm=[cv_r[n]['mean'] for n in cn]; cs=[cv_r[n]['std'] for n in cn]
colors_cv=plt.cm.viridis(np.linspace(0.15,0.95,len(cn)))
bars=ax.barh(range(len(cn)),cm,xerr=cs,capsize=8,color=colors_cv,edgecolor='white',linewidth=1.5)
ax.set_yticks(range(len(cn))); ax.set_yticklabels(cn,fontsize=10)
ax.set_xlabel('Weighted F1 Score',fontsize=12); ax.invert_yaxis()
ax.set_title('5-Fold Cross-Validation — Weighted F1 Scores',fontsize=14,fontweight='bold')
for bar,mean,std in zip(bars,cm,cs):
    ax.text(mean+std+0.005,bar.get_y()+bar.get_height()/2,f'{mean:.4f} ±{std:.4f}',
            va='center',fontsize=9,fontweight='bold')
ax.set_xlim(0,max(cm)+max(cs)+0.08); plt.tight_layout()
plt.savefig(f'{OUT_DIR}/cv_comparison.png',dpi=150,bbox_inches='tight'); plt.close()
print("    ✓")

# 6. Confusion matrix (best model, top 12)
print("  [6/8] Confusion matrix...")
top12=df['main_genre'].value_counts().head(12).index
tgl=le.inverse_transform(Yte); t12m=np.isin(tgl,top12)
if t12m.sum()>50:
    yts=Yte[t12m]; yps=best_pred[t12m]
    sle=LabelEncoder(); sbl=le.inverse_transform(yts); yte_enc=sle.fit_transform(sbl)
    ypl=le.inverse_transform(yps); vpm=np.isin(ypl,list(sle.classes_))
    yte_f=yte_enc[vpm]; ype_f=sle.transform(ypl[vpm])
    cm=confusion_matrix(yte_f,ype_f); cmn=cm.astype(float)/cm.sum(axis=1)[:,np.newaxis]
    fig,ax=plt.subplots(figsize=(15,13))
    sns.heatmap(cmn,annot=True,fmt='.2f',cmap='YlOrRd',ax=ax,
                xticklabels=sle.classes_,yticklabels=sle.classes_,
                linewidths=0.5,linecolor='white',cbar_kws={'label':'Proportion'},
                vmin=0,vmax=1,annot_kws={'fontsize':8})
    ax.set_title(f'Confusion Matrix — {best_name} (Top 12 Genres)',fontsize=16,fontweight='bold')
    ax.set_xlabel('Predicted Genre',fontsize=12); ax.set_ylabel('True Genre',fontsize=12)
    plt.xticks(rotation=45,ha='right',fontsize=9); plt.yticks(rotation=0,fontsize=9)
    plt.tight_layout(); plt.savefig(f'{OUT_DIR}/best_model_confusion.png',dpi=150,bbox_inches='tight'); plt.close()
    print("    ✓")

# 7. Feature importance
print("  [7/8] Feature importance...")
if 'XGBoost' in best_name:
    best_model_obj = base_models.get('XGBoost')
elif 'LightGBM' in best_name:
    best_model_obj = base_models.get('LightGBM')
elif 'CatBoost' in best_name:
    best_model_obj = base_models.get('CatBoost')
elif 'Random Forest' in best_name:
    best_model_obj = base_models.get('Random Forest')
else:
    best_model_obj = None
    for name in [m[0] for m in sorted_m]:
        if name in base_models and hasattr(base_models[name],'feature_importances_'):
            best_model_obj = base_models[name]; break

if best_model_obj is not None and hasattr(best_model_obj,'feature_importances_'):
    fi=best_model_obj.feature_importances_
    fi_idx=np.argsort(fi)[-30:]
    fig,ax=plt.subplots(figsize=(10,9))
    ax.barh(range(30),fi[fi_idx],color='steelblue')
    ax.set_yticks(range(30)); ax.set_yticklabels([all_fc[i] for i in fi_idx],fontsize=8)
    ax.set_title(f'Top 30 Feature Importances — {best_name}',fontsize=14,fontweight='bold')
    ax.set_xlabel('Importance',fontsize=12)
    for i,(j,v) in enumerate(zip(fi_idx,fi[fi_idx])):
        ax.text(v+0.001,i,f'{v:.4f}',va='center',fontsize=7)
    plt.tight_layout(); plt.savefig(f'{OUT_DIR}/feature_importance.png',dpi=150,bbox_inches='tight'); plt.close()
    print("    ✓")

# 8. Category comparison
print("  [8/8] Category comparison...")
categories={
    'Linear/Simple':['Logistic Regression','SVM (Linear)'],
    'Tree-based':['Decision Tree','Random Forest','Extra Trees'],
    'Gradient Boosting':['HistGradient Boosting','XGBoost','LightGBM','CatBoost',
                          'XGBoost (Tuned)','LightGBM (Tuned)','CatBoost (Tuned)'],
    'Neural Network':['MLP Neural Net'],
    'Ensemble':['Stacking Ensemble','Voting Ensemble'],
    'Distance-based':['KNN (k=15)'],
}
cat_results={}
for cat,model_names in categories.items():
    cat_vals={n:results[n] for n in model_names if n in results}
    if cat_vals:
        best_in_cat=max(cat_vals.items(),key=lambda x:x[1]['WF1'])
        cat_results[cat]={'best_model':best_in_cat[0],'WF1':best_in_cat[1]['WF1'],
                          'Acc':best_in_cat[1]['Acc'],'count':len(cat_vals)}

fig,ax=plt.subplots(figsize=(12,6))
cats=list(cat_results.keys()); wf1s=[cat_results[c]['WF1'] for c in cats]
accs=[cat_results[c]['Acc'] for c in cats]
x=np.arange(len(cats)); w=0.35
b1=ax.bar(x-w/2,wf1s,w,label='Weighted F1',color='#e74c3c',edgecolor='white')
b2=ax.bar(x+w/2,accs,w,label='Accuracy',color='#3498db',edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(cats,fontsize=10)
for bar,val in zip(b1,wf1s):
    ax.text(bar.get_x()+bar.get_width()/2,val+0.01,f'{val:.3f}',ha='center',fontsize=9,fontweight='bold')
for bar,val in zip(b2,accs):
    ax.text(bar.get_x()+bar.get_width()/2,val+0.01,f'{val:.3f}',ha='center',fontsize=9)
ax.set_ylabel('Score',fontsize=12); ax.legend(); ax.set_ylim(0,0.85)
ax.set_title('Best Model Performance by Category',fontsize=14,fontweight='bold')
plt.tight_layout(); plt.savefig(f'{OUT_DIR}/category_comparison.png',dpi=150,bbox_inches='tight'); plt.close()
print("    ✓")

# ====== FINAL REPORT ======
print("\n" + "=" * 70)
print("FINAL MODEL RANKINGS (by Weighted F1)")
print("=" * 70)
for rank,(name,met) in enumerate(sorted_m,1):
    star=" ⭐" if rank==1 else "  "
    bar_str="█"*int(met['WF1']*40)
    print(f"{star} {rank:2d}. {name:<28s} | WF1={met['WF1']:.4f} Acc={met['Acc']:.4f} "
          f"MF1={met['MF1']:.4f} Time={met['Time']:.0f}s {bar_str}")

print(f"\n{'='*70}")
print(f"BEST MODEL: {best_name}")
print(f"{'='*70}")
for k in ['Acc','MF1','WF1','MP','MR','Time']:
    print(f"  {k:<20s}: {results[best_name][k]}")

report=classification_report(Yte,best_pred,target_names=le.classes_,digits=3,output_dict=True)
rdf_r=pd.DataFrame(report).T
gr=rdf_r.drop(['accuracy','macro avg','weighted avg'],errors='ignore').sort_values('f1-score',ascending=False)
print("\n  BEST 10 GENRES:")
for idx,row in gr.head(10).iterrows():
    print(f"    {idx:<25s} F1={row['f1-score']:.3f} P={row['precision']:.3f} R={row['recall']:.3f} sup={int(row['support'])}")
print("\n  WORST 10 GENRES:")
for idx,row in gr.tail(10).iterrows():
    print(f"    {idx:<25s} F1={row['f1-score']:.3f} P={row['precision']:.3f} R={row['recall']:.3f} sup={int(row['support'])}")

# Save comprehensive report
with open(f'{OUT_DIR}/best_model_report.txt','w') as f:
    f.write(f"BEST MODEL: {best_name}\n{'='*60}\n")
    for k,v in results[best_name].items(): f.write(f"{k}: {v}\n")
    f.write(f"\nCV RESULTS:\n")
    for k,v in cv_r.items(): f.write(f"  {k}: {v['mean']:.4f} +/- {v['std']:.4f}\n")
    f.write(f"\nTUNED PARAMETERS:\n")
    for k,v in tuned_params.items(): f.write(f"  {k}: {v}\n")
    f.write(f"\nFULL CLASSIFICATION REPORT:\n")
    f.write(classification_report(Yte,best_pred,target_names=le.classes_,digits=3))

print(f"\n{'='*70}")
print(f"ALL DONE! Outputs in {OUT_DIR}/")
print(f"{'='*70}")
for f in sorted(os.listdir(OUT_DIR)):
    sz=os.path.getsize(os.path.join(OUT_DIR,f))
    print(f"  {f} ({sz:>10,} bytes)")