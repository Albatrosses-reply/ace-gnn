#!/usr/bin/env python3
"""Fair tabular baselines: every tabular learner receives EXACTLY the node features the graph models
receive, i.e. [Xz (28 standardised ratios), OWN (own problem indicator at t and t-1)].
This is the "fair-comparison protocol" of the paper (Sec. 5.1); benchmark.py's tabular block uses Xz only.

Writes  data/ext/pure/tabular_fair_{MODE}_{LABEL}.json        (metrics per model, same schema as benchmark.py)
        data/ext/pure/tabular_fair_{MODE}_{LABEL}_preds.npz   (y + test-set predictions per model, for paired_bootstrap.py)
ENV: MODE=recent|panel  LABEL=severe|label|adverse
Run from repo root:  MODE=recent LABEL=severe python3 src/4_analysis/tabular_fair.py
"""
import json, sys, os, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb, lightgbm as lgb, catboost as cb

torch.manual_seed(0); np.random.seed(0)
MODE=os.environ.get("MODE","recent"); LABEL_MODE=os.environ.get("LABEL","severe")
os.makedirs("data/ext/pure",exist_ok=True)
OUT=os.environ.get("OUT",f"data/ext/pure/tabular_fair_{MODE}_{LABEL_MODE}.json")
LKEY={"adverse":"label_adverse","severe":"label_severe"}.get(LABEL_MODE,"label")
RKEY={"adverse":"restated_now_adverse","severe":"restated_now_severe"}.get(LABEL_MODE,"restated_now")
G=torch.load(os.environ.get("GRAPH","data/ext/graph.pt"), weights_only=False)
YEARS=G["years"]; yidx={y:i for i,y in enumerate(YEARS)}
X=G["X"].numpy().copy(); active=G["active"].numpy(); label=G[LKEY].numpy(); rn=G[RKEY].numpy().astype(np.float32)
Tall,N,Fdim=X.shape
if MODE=="panel": USE=[yidx[y] for y in range(2005,2020)]; TR=USE[:11]; VA=[USE[11]]; TE=USE[12:15]
else: USE=[yidx[y] for y in range(2017,2023)]; TR=USE[:3]; VA=[USE[3]]; TE=USE[4:6]
print(f"[tabular_fair] MODE={MODE} label={LABEL_MODE} test={[YEARS[i] for i in TE]}",file=sys.stderr)

# identical preprocessing to benchmark.py / ace_v2.py: winsorise at train 1/99 pct, median-impute, z-score on train
tm=np.zeros((Tall,N),bool)
for t in TR: tm[t]=active[t]&(label[t]>=0)
Xtr0=X[tm]; lo=np.nanpercentile(Xtr0,1,0); hi=np.nanpercentile(Xtr0,99,0); med=np.nanmedian(Xtr0,0)
Xc=np.clip(X,lo,hi); ix=np.where(np.isnan(Xc)); Xc[ix]=np.take(med,ix[2])
mu=np.nanmean(np.clip(Xtr0,lo,hi),0); sd=np.nanstd(np.clip(Xtr0,lo,hi),0); sd[sd<1e-6]=1
Xz=((Xc-mu)/sd).astype(np.float32)
OWN=np.zeros((Tall,N,2),np.float32)
for t in range(Tall): OWN[t,:,0]=rn[t]; OWN[t,:,1]=rn[max(t-1,0)]
NF=np.concatenate([Xz,OWN],2).astype(np.float32)          # == the GNN node features in benchmark.py

def pool(ts,arr):
    xs=[];ys=[]
    for t in ts:
        a=np.where(active[t]&(label[t]>=0))[0]; xs.append(arr[t][a]); ys.append(label[t][a])
    return np.concatenate(xs),np.concatenate(ys)
Xtr,ytr=pool(TR,NF); Xte,yte=pool(TE,NF)
spw=float((ytr==0).sum()/max((ytr==1).sum(),1))
def metr(y,p):
    o=np.argsort(-p); return {"roc":round(float(roc_auc_score(y,p)),4),"pr":round(float(average_precision_score(y,p)),4),
      "recall@10%":round(float(y[o[:max(1,len(p)//10)]].sum()/max(y.sum(),1)),4)}
PREDS={}; RES={"mode":MODE,"label":LABEL_MODE,"features":"[Xz(28 ratios), OWN(r_t, r_{t-1})] — identical to GNN node features",
               "n_test":int(len(yte)),"pos_test":int(yte.sum()),"leaderboard":{}}
def add(name,p): PREDS[name]=p; RES["leaderboard"][name]=metr(yte,p); print(f"[{name}] {RES['leaderboard'][name]}",file=sys.stderr)
add("LogisticRegression", LogisticRegression(max_iter=2000,class_weight="balanced").fit(Xtr,ytr).predict_proba(Xte)[:,1])
add("RandomForest", RandomForestClassifier(n_estimators=400,max_depth=None,min_samples_leaf=5,class_weight="balanced_subsample",n_jobs=4,random_state=0).fit(Xtr,ytr).predict_proba(Xte)[:,1])
add("ExtraTrees", ExtraTreesClassifier(n_estimators=400,min_samples_leaf=5,class_weight="balanced_subsample",n_jobs=4,random_state=0).fit(Xtr,ytr).predict_proba(Xte)[:,1])
add("HistGradBoosting", HistGradientBoostingClassifier(max_depth=4,learning_rate=0.05,l2_regularization=2.0,class_weight="balanced",random_state=0).fit(Xtr,ytr).predict_proba(Xte)[:,1])
add("LightGBM", lgb.LGBMClassifier(n_estimators=400,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,reg_lambda=2.0,min_child_samples=20,scale_pos_weight=spw,n_jobs=4,verbose=-1,random_state=0).fit(Xtr,ytr).predict_proba(Xte)[:,1])
add("CatBoost", cb.CatBoostClassifier(iterations=400,depth=4,learning_rate=0.05,l2_leaf_reg=3.0,scale_pos_weight=spw,verbose=0,random_seed=0).fit(Xtr,ytr).predict_proba(Xte)[:,1])
add("XGBoost", xgb.XGBClassifier(n_estimators=400,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,min_child_weight=5,reg_lambda=2.0,scale_pos_weight=spw,eval_metric="aucpr",n_jobs=4,random_state=0).fit(Xtr,ytr).predict_proba(Xte)[:,1])
add("MLP", MLPClassifier(hidden_layer_sizes=(128,64),alpha=1e-3,max_iter=300,random_state=0).fit(Xtr,ytr).predict_proba(Xte)[:,1])
json.dump(RES,open(OUT,"w"),indent=2,ensure_ascii=False)
np.savez_compressed(OUT.replace(".json","_preds.npz"), y=yte, **{k.replace(" ","_"):v for k,v in PREDS.items()})
print(f"[done -> {OUT}]",file=sys.stderr)
