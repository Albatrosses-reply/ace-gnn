#!/usr/bin/env python3
"""Placebo / leakage check: permute TRAIN labels, refit XGBoost+graph, eval on REAL test labels.
If the graph/exposure pipeline has no look-ahead leakage, test AUC should collapse to ~0.5."""
import os, warnings, json; warnings.filterwarnings("ignore")
import numpy as np, torch
from sklearn.metrics import roc_auc_score
import xgboost as xgb
MODE=os.environ.get("MODE","recent"); LABEL=os.environ.get("LABEL","severe")
LKEY={"adverse":"label_adverse","severe":"label_severe"}.get(LABEL,"label")
RKEY={"adverse":"restated_now_adverse","severe":"restated_now_severe"}.get(LABEL,"restated_now")
G=torch.load("data/ext/graph.pt",weights_only=False); YEARS=G["years"]; yidx={y:i for i,y in enumerate(YEARS)}
X=G["X"].numpy().copy(); active=G["active"].numpy(); label=G[LKEY].numpy(); rn=G[RKEY].numpy().astype(np.float32)
Tall,N,Fdim=X.shape; snaps=G["snapshots"]
REL=["partner","office","auditor","board","ownership"] if MODE=="recent" else ["office","auditor","board","ownership"]
if MODE=="panel": USE=[yidx[y] for y in range(2005,2020)];TR=USE[:11];TE=USE[12:15]
else: USE=[yidx[y] for y in range(2017,2023)];TR=USE[:3];TE=USE[4:6]
from torch_geometric.utils import scatter
def adj(t,c): ei,w=snaps[t][c];return ei,w
def magg(v,ei,w):
    if ei.size(1)==0: return torch.zeros_like(v)
    s,d=ei; vv=v.unsqueeze(1); num=scatter(vv[s]*w.unsqueeze(1),d,0,dim_size=N,reduce='sum'); den=scatter(w,d,0,dim_size=N,reduce='sum').clamp(min=1e-6); return (num/den.unsqueeze(1)).squeeze(1)
EXP=np.zeros((Tall,N,len(REL)*2),np.float32)
for t in range(Tall):
    for ci,c in enumerate(REL):
        ei,w=adj(t,c); EXP[t,:,ci]=magg(torch.from_numpy(rn[t]),ei,w).numpy()
        tl=max(t-1,0); ei2,w2=adj(tl,c); EXP[t,:,len(REL)+ci]=magg(torch.from_numpy(rn[tl]),ei2,w2).numpy()
OWN=np.zeros((Tall,N,2),np.float32)
for t in range(Tall): OWN[t,:,0]=rn[t];OWN[t,:,1]=rn[max(t-1,0)]
lo=np.nanpercentile(X[TR[0]],1,0)  # cheap impute
Xc=np.nan_to_num(X,nan=0.0); GF=np.concatenate([Xc,OWN,EXP],2)
def pool(ts):
    xs=[];ys=[]
    for t in ts:
        a=np.where(active[t]&(label[t]>=0))[0]; xs.append(GF[t][a]); ys.append(label[t][a])
    return np.concatenate(xs),np.concatenate(ys)
Xtr,ytr=pool(TR); Xte,yte=pool(TE); spw=float((ytr==0).sum()/max((ytr==1).sum(),1))
def fit(yt): return xgb.XGBClassifier(n_estimators=150,max_depth=4,learning_rate=0.05,scale_pos_weight=spw,n_jobs=4,verbosity=0,tree_method="hist").fit(Xtr,yt).predict_proba(Xte)[:,1]
real=roc_auc_score(yte,fit(ytr))
rng=np.random.default_rng(0); shuf=[roc_auc_score(yte,fit(rng.permutation(ytr))) for _ in range(5)]
out={"mode":MODE,"label":LABEL,"real_test_auc":round(float(real),4),"placebo_shuffled_label_auc_mean":round(float(np.mean(shuf)),4),"placebo_aucs":[round(float(x),4) for x in shuf]}
json.dump(out,open(f"data/ext/placebo_{MODE}_{LABEL}.json","w"),indent=2)
print(json.dumps(out,ensure_ascii=False))
