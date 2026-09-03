#!/usr/bin/env python3
"""Honest marginal-value decomposition on the 10-yr panel: isolate the graph's contribution
OVER firm features and own-persistence. XGBoost, feat / +own / +graph-exposure / +neighbor-feat."""
import json, sys
import numpy as np, torch
from torch_geometric.utils import scatter
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
G=torch.load("data/graph.pt", weights_only=False)
YEARS=G["years"]; X=G["X"].numpy(); active=G["active"].numpy(); snaps=G["snapshots"]
T,N,Fdim=X.shape; REL=["auditor","office","board","ownership"]; nrel=len(REL)
TRAIN_T=[0,1,2,3,4,5]; TEST_T=[7,8]
def agg(vec,ei,w):
    if ei.size(1)==0: return torch.zeros_like(vec)
    src,dst=ei; v=vec if vec.dim()>1 else vec.unsqueeze(1)
    num=scatter(v[src]*w.unsqueeze(1),dst,0,dim_size=N,reduce='sum')
    den=scatter(w,dst,0,dim_size=N,reduce='sum').clamp(min=1e-6); o=num/den.unsqueeze(1)
    return o if vec.dim()>1 else o.squeeze(1)
OUT={}
for lab,(lk,rk) in {"any":("label","restated_now"),"severe":("label_severe","restated_now_severe")}.items():
    label=G[lk].numpy(); rn=G[rk].numpy().astype(np.float32)
    EXP=np.zeros((T,N,nrel),np.float32); NFA=np.zeros((T,N,nrel*Fdim),np.float32); OWN=np.zeros((T,N,2),np.float32)
    for t in range(T):
        rnv=torch.from_numpy(rn[t]); xz=torch.from_numpy(np.nan_to_num(X[t]))
        for ci,c in enumerate(REL):
            ei,w=snaps[t][c]; EXP[t,:,ci]=agg(rnv,ei,w).numpy(); NFA[t,:,ci*Fdim:(ci+1)*Fdim]=agg(xz,ei,w).numpy()
        OWN[t,:,0]=rn[t]; OWN[t,:,1]=rn[max(t-1,0)]
    def pool(ts,blocks):
        xs=[];ys=[]
        for t in ts:
            a=np.where(active[t]&(label[t]>=0))[0]; xs.append(np.concatenate([b[t][a] for b in blocks],1)); ys.append(label[t][a])
        return np.concatenate(xs),np.concatenate(ys)
    sets={"feat":[X],"feat+own":[X,OWN],"feat+own+graphExp":[X,OWN,EXP],"feat+own+graphExp+nbrFeat":[X,OWN,EXP,NFA]}
    OUT[lab]={}
    for name,bl in sets.items():
        Xtr,ytr=pool(TRAIN_T,bl); Xte,yte=pool(TEST_T,bl)
        spw=float((ytr==0).sum()/max((ytr==1).sum(),1))
        m=xgb.XGBClassifier(n_estimators=400,max_depth=4,learning_rate=0.04,subsample=0.8,colsample_bytree=0.8,
            min_child_weight=5,reg_lambda=2.0,scale_pos_weight=spw,eval_metric="aucpr",n_jobs=4,tree_method="hist")
        m.fit(Xtr,ytr); p=m.predict_proba(Xte)[:,1]
        OUT[lab][name]={"roc":round(float(roc_auc_score(yte,p)),4),"pr":round(float(average_precision_score(yte,p)),4)}
        print(f"[{lab}] {name:26s} ROC={OUT[lab][name]['roc']} PR={OUT[lab][name]['pr']}",file=sys.stderr)
json.dump(OUT,open("data/panel_decomp.json","w"),indent=2)
print(json.dumps(OUT,indent=2))
