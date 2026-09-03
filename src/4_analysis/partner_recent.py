#!/usr/bin/env python3
"""Decisive test of the FINEST graph tie: does adding individual engagement-partner
contagion exposure improve recent-period (2017-2019) prediction beyond firm-level graph
signal? Feature-set ablation with XGBoost (the champion). -> data/partner_recent.json"""
import json, itertools, sys
import numpy as np, pandas as pd, torch
from collections import defaultdict
from torch_geometric.utils import scatter
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
np.random.seed(0)

G=torch.load("data/graph.pt", weights_only=False)
gvkeys=G["gvkeys"]; YEARS=G["years"]; N=len(gvkeys); gid={g:i for i,g in enumerate(gvkeys)}
yidx={y:i for i,y in enumerate(YEARS)}
X=G["X"].numpy(); active=G["active"].numpy(); snaps=G["snapshots"]
comp=pd.read_pickle("data/comp_company.pkl").dropna(subset=["cik"]); comp["cik"]=comp["cik"].astype("int64")
g2cik=dict(zip(comp["gvkey"],comp["cik"]))
uni=pd.DataFrame({"gvkey":gvkeys}); uni["cik"]=uni["gvkey"].map(g2cik); uni=uni.dropna(subset=["cik"]); uni["cik"]=uni["cik"].astype("int64")
dup=uni["cik"].duplicated(keep=False); cik2g={int(c):g for g,c in zip(uni.loc[~dup,"gvkey"],uni.loc[~dup,"cik"])}
fap=pd.read_pickle("data/formap.pkl").dropna(subset=["partner_id","cik","fpe_year"]); fap["cik"]=fap["cik"].astype("int64")

def agg(vec,ei,w):
    if ei.size(1)==0: return torch.zeros(N)
    src,dst=ei; num=scatter(vec[src]*w,dst,0,dim_size=N,reduce='sum')
    den=scatter(w,dst,0,dim_size=N,reduce='sum').clamp(min=1e-6); return num/den
def partner_ei(year):
    sub=fap[fap["fpe_year"]==year]; grp=defaultdict(set)
    for r in sub.itertuples():
        g=cik2g.get(int(r.cik));
        if g is not None: grp[r.partner_id].add(gid[g])
    E=set()
    for pid,m in grp.items():
        m=sorted(m)
        if 2<=len(m)<=50:
            for a,b in itertools.combinations(m,2): E.add((a,b)); E.add((b,a))
    if not E: return torch.zeros((2,0),dtype=torch.long)
    return torch.tensor([[a for a,b in E],[b for a,b in E]],dtype=torch.long)
PART={y:partner_ei(y) for y in [2016,2017,2018,2019]}  # 2016 empty (pre-FormAP) -> zeros

LAB={"any":("label","restated_now"),"severe":("label_severe","restated_now_severe")}
PANEL=["auditor","office","board","ownership"]

def exposures(t, rnv):
    rn=torch.from_numpy(rnv[t])
    feats={}
    for c in PANEL:
        ei,w=snaps[t][c]; feats[c]=agg(rn,ei,w).numpy()
    ei=PART[YEARS[t]]; w=torch.ones(ei.size(1))
    feats["partner"]=agg(rn,ei,w).numpy()
    return feats

def build(t, rnv, label, cols):
    a=np.where(active[t]&(label[t]>=0))[0]
    base=[X[t][a], rnv[t][a,None], rnv[max(t-1,0)][a,None]]
    ex=exposures(t,rnv)
    blocks=list(base)
    if "panel" in cols: blocks += [np.stack([ex[c][a] for c in PANEL],1)]
    if "partner" in cols: blocks += [ex["partner"][a,None]]
    return np.concatenate(blocks,1), label[t][a]

def xgbfit(Xtr,ytr,Xte,yte):
    spw=float((ytr==0).sum()/max((ytr==1).sum(),1))
    m=xgb.XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,
        min_child_weight=5,reg_lambda=2.0,scale_pos_weight=spw,eval_metric="aucpr",n_jobs=4,tree_method="hist")
    m.fit(Xtr,ytr); p=m.predict_proba(Xte)[:,1]
    return float(roc_auc_score(yte,p)), float(average_precision_score(yte,p))

OUT={}
SPLITS=[(2017,2018),(2018,2019)]   # (train fiscal yr, test fiscal yr); each predicts +1
for lab,(lk,rk) in LAB.items():
    label=G[lk].numpy(); rnv=G[rk].numpy().astype(np.float32)
    OUT[lab]={}
    for cols,name in [([],"base(feat+own)"),(["panel"],"+panel graph"),(["panel","partner"],"+panel+PARTNER")]:
        rocs=[];prs=[]
        for (tr,te) in SPLITS:
            Xtr,ytr=build(yidx[tr],rnv,label,cols); Xte,yte=build(yidx[te],rnv,label,cols)
            r,p=xgbfit(Xtr,ytr,Xte,yte); rocs.append(r); prs.append(p)
        OUT[lab][name]={"roc":round(float(np.mean(rocs)),4),"pr":round(float(np.mean(prs)),4),
                        "roc_by_split":[round(x,3) for x in rocs]}
        print(f"[{lab}] {name:20s} ROC={np.mean(rocs):.3f} PR={np.mean(prs):.3f}",file=sys.stderr)
for y in [2017,2018,2019]: print(f"partner edges fpe{y}: {PART[y].size(1)//2}",file=sys.stderr)
json.dump(OUT,open("data/partner_recent.json","w"),indent=2,ensure_ascii=False)
print(json.dumps(OUT,ensure_ascii=False,indent=2))
