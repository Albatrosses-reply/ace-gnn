#!/usr/bin/env python3
"""Add material/adverse-restatement labels to graph.pt (no edge rebuild)."""
import numpy as np, pandas as pd, torch
from collections import defaultdict
G=torch.load("data/graph.pt", weights_only=False)
gvkeys=G["gvkeys"]; YEARS=G["years"]; N=len(gvkeys); gid={g:i for i,g in enumerate(gvkeys)}
active=G["active"].numpy()
comp=pd.read_pickle("data/comp_company.pkl").dropna(subset=["cik"]); comp["cik"]=comp["cik"].astype("int64")
g2cik=dict(zip(comp["gvkey"],comp["cik"]))
uni=pd.DataFrame({"gvkey":gvkeys}); uni["cik"]=uni["gvkey"].map(g2cik); uni=uni.dropna(subset=["cik"]); uni["cik"]=uni["cik"].astype("int64")
dup=uni["cik"].duplicated(keep=False)
cik2g={int(c):g for g,c in zip(uni.loc[~dup,"gvkey"],uni.loc[~dup,"cik"])}
res=pd.read_pickle("data/restate.pkl").dropna(subset=["cik"]); res["cik"]=res["cik"].astype("int64")
adv=res[res["res_adverse"]==1.0]
ann_adv=defaultdict(set)
for r in adv.itertuples():
    g=cik2g.get(int(r.cik))
    if g is not None: ann_adv[int(r.ann_year)].add(g)
label_adv=np.full((len(YEARS),N),-1,dtype=np.int64)
restated_now_adv=np.zeros((len(YEARS),N),dtype=np.int64)
for yi,y in enumerate(YEARS):
    posN=ann_adv.get(y+1,set()); pos0=ann_adv.get(y,set())
    for i in range(N):
        if active[yi,i]: label_adv[yi,i]=1 if gvkeys[i] in posN else 0
        if gvkeys[i] in pos0: restated_now_adv[yi,i]=1
for yi,y in enumerate(YEARS):
    m=label_adv[yi]>=0
    if m.sum(): print(f"  adverse y{y}->{y+1}: n={m.sum()} pos={int((label_adv[yi][m]==1).sum())} ({(label_adv[yi][m]==1).mean()*100:.1f}%)")
G["label_adverse"]=torch.from_numpy(label_adv); G["restated_now_adverse"]=torch.from_numpy(restated_now_adv)
torch.save(G,"data/graph.pt"); print("saved adverse labels to graph.pt")
