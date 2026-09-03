#!/usr/bin/env python3
"""Form AP engagement-partner contagion (2017-2019). Compares the finest channel
(shared individual partner) vs office vs auditor on the SAME firm-years. -> data/partner_contagion.json"""
import json, itertools
import numpy as np, pandas as pd, torch
from collections import defaultdict
from torch_geometric.utils import scatter

G=torch.load("data/graph.pt", weights_only=False)
gvkeys=G["gvkeys"]; YEARS=G["years"]; N=len(gvkeys); gid={g:i for i,g in enumerate(gvkeys)}
active=G["active"].numpy(); snaps=G["snapshots"]
yidx={y:i for i,y in enumerate(YEARS)}

comp=pd.read_pickle("data/comp_company.pkl").dropna(subset=["cik"]); comp["cik"]=comp["cik"].astype("int64")
g2cik=dict(zip(comp["gvkey"],comp["cik"]))
uni=pd.DataFrame({"gvkey":gvkeys}); uni["cik"]=uni["gvkey"].map(g2cik); uni=uni.dropna(subset=["cik"]); uni["cik"]=uni["cik"].astype("int64")
dup=uni["cik"].duplicated(keep=False); cik2g={int(c):g for g,c in zip(uni.loc[~dup,"gvkey"],uni.loc[~dup,"cik"])}

fap=pd.read_pickle("data/formap.pkl").dropna(subset=["partner_id","cik","fpe_year"])
fap["cik"]=fap["cik"].astype("int64")

# build partner-sharing edge_index per year (fiscal_period_end year), among node universe
def partner_ei(year):
    sub=fap[fap["fpe_year"]==year]
    grp=defaultdict(set)
    for r in sub.itertuples():
        g=cik2g.get(int(r.cik))
        if g is not None: grp[r.partner_id].add(gid[g])
    edges=set()
    for pid,members in grp.items():
        m=sorted(members)
        if len(m)<2 or len(m)>50: continue
        for a,b in itertools.combinations(m,2): edges.add((a,b))
    if not edges: return torch.zeros((2,0),dtype=torch.long)
    aa=[a for a,b in edges]+[b for a,b in edges]; bb=[b for a,b in edges]+[a for a,b in edges]
    return torch.tensor([aa,bb],dtype=torch.long)

partner_snap={y:partner_ei(y) for y in [2017,2018,2019]}
for y in [2017,2018,2019]:
    print(f"[partner edges {y}] {partner_snap[y].size(1)//2} undirected")

def neigh_restated(ei, rn_vec):
    if ei.size(1)==0: return np.zeros(N)
    src,dst=ei; rn=torch.from_numpy(rn_vec.astype(np.float32))
    return scatter(rn[src],dst,dim=0,dim_size=N,reduce='sum').numpy()

LAB={"any":("label","restated_now"),"material":("label_adverse","restated_now_adverse"),
     "severe":("label_severe","restated_now_severe")}
TEST_FEATURE_YEARS=[2017,2018]   # predict 2018, 2019

def rel_ei(rel,y):
    ti=yidx[y]
    if rel=="partner": return partner_snap[y]
    return snaps[ti][rel][0]

out={}
for lab,(lk,rk) in LAB.items():
    label=G[lk].numpy(); rn=G[rk].numpy()
    out[lab]={}
    for rel in ["partner","office","auditor","board","ownership"]:
        ne=nn_=0; pe=pn=0.0; exp_pos=nexp_pos=0
        for y in TEST_FEATURE_YEARS:
            ti=yidx[y]; a=active[ti]&(label[ti]>=0)
            nb=neigh_restated(rel_ei(rel,y), rn[ti])
            exp=(nb>0)&a; nex=(nb==0)&a
            ne+=int(exp.sum()); nn_+=int(nex.sum())
            exp_pos+=int(label[ti][exp].sum()); nexp_pos+=int(label[ti][nex].sum())
        pe=exp_pos/ne if ne else float('nan'); pn=nexp_pos/nn_ if nn_ else float('nan')
        out[lab][rel]={"n_exposed":ne,"P_exposed":round(pe,4),"n_not":nn_,"P_not":round(pn,4),
                       "lift":round(pe/pn,2) if pn else None}
print(json.dumps(out,ensure_ascii=False,indent=2))
json.dump(out,open("data/partner_contagion.json","w"),indent=2,ensure_ascii=False)
