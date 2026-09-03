#!/usr/bin/env python3
"""Add (1) severe composite label (ICFR material weakness + fraud restatement + AAER)
and (2) audit-office relation (auditor x city) to graph.pt."""
import sys, itertools
import numpy as np, pandas as pd, torch
from collections import defaultdict
AUD_GROUP_CAP=40
G=torch.load("data/graph.pt", weights_only=False)
gvkeys=G["gvkeys"]; YEARS=G["years"]; N=len(gvkeys); gid={g:i for i,g in enumerate(gvkeys)}
active=G["active"].numpy()

comp=pd.read_pickle("data/comp_company.pkl").dropna(subset=["cik"]); comp["cik"]=comp["cik"].astype("int64")
g2cik=dict(zip(comp["gvkey"],comp["cik"])); g2sic=dict(zip(comp["gvkey"],comp["sic"]))
uni=pd.DataFrame({"gvkey":gvkeys}); uni["cik"]=uni["gvkey"].map(g2cik); uni=uni.dropna(subset=["cik"]); uni["cik"]=uni["cik"].astype("int64")
dup=uni["cik"].duplicated(keep=False)
cik2g={int(c):g for g,c in zip(uni.loc[~dup,"gvkey"],uni.loc[~dup,"cik"])}

sox=pd.read_pickle("data/sox404.pkl")
aaer=pd.read_pickle("data/aaer.pkl")
res=pd.read_pickle("data/restate.pkl").dropna(subset=["cik"]); res["cik"]=res["cik"].astype("int64")

# ---- severe events per announce-year (gvkey sets) ----
sev=defaultdict(set)
# ICFR material weakness (ineffective) -> file_year
s_mw=sox[(sox["ic_is_effective"]=="N") & sox["cik"].notna() & sox["file_year"].notna()]
for r in s_mw.itertuples():
    g=cik2g.get(int(r.cik))
    if g is not None: sev[int(r.file_year)].add(g)
# fraud restatement
s_fr=res[res["res_fraud"]==1.0]
for r in s_fr.itertuples():
    g=cik2g.get(int(r.cik))
    if g is not None: sev[int(r.ann_year)].add(g)
# AAER
for r in aaer.dropna(subset=["cik","ann_year"]).itertuples():
    g=cik2g.get(int(r.cik))
    if g is not None: sev[int(r.ann_year)].add(g)
print("[severe events/yr]",{y:len(sev[y]) for y in range(2011,2021)},file=sys.stderr)

label_sev=np.full((len(YEARS),N),-1,dtype=np.int64)
rn_sev=np.zeros((len(YEARS),N),dtype=np.int64)
for yi,y in enumerate(YEARS):
    posN=sev.get(y+1,set()); pos0=sev.get(y,set())
    for i in range(N):
        if active[yi,i]: label_sev[yi,i]=1 if gvkeys[i] in posN else 0
        if gvkeys[i] in pos0: rn_sev[yi,i]=1
for yi,y in enumerate(YEARS):
    m=label_sev[yi]>=0
    if m.sum(): print(f"  severe y{y}->{y+1}: n={m.sum()} pos={int((label_sev[yi][m]==1).sum())} ({(label_sev[yi][m]==1).mean()*100:.1f}%)",file=sys.stderr)
G["label_severe"]=torch.from_numpy(label_sev); G["restated_now_severe"]=torch.from_numpy(rn_sev)

# ---- audit-office relation: auditor_fkey x aud_city, per fiscal year (fy_ic_op) ----
sox_off=sox[sox["aud_city"].notna() & sox["auditor_fkey"].notna() & sox["fy"].notna() & sox["cik"].notna()].copy()
def to_ei(edic):
    if not edic: return torch.zeros((2,0),dtype=torch.long), torch.zeros((0,),dtype=torch.float)
    es=list(edic.items())
    ei=torch.tensor([[a for (a,b),_ in es]+[b for (a,b),_ in es],
                     [b for (a,b),_ in es]+[a for (a,b),_ in es]],dtype=torch.long)
    w=torch.tensor([w for _,w in es]+[w for _,w in es],dtype=torch.float)
    return ei,w
snaps=G["snapshots"]
for yi,y in enumerate(YEARS):
    sub=sox_off[sox_off["fy"]==y]
    grp=defaultdict(set)
    for r in sub.itertuples():
        g=cik2g.get(int(r.cik))
        if g is None: continue
        grp[(r.auditor_fkey, str(r.aud_city).strip().upper(), str(r.aud_state))].add(gid[g])
    edic=defaultdict(float)
    for key,members in grp.items():
        m=sorted(members)
        if len(m)<2: continue
        if len(m)<=AUD_GROUP_CAP+1:
            for a,b in itertools.combinations(m,2): edic[(a,b)]+=1.0
        else:
            L=len(m)
            for ii,a in enumerate(m):
                for off in range(1,AUD_GROUP_CAP+1):
                    b=m[(ii+off)%L]
                    if a!=b: edic[((a,b) if a<b else (b,a))]+=1.0
    ei,w=to_ei(edic); snaps[yi]["office"]=(ei,w)
    print(f"[office y{y}] edges={len(edic)}",file=sys.stderr)
G["snapshots"]=snaps
torch.save(G,"data/graph.pt"); print("[saved severe label + office relation]",file=sys.stderr)
