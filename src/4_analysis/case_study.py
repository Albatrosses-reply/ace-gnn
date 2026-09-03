#!/usr/bin/env python3
"""Identify 4 illustrative case studies from recent/severe test predictions + neighbor exposures.
Cases: (A) auditor-contagion catch where graph beats features, (B) shared engagement-partner channel,
(C) audit-office co-restatement cluster, (D) selective non-contagion (board/ownership exposed but low-risk).
-> data/ext/case_studies.json  (anonymized labels + real gvkey/cik for de-anonymization)
Run from repo root: python3 src/4_analysis/case_study.py"""
import json, numpy as np, pandas as pd, torch
from collections import defaultdict
G=torch.load("data/ext/graph.pt", weights_only=False)
gv=G["gvkeys"]; YEARS=G["years"]; yidx={y:i for i,y in enumerate(YEARS)}; REL=G["relations"]
active=G["active"].numpy(); label=G["label_severe"].numpy(); rn=G["restated_now_severe"].numpy()
snaps=G["snapshots"]; N=len(gv)
TE=[yidx[2021],yidx[2022]]
d=np.load("data/ext/ace3_recent_severe_preds.npz"); pace=d["ens"]; pxgb=d["xgb"]; yy=d["y"]
# reconstruct (node,year) order matching npz
rows=[]
for t in TE:
    for i in np.where(active[t]&(label[t]>=0))[0]: rows.append((int(i),t))
assert len(rows)==len(yy) and np.array_equal(np.array([label[t][i] for i,t in rows]),yy), "order mismatch"
prob={(i,t):(float(pace[k]),float(pxgb[k])) for k,(i,t) in enumerate(rows)}
# names / ids
comp=pd.read_pickle("data/legacy/comp_company.pkl"); g2cik=dict(zip(comp["gvkey"],comp["cik"])); g2sic=dict(zip(comp["gvkey"],comp["sic"]))
def neigh(i,t,rel):
    ei,_=snaps[t][rel]
    if ei.size(1)==0: return []
    e=ei.numpy(); return list(np.unique(e[1][e[0]==i]))
def exposed(i,t,rel):  # neighbors with a severe event in year t
    return [j for j in neigh(i,t,rel) if rn[t][j]==1]
def info(i):
    sic=g2sic.get(gv[i]);
    return {"gvkey":gv[i],"cik":int(g2cik.get(gv[i],-1)) if pd.notna(g2cik.get(gv[i])) else -1,"sic":(int(float(sic)) if pd.notna(sic) else None)}
def expo_summary(i,t):
    return {r:len(exposed(i,t,r)) for r in REL}

CASES={}
# ---- Case A: confident correct flag whose mechanism is the shared-auditor channel ----
cand=[]
for (i,t) in rows:
    e=expo_summary(i,t)
    if label[t][i]==1 and e["auditor"]>=2:
        pa,px=prob[(i,t)]; cand.append((pa,i,t,pa,px,e))
cand.sort(reverse=True)
if cand:
    i,t=cand[0][1],cand[0][2]; pa,px=cand[0][3],cand[0][4]
    CASES["A_auditor_contagion_catch"]={"focal":info(i),"year":int(YEARS[t]),"label":1,"ace_prob":round(pa,3),"xgb_prob":round(px,3),
        "exposure":expo_summary(i,t),"restating_auditor_neighbors":[info(j) for j in exposed(i,t,"auditor")][:6],
        "note":"Confident correct high-risk flag; the multiplex graph reveals the mechanism — several shared-auditor co-clients had severe events the same year (auditor-channel contagion)."}
# ---- Case B: shared engagement partner (2017+) ----
candB=[]
for (i,t) in rows:
    if label[t][i]==1 and len(exposed(i,t,"partner"))>=1:
        pa,px=prob[(i,t)]; candB.append((pa,i,t,pa,px))
candB.sort(reverse=True)
if candB:
    i,t=candB[0][1],candB[0][2]; pa,px=candB[0][3],candB[0][4]
    CASES["B_shared_partner"]={"focal":info(i),"year":int(YEARS[t]),"label":1,"ace_prob":round(pa,3),"xgb_prob":round(px,3),
        "exposure":expo_summary(i,t),"restating_partner_neighbors":[info(j) for j in exposed(i,t,"partner")][:6],
        "note":"Same individual PCAOB engagement partner as a firm that had a severe event (finest tie, lift 2.54)."}
# ---- Case C: audit-office cluster (auditor x city) with >=2 co-restating clients in a test year ----
sox=pd.read_pickle("data/ext/sox404.pkl"); g2i={g:k for k,g in enumerate(gv)}
comp2=comp.dropna(subset=["cik"]).copy(); comp2["cik"]=comp2["cik"].astype("int64"); cik2g={int(c):g for g,c in zip(comp2["gvkey"],comp2["cik"]) }
bestC=None
for t in TE:
    fy=YEARS[t]; sub=sox[(sox["fy"]==fy)&sox["aud_city"].notna()&sox["auditor_fkey"].notna()&sox["cik"].notna()]
    grp=defaultdict(list)
    for r in sub.itertuples():
        g=cik2g.get(int(r.cik));
        if g is not None and g in g2i: grp[(r.auditor_fkey,str(r.aud_city).strip().upper())].append(g2i[g])
    for key,members in grp.items():
        m=sorted(set(members)); restating=[j for j in m if rn[t][j]==1]
        if len(m)>=4 and len(restating)>=2:
            score=len(restating)
            if bestC is None or score>bestC[0]: bestC=(score,key,m,restating,t)
if bestC:
    _,key,m,restating,t=bestC
    CASES["C_office_cluster"]={"office":{"auditor_fkey":float(key[0]),"city":key[1]},"year":int(YEARS[t]),
        "n_clients_in_office":len(m),"n_restating":len(restating),
        "restating_clients":[info(j) for j in restating][:8],"sample_other_clients":[info(j) for j in m if j not in restating][:6],
        "note":"Multiple clients of the same audit office (auditor x city) co-experience severe events."}
# ---- Case D: selective non-contagion (board/ownership exposed, auditor NOT, label=0, low ACE risk) ----
candD=[]
for (i,t) in rows:
    e=expo_summary(i,t)
    if label[t][i]==0 and (e["board"]+e["ownership"])>=2 and e["auditor"]==0 and e["office"]==0:
        pa,px=prob[(i,t)]; candD.append((pa,i,t,pa,px,e))
candD.sort()  # lowest ACE prob = most confidently (correctly) not flagged
if candD:
    i,t=candD[0][1],candD[0][2]; pa,px=candD[0][3],candD[0][4]; e=candD[0][5]
    CASES["D_selective_noncontagion"]={"focal":info(i),"year":int(YEARS[t]),"label":0,"ace_prob":round(pa,3),"xgb_prob":round(px,3),
        "exposure":e,"restating_board_neighbors":[info(j) for j in exposed(i,t,"board")][:5],
        "restating_ownership_neighbors":[info(j) for j in exposed(i,t,"ownership")][:5],
        "note":"Connected via board/common-ownership to firms that had events, but NO auditor tie -> correctly low risk (board/ownership are not contagion channels)."}
json.dump(CASES,open("data/ext/case_studies.json","w"),indent=2,ensure_ascii=False)
for k,v in CASES.items():
    print(f"\n### {k}")
    print(json.dumps(v,ensure_ascii=False)[:600])
