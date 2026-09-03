#!/usr/bin/env python3
"""Build extended multiplex graph 2005-2022 with NESTED auditor-tie hierarchy
(partner-share L1, office-share L2, auditorXSIC2-share L3) + board + ownership.
Labels: any / adverse / severe (ICFR-MW + fraud + AAER). -> data/ext/graph.pt"""
import sys, time, itertools
import numpy as np, pandas as pd, torch, scipy.sparse as sp
from collections import defaultdict
t0=time.time()
YEARS=list(range(2005,2023)); CAP=40; PCAP=50; OWN_TOPK=15; OWN_MIN=4; OWN_SKIP=1000
FEAT=["roa","roe","npm","opmad","gpm","cfm","ptpm","gprof","bm","ptb","pe_inc","divyield",
      "de_ratio","debt_at","debt_ebitda","lt_debt","intcov_ratio","capital_ratio",
      "curr_ratio","quick_ratio","cash_ratio","ocf_lct","at_turn","inv_turn","rect_turn",
      "accrual","mktcap","ret_crsp"]
E="data/ext/"
comp=pd.read_pickle("data/comp_company.pkl").dropna(subset=["cik"]); comp["cik"]=comp["cik"].astype("int64")
fr=pd.read_pickle(E+"firm_ratio.pkl"); fr=fr[fr["year"].isin(YEARS)].copy()
aud=pd.read_pickle(E+"auditor.pkl"); res=pd.read_pickle(E+"restate.pkl"); bd=pd.read_pickle(E+"board_edges.pkl")
sox=pd.read_pickle(E+"sox404.pkl"); aaer=pd.read_pickle(E+"aaer.pkl"); fap=pd.read_pickle(E+"formap.pkl")
hold=pd.read_pickle(E+"holdings.pkl")

gvkeys=sorted(fr["gvkey"].unique()); gid={g:i for i,g in enumerate(gvkeys)}; N=len(gvkeys); TY=len(YEARS)
print(f"[nodes] {N}  years {YEARS[0]}-{YEARS[-1]}",file=sys.stderr)
g2cik=dict(zip(comp["gvkey"],comp["cik"])); g2sic=dict(zip(comp["gvkey"],comp["sic"]))
uni=pd.DataFrame({"gvkey":gvkeys}); uni["cik"]=uni["gvkey"].map(g2cik); uni=uni.dropna(subset=["cik"]); uni["cik"]=uni["cik"].astype("int64")
dup=uni["cik"].duplicated(keep=False); cik2g={int(c):g for g,c in zip(uni.loc[~dup,"gvkey"],uni.loc[~dup,"cik"])}
cy2g={}
for r in fr[["gvkey","cusip","year"]].dropna().itertuples():
    if isinstance(r.cusip,str) and len(r.cusip)==8: cy2g[(r.cusip,int(r.year))]=r.gvkey

# features / active / labels
X=np.full((TY,N,len(FEAT)),np.nan,np.float32); active=np.zeros((TY,N),bool); ffi=np.full((TY,N),-1,np.int64)
for yi,y in enumerate(YEARS):
    sub=fr[fr["year"]==y]
    arr=sub[FEAT].to_numpy(np.float64); arr[~np.isfinite(arr)]=np.nan
    idx=[gid[g] for g in sub["gvkey"]]
    X[yi,idx,:]=arr.astype(np.float32)
    for r in sub.itertuples(): active[yi,gid[r.gvkey]]=True; ffi[yi,gid[r.gvkey]]=int(r.ffi12) if pd.notna(r.ffi12) else -1

res=res.dropna(subset=["cik"]); res["cik"]=res["cik"].astype("int64")
def annset(df,col=None,val=None):
    d=defaultdict(set); sub=df if col is None else df[df[col]==val]
    for r in sub.itertuples():
        g=cik2g.get(int(r.cik))
        if g is not None: d[int(r.ann_year)].add(g)
    return d
ann_any=annset(res); ann_adv=annset(res,"res_adverse",1.0); ann_fraud=annset(res,"res_fraud",1.0)
aaer2=aaer.dropna(subset=["cik","ann_year"]); aaer2["cik"]=aaer2["cik"].astype("int64")
ann_aaer=defaultdict(set)
for r in aaer2.itertuples():
    g=cik2g.get(int(r.cik));
    if g is not None: ann_aaer[int(r.ann_year)].add(g)
sox_mw=sox[(sox["ic_is_effective"]=="N")&sox["cik"].notna()&sox["file_year"].notna()]
ann_mw=defaultdict(set)
for r in sox_mw.itertuples():
    g=cik2g.get(int(r.cik));
    if g is not None: ann_mw[int(r.file_year)].add(g)
ann_sev={y:(ann_mw.get(y,set())|ann_fraud.get(y,set())|ann_aaer.get(y,set())) for y in range(2005,2025)}

def mk_label(ann):
    lab=np.full((TY,N),-1,np.int64); rnow=np.zeros((TY,N),np.int64)
    for yi,y in enumerate(YEARS):
        pos=ann.get(y+1,set()); now=ann.get(y,set())
        for i in range(N):
            if active[yi,i]: lab[yi,i]=1 if gvkeys[i] in pos else 0
            if gvkeys[i] in now: rnow[yi,i]=1
    return lab,rnow
label,rn=mk_label(ann_any); label_adv,rn_adv=mk_label(ann_adv); label_sev,rn_sev=mk_label(ann_sev)
for nm,l in [("any",label),("adv",label_adv),("sev",label_sev)]:
    rates=[round(float(l[yi][active[yi]&(l[yi]>=0)].mean()),3) for yi in range(TY)]
    print(f"[label {nm}] pos-rate/yr {rates}",file=sys.stderr)

def to_ei(edic):
    if not edic: return torch.zeros((2,0),dtype=torch.long),torch.zeros((0,),dtype=torch.float)
    es=list(edic.items()); a=[x for (x,y),_ in es]; b=[y for (x,y),_ in es]; w=[w for _,w in es]
    return torch.tensor([a+b,b+a],dtype=torch.long), torch.tensor(w+w,dtype=torch.float)
def clique(members,edic,cap):
    m=sorted(set(members))
    if len(m)<2: return
    if len(m)<=cap+1:
        for a,b in itertools.combinations(m,2): edic[(a,b)]+=1.0
    else:
        L=len(m)
        for ii,a in enumerate(m):
            for off in range(1,cap+1):
                b=m[(ii+off)%L]
                if a!=b: edic[((a,b) if a<b else (b,a))]+=1.0

# auditor x SIC2 (L3)
aud=aud.dropna(subset=["cik","auditor_fkey"]); aud["cik"]=aud["cik"].astype("int64")
audE={y:defaultdict(float) for y in YEARS}
for y in YEARS:
    grp=defaultdict(list)
    for r in aud[aud["fyear"]==y].itertuples():
        g=cik2g.get(int(r.cik)); s=g2sic.get(g) if g is not None else None
        if g is None or s is None or (isinstance(s,float) and np.isnan(s)): continue
        grp[(r.auditor_fkey,int(float(s))//100)].append(gid[g])
    for k,m in grp.items(): clique(m,audE[y],CAP)
# office (L2): auditor x city
sox_off=sox[sox["aud_city"].notna()&sox["auditor_fkey"].notna()&sox["fy"].notna()&sox["cik"].notna()]
offE={y:defaultdict(float) for y in YEARS}
for y in YEARS:
    grp=defaultdict(list)
    for r in sox_off[sox_off["fy"]==y].itertuples():
        g=cik2g.get(int(r.cik))
        if g is not None: grp[(r.auditor_fkey,str(r.aud_city).strip().upper(),str(r.aud_state))].append(gid[g])
    for k,m in grp.items(): clique(m,offE[y],CAP)
# partner (L1): shared engagement partner (2017+)
fap=fap.dropna(subset=["partner_id","cik","fpe_year"]); fap["cik"]=fap["cik"].astype("int64")
parE={y:defaultdict(float) for y in YEARS}
for y in YEARS:
    if y<2017: continue
    grp=defaultdict(list)
    for r in fap[fap["fpe_year"]==y].itertuples():
        g=cik2g.get(int(r.cik))
        if g is not None: grp[r.partner_id].append(gid[g])
    for k,m in grp.items(): clique(m,parE[y],PCAP)
# board
bd=bd.dropna(subset=["src_cik","dst_cik"]).copy()
for c in ["src_cik","dst_cik","y0","y1"]: bd[c]=bd[c].astype("int64")
bdE={y:defaultdict(float) for y in YEARS}
for r in bd.itertuples():
    ga=cik2g.get(int(r.src_cik)); gb=cik2g.get(int(r.dst_cik))
    if ga is None or gb is None or ga==gb: continue
    a,b=gid[ga],gid[gb]; e=(a,b) if a<b else (b,a)
    for y in range(max(2005,int(r.y0)),min(2022,int(r.y1))+1): bdE[y][e]+=float(r.n_shared_dir)
# ownership kNN
hold["mgrno"]=hold["mgrno"].astype(str); ownE={y:defaultdict(float) for y in YEARS}
for y in YEARS:
    sub=hold[hold["year"]==y].drop_duplicates(["mgrno","cusip"]); firms=[];mgrs=[];mid={}
    for r in sub.itertuples():
        g=cy2g.get((r.cusip,y))
        if g is None: continue
        firms.append(gid[g]); mgrs.append(mid.setdefault(r.mgrno,len(mid)))
    if not firms: continue
    M=sp.csr_matrix((np.ones(len(firms)),(firms,mgrs)),shape=(N,len(mid)))
    cs=np.asarray(M.sum(0)).ravel(); M=M[:,cs<=OWN_SKIP]; C=(M@M.T).tocoo()
    rows=defaultdict(list)
    for a,b,c in zip(C.row,C.col,C.data):
        if a<b and c>=OWN_MIN: rows[a].append((c,b)); rows[b].append((c,a))
    for a,lst in rows.items():
        lst.sort(reverse=True)
        for c,b in lst[:OWN_TOPK]: e=(a,b) if a<b else (b,a); ownE[y][e]=max(ownE[y][e],float(c))

snaps=[]
for yi,y in enumerate(YEARS):
    rels={}
    for nm,E_ in [("partner",parE),("office",offE),("auditor",audE),("board",bdE),("ownership",ownE)]:
        rels[nm]=to_ei(E_[y])
    snaps.append(rels)
    print(f"[edges {y}] P={len(parE[y])} O={len(offE[y])} A={len(audE[y])} B={len(bdE[y])} Own={len(ownE[y])}",file=sys.stderr)

torch.save({"gvkeys":gvkeys,"years":YEARS,"feat_names":FEAT,
  "X":torch.from_numpy(X),"active":torch.from_numpy(active),"ffi":torch.from_numpy(ffi),
  "label":torch.from_numpy(label),"restated_now":torch.from_numpy(rn),
  "label_adverse":torch.from_numpy(label_adv),"restated_now_adverse":torch.from_numpy(rn_adv),
  "label_severe":torch.from_numpy(label_sev),"restated_now_severe":torch.from_numpy(rn_sev),
  "snapshots":snaps,"relations":["partner","office","auditor","board","ownership"]},
  "data/ext/graph.pt")
print(f"[build_graph_ext done {time.time()-t0:.1f}s -> data/ext/graph.pt]",file=sys.stderr)
