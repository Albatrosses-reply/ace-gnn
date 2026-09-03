#!/usr/bin/env python3
"""Build temporal multiplex firm graph from Wave1/Wave2 pkls -> data/graph.pt
Nodes = union of firm_ratio gvkeys. Per year: features, active mask, t+1 restatement
label, 3 relations (auditor / board / ownership). Standardization deferred to train."""
import sys, time, itertools
import numpy as np, pandas as pd, torch
from collections import defaultdict

t0=time.time()
YEARS=list(range(2010,2020))
FEAT=["roa","roe","npm","opmad","gpm","cfm","ptpm","gprof","bm","ptb","pe_inc","divyield",
      "de_ratio","debt_at","debt_ebitda","lt_debt","intcov_ratio","capital_ratio",
      "curr_ratio","quick_ratio","cash_ratio","ocf_lct","at_turn","inv_turn","rect_turn",
      "accrual","mktcap","ret_crsp"]
AUD_GROUP_CAP=40        # max co-clients linked per firm within an auditor×SIC2 group
OWN_TOPK=15             # common-ownership = each firm linked to its top-K most co-owned peers
OWN_MIN_SHARED=4        # ...requiring at least this many shared institutions
OWN_SKIP_MGR_OVER=1000  # drop ubiquitous index-fund managers (hold > this many of our firms)

comp=pd.read_pickle("data/comp_company.pkl")
fr=pd.read_pickle("data/firm_ratio.pkl")
aud=pd.read_pickle("data/auditor.pkl")
res=pd.read_pickle("data/restate.pkl")
bd=pd.read_pickle("data/board_edges.pkl")
try:
    hold=pd.read_pickle("data/holdings.pkl")
except FileNotFoundError:
    hold=None; print("[WARN] holdings.pkl missing -> ownership layer empty",file=sys.stderr)

# ---- node universe ----
fr=fr[fr["year"].isin(YEARS)].copy()
gvkeys=sorted(fr["gvkey"].unique())
gid={g:i for i,g in enumerate(gvkeys)}; N=len(gvkeys)
print(f"[nodes] {N} gvkeys",file=sys.stderr)

# ---- id maps ----
comp=comp.dropna(subset=["cik"]); comp["cik"]=comp["cik"].astype("int64")
g2cik=dict(zip(comp["gvkey"],comp["cik"]))
g2sic=dict(zip(comp["gvkey"],comp["sic"]))
# cik->gvkey restricted to universe (drop ambiguous duplicates)
uni=pd.DataFrame({"gvkey":gvkeys}); uni["cik"]=uni["gvkey"].map(g2cik)
uni=uni.dropna(subset=["cik"]); uni["cik"]=uni["cik"].astype("int64")
dup=uni["cik"].duplicated(keep=False)
cik2g={int(c):g for g,c in zip(uni.loc[~dup,"gvkey"],uni.loc[~dup,"cik"])}
print(f"[idmap] gvkey->cik {len(g2cik)} ; unique cik->gvkey {len(cik2g)} (dropped {dup.sum()} ambiguous)",file=sys.stderr)

# cusip->gvkey per year (from firm_ratio)
cusip_year2g={}
for r in fr[["gvkey","cusip","year"]].dropna().itertuples():
    if isinstance(r.cusip,str) and len(r.cusip)==8:
        cusip_year2g[(r.cusip,int(r.year))]=r.gvkey

# ---- features per (gvkey,year) ----
fr2=fr.set_index(["gvkey","year"])
X=np.full((len(YEARS),N,len(FEAT)),np.nan,dtype=np.float32)
active=np.zeros((len(YEARS),N),dtype=bool)
ffi=np.full((len(YEARS),N),-1,dtype=np.int64)
for yi,y in enumerate(YEARS):
    sub=fr[fr["year"]==y]
    for r in sub.itertuples():
        i=gid[r.gvkey]; active[yi,i]=True
        ffi[yi,i]=int(r.ffi12) if pd.notna(r.ffi12) else -1
    fvals=sub[["gvkey"]+FEAT].copy()
    arr=fvals[FEAT].to_numpy(dtype=np.float64)
    arr[~np.isfinite(arr)]=np.nan
    idx=[gid[g] for g in fvals["gvkey"]]
    X[yi,idx,:]=arr.astype(np.float32)
print(f"[features] active firm-years/yr: {active.sum(1)}",file=sys.stderr)

# ---- labels: restatement announced in year y+1 ----
res=res.dropna(subset=["cik"]); res["cik"]=res["cik"].astype("int64")
ann=defaultdict(set)  # year -> set of gvkeys announcing that year
for r in res.itertuples():
    g=cik2g.get(int(r.cik))
    if g is not None: ann[int(r.ann_year)].add(g)
label=np.full((len(YEARS),N),-1,dtype=np.int64)  # -1 = undefined
for yi,y in enumerate(YEARS):
    if y+1>2020: continue
    pos=ann.get(y+1,set())
    for i in range(N):
        if active[yi,i]:
            label[yi,i]= 1 if gvkeys[i] in pos else 0
for yi,y in enumerate(YEARS):
    m=label[yi]>=0
    if m.sum(): print(f"  label y{y}->{y+1}: n={m.sum()} pos={int((label[yi][m]==1).sum())} ({(label[yi][m]==1).mean()*100:.1f}%)",file=sys.stderr)
# restated_now[t,i] = firm announced a restatement in calendar year YEARS[t] (for contagion analysis)
restated_now=np.zeros((len(YEARS),N),dtype=np.int64)
for yi,y in enumerate(YEARS):
    pos=ann.get(y,set())
    for i in range(N):
        if gvkeys[i] in pos: restated_now[yi,i]=1

def undirected_unique(pairs):
    s=set()
    for a,b in pairs:
        if a==b: continue
        s.add((a,b) if a<b else (b,a))
    return s

# ---- relation 1: shared auditor within SIC2 ----
aud=aud.dropna(subset=["cik","auditor_fkey"]); aud["cik"]=aud["cik"].astype("int64")
aud_edges={y:defaultdict(float) for y in YEARS}
for y in YEARS:
    sub=aud[aud["fyear"]==y]
    grp=defaultdict(list)
    for r in sub.itertuples():
        g=cik2g.get(int(r.cik))
        if g is None: continue
        sic=g2sic.get(g)
        if sic is None or (isinstance(sic,float) and np.isnan(sic)): continue
        sic2=int(float(sic))//100
        grp[(r.auditor_fkey,sic2)].append(gid[g])
    for key,members in grp.items():
        members=sorted(set(members))
        if len(members)<2: continue
        if len(members)<=AUD_GROUP_CAP+1:
            for a,b in itertools.combinations(members,2):
                aud_edges[y][(a,b)]+=1.0
        else:  # cap: each node links to AUD_GROUP_CAP others (ring+random-ish via stride)
            m=members; L=len(m)
            for ii,a in enumerate(m):
                for off in range(1,AUD_GROUP_CAP+1):
                    b=m[(ii+off)%L]
                    e=(a,b) if a<b else (b,a)
                    if a!=b: aud_edges[y][e]+=1.0

# ---- relation 2: board interlock ----
bd2=bd.dropna(subset=["src_cik","dst_cik"]).copy()
for c in ["src_cik","dst_cik","y0","y1"]: bd2[c]=bd2[c].astype("int64")
board_edges={y:defaultdict(float) for y in YEARS}
for r in bd2.itertuples():
    ga=cik2g.get(int(r.src_cik)); gb=cik2g.get(int(r.dst_cik))
    if ga is None or gb is None or ga==gb: continue
    a,b=gid[ga],gid[gb]; e=(a,b) if a<b else (b,a)
    for y in range(max(2010,int(r.y0)),min(2019,int(r.y1))+1):
        board_edges[y][e]+=float(r.n_shared_dir)

# ---- relation 3: common ownership (kNN by #shared institutions, via sparse M@M^T) ----
import scipy.sparse as sp
own_edges={y:defaultdict(float) for y in YEARS}
if hold is not None:
    hold=hold.copy(); hold["mgrno"]=hold["mgrno"].astype(str)
    for y in YEARS:
        sub=hold[hold["year"]==y].drop_duplicates(["mgrno","cusip"])
        firms=[]; mgrs=[]
        mgr_id={}
        for r in sub.itertuples():
            g=cusip_year2g.get((r.cusip,y))
            if g is None: continue
            mi=mgr_id.setdefault(r.mgrno,len(mgr_id))
            firms.append(gid[g]); mgrs.append(mi)
        if not firms: continue
        nM=len(mgr_id)
        M=sp.csr_matrix((np.ones(len(firms)),(firms,mgrs)),shape=(N,nM))
        # drop ubiquitous managers (column sum over our firms > threshold)
        colsum=np.asarray(M.sum(0)).ravel()
        keep=colsum<=OWN_SKIP_MGR_OVER
        M=M[:,keep]
        C=(M@M.T).tocoo()   # shared-manager counts (incl diagonal)
        # top-K per row with count>=MIN, excluding self
        rows=defaultdict(list)
        for a,b,c in zip(C.row,C.col,C.data):
            if a<b and c>=OWN_MIN_SHARED:   # a<b: each undirected pair once
                rows[a].append((c,b)); rows[b].append((c,a))
        for a,lst in rows.items():
            lst.sort(reverse=True)
            for c,b in lst[:OWN_TOPK]:
                e=(a,b) if a<b else (b,a)
                own_edges[y][e]=max(own_edges[y][e],float(c))

def to_ei(edic):
    if not edic: return torch.zeros((2,0),dtype=torch.long), torch.zeros((0,),dtype=torch.float)
    es=list(edic.items())
    ei=torch.tensor([[a for (a,b),_ in es]+[b for (a,b),_ in es],
                     [b for (a,b),_ in es]+[a for (a,b),_ in es]],dtype=torch.long)  # symmetric
    w=torch.tensor([w for _,w in es]+[w for _,w in es],dtype=torch.float)
    return ei,w

snap=[]
for yi,y in enumerate(YEARS):
    rels={}
    for name,edic in [("auditor",aud_edges[y]),("board",board_edges[y]),("ownership",own_edges[y])]:
        ei,w=to_ei(edic); rels[name]=(ei,w)
    snap.append(rels)
    print(f"[edges y{y}] auditor={aud_edges[y].__len__()} board={board_edges[y].__len__()} ownership={own_edges[y].__len__()}",file=sys.stderr)

torch.save({
  "gvkeys":gvkeys,"years":YEARS,"feat_names":FEAT,
  "X":torch.from_numpy(X),"active":torch.from_numpy(active),
  "ffi":torch.from_numpy(ffi),"label":torch.from_numpy(label),
  "restated_now":torch.from_numpy(restated_now),
  "snapshots":snap,
  "params":{"AUD_GROUP_CAP":AUD_GROUP_CAP,"OWN_MIN_SHARED":OWN_MIN_SHARED},
}, "data/graph.pt")
print(f"[build_graph done {time.time()-t0:.1f}s -> data/graph.pt]",file=sys.stderr)
