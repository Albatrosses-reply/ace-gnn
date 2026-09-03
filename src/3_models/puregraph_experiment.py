#!/usr/bin/env python3
"""PURE-GRAPH approach — NO GBDT, NO tabular ensemble. A completely new architecture whose ONLY engine is
graph neural message passing. Bet: the signal the strong tabular+exposure baselines (RF/XGB+graph) miss is
(i) the TIME axis (risk accumulating year over year) and (ii) MULTI-HOP contagion through the auditor multiplex,
neither of which a static 1-hop hand-crafted exposure feature captures.

Architecture "TXM-Net" (Temporal multipleX Message net), v-tunable:
  per year t:  x -> node encoder -> L layers of RELATION-AWARE message passing over the multiplex
               -> GRUCell accumulates a per-firm RISK STATE across years (temporal contagion)
               -> head -> base logit -> APPNP collective-inference propagation of PREDICTED risk (K steps)
  prediction = seed-bagged GNN output. No tree, no tabular stacking.
ENV: MODE=recent|panel LABEL=severe|label  VARIANT=v1...  (knobs via env: H, L, KPROP, NSEED, AGG, TEMPORAL, REL)
Run from repo root: MODE=recent LABEL=severe python3 src/3_models/puregraph_experiment.py
"""
import json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.utils import scatter, softmax as gsoftmax
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb, lightgbm as lgb
torch.manual_seed(0); np.random.seed(0); DEV="cpu"
MODE=os.environ.get("MODE","recent"); LABEL=os.environ.get("LABEL","severe"); VAR=os.environ.get("VARIANT","v1")
H=int(os.environ.get("H","96")); L=int(os.environ.get("L","2")); KPROP=int(os.environ.get("KPROP","5"))
NSEED=int(os.environ.get("NSEED","5")); AGG=os.environ.get("AGG","mean"); TEMPORAL=int(os.environ.get("TEMPORAL","1"))
ENC_DEPTH=int(os.environ.get("ENC_DEPTH","1")); SKIP=int(os.environ.get("SKIP","1"))  # deep residual encoder + feature skip to head
USE_EXP=int(os.environ.get("USE_EXP","0"))  # feed graph-derived neighbor-exposure features into the GNN node inputs (still pure-graph)
LKEY={"adverse":"label_adverse","severe":"label_severe"}.get(LABEL,"label")
RKEY={"adverse":"restated_now_adverse","severe":"restated_now_severe"}.get(LABEL,"restated_now")
G=torch.load("data/ext/graph.pt",weights_only=False); YEARS=G["years"]; yidx={y:i for i,y in enumerate(YEARS)}
X=G["X"].numpy().copy(); active=G["active"].numpy(); label=G[LKEY].numpy(); rn=G[RKEY].numpy().astype(np.float32)
Tall,N,Fdim=X.shape; snaps=G["snapshots"]
REL=["partner","office","auditor","board","ownership"] if MODE=="recent" else ["office","auditor","board","ownership"]
AUD=[r for r in REL if r in ("partner","office","auditor")]
if MODE=="panel": USE=[yidx[y] for y in range(2005,2020)];TR=USE[:11];VA=[USE[11]];TE=USE[12:15]
else: USE=[yidx[y] for y in range(2017,2023)];TR=USE[:3];VA=[USE[3]];TE=USE[4:6]
SEQ=USE  # temporal context = all observed years up to test
print(f"[puregraph {VAR}] MODE={MODE} label={LABEL} H={H} L={L} K={KPROP} agg={AGG} temporal={TEMPORAL} rels={REL} test={[YEARS[i] for i in TE]}",file=sys.stderr)
# train-stat preprocessing (no leakage)
tm=np.zeros((Tall,N),bool)
for t in TR: tm[t]=active[t]&(label[t]>=0)
Xtr0=X[tm]; lo=np.nanpercentile(Xtr0,1,0); hi=np.nanpercentile(Xtr0,99,0); med=np.nanmedian(Xtr0,0)
Xc=np.clip(X,lo,hi); ix=np.where(np.isnan(Xc)); Xc[ix]=np.take(med,ix[2])
mu=np.nanmean(np.clip(Xtr0,lo,hi),0); sd=np.nanstd(np.clip(Xtr0,lo,hi),0); sd[sd<1e-6]=1
Xz=((Xc-mu)/sd).astype(np.float32)
OWN=np.zeros((Tall,N,2),np.float32)
for t in range(Tall): OWN[t,:,0]=rn[t]; OWN[t,:,1]=rn[max(t-1,0)]
def adj(t,c): ei,w=snaps[t][c]; return ei.to(DEV),w.to(DEV)
A=[{c:adj(t,c) for c in REL} for t in range(Tall)]
def _magg(v,ei,w):
    if ei.size(1)==0: return torch.zeros(N)
    s,d=ei; num=scatter((v[s]*w),d,0,dim_size=N,reduce='sum'); den=scatter(w,d,0,dim_size=N,reduce='sum').clamp(min=1e-6); return num/den
EXP=np.zeros((Tall,N,len(REL)*2),np.float32)   # neighbor problem-rate per relation (current + 1yr lag); graph-derived
for t in range(Tall):
    for ci,c in enumerate(REL):
        ei,w=A[t][c]; EXP[t,:,ci]=_magg(torch.from_numpy(rn[t]),ei,w).numpy()
        tl=max(t-1,0); ei2,w2=A[tl][c]; EXP[t,:,len(REL)+ci]=_magg(torch.from_numpy(rn[tl]),ei2,w2).numpy()
_parts=[Xz,OWN]+([EXP] if USE_EXP else [])
NF=np.concatenate(_parts,2).astype(np.float32); NF_t=torch.from_numpy(NF); lab_t=torch.from_numpy(label); nfeat=NF.shape[2]
ytr=np.concatenate([label[t][active[t]&(label[t]>=0)] for t in TR]); spw=float((ytr==0).sum()/max((ytr==1).sum(),1))
yte=np.concatenate([label[t][active[t]&(label[t]>=0)] for t in TE]); POSW=torch.tensor([spw])
def metr(y,p):
    o=np.argsort(-p); return {"roc":round(float(roc_auc_score(y,p)),4),"pr":round(float(average_precision_score(y,p)),4),
        "recall@10%":round(float(y[o[:max(1,len(p)//10)]].sum()/max(y.sum(),1)),4)}

def agg_rel(h,ei,w,kind):
    if ei.size(1)==0: return torch.zeros_like(h)
    s,d=ei
    if kind=="max":
        out=scatter(h[s],d,0,dim_size=N,reduce='max')
        deg=scatter(torch.ones_like(w),d,0,dim_size=N,reduce='sum')
        return out*(deg>0).float().unsqueeze(1)   # zero out no-neighbor rows without inplace
    num=scatter(h[s]*w.unsqueeze(1),d,0,dim_size=N,reduce='sum'); den=scatter(w,d,0,dim_size=N,reduce='sum').clamp(min=1e-6)
    return num/den.unsqueeze(1)

class TXM(nn.Module):
    """Temporal multipleX message net. PURE graph: deep residual encoder + relation-aware MP + temporal GRU +
    feature-skip to head + collective inference. No tree, no tabular ensemble."""
    def __init__(s,drop=0.3):
        super().__init__()
        s.enc_in=nn.Linear(nfeat,H)
        s.enc_blocks=nn.ModuleList([nn.Sequential(nn.LayerNorm(H),nn.Linear(H,H),nn.ReLU(),nn.Dropout(drop)) for _ in range(ENC_DEPTH)])
        s.rlin=nn.ModuleList([nn.ModuleList([nn.Linear(H,H) for _ in REL]) for _ in range(L)])
        s.rw=nn.Parameter(torch.zeros(L,len(REL)))                # learned relation weights (softmax), per layer
        s.upd=nn.ModuleList([nn.Linear(2*H,H) for _ in range(L)])
        s.gru=nn.GRUCell(H,H) if TEMPORAL else None
        hd=2*H if SKIP else H
        s.head=nn.Sequential(nn.Linear(hd,H),nn.ReLU(),nn.Dropout(drop),nn.Linear(H,1))
        s.beta=nn.Parameter(torch.tensor(0.0)); s.drop=nn.Dropout(drop)
    def encode_year(s,t):
        h=F.relu(s.enc_in(NF_t[t]))
        for blk in s.enc_blocks: h=h+blk(h)       # deep residual node-feature encoder (closes the tree-vs-MLP gap)
        enc=h                                      # pre-message-passing feature embedding (skip source)
        for l in range(L):
            rw=torch.softmax(s.rw[l],0); msg=0
            for ri,c in enumerate(REL):
                ei,w=A[t][c]; msg=msg+rw[ri]*agg_rel(s.rlin[l][ri](h),ei,w,AGG)
            h=F.relu(s.upd[l](torch.cat([h,msg],1))); h=s.drop(h)
        return h,enc
    def collective(s,y,t):
        b=torch.sigmoid(s.beta)*0.6; yk=y
        for _ in range(KPROP):
            m=0
            for c in AUD:
                ei,w=A[t][c]; m=m+agg_rel(yk.unsqueeze(1),ei,w,"mean").squeeze(1)
            yk=(1-b)*y+b*(m/max(len(AUD),1))
        return yk
    def forward(s,pred_years):
        state=torch.zeros(N,H); out={}
        for t in SEQ:
            h,enc=s.encode_year(t)
            state=s.gru(h,state) if (TEMPORAL and s.gru is not None) else h
            hin=torch.cat([state,enc],1) if SKIP else state   # feature skip: raw-feature signal reaches head directly
            y=s.head(hin).squeeze(1); out[t]=s.collective(y,t)
        return out

def collect(m,ts):
    ys=[];ps=[]
    with torch.no_grad():
        out=m(ts)
        for t in ts: a=np.where(active[t]&(label[t]>=0))[0]; ps.append(torch.sigmoid(out[t][a]).numpy()); ys.append(label[t][a])
    return np.concatenate(ys),np.concatenate(ps)
def train_one(seed):
    torch.manual_seed(seed); m=TXM()
    opt=torch.optim.Adam(m.parameters(),lr=3e-3,weight_decay=2e-4)
    tri={t:torch.from_numpy(np.where(active[t]&(label[t]>=0))[0]).long() for t in TR}
    best=-1;bs=None;bad=0
    for ep in range(300):
        m.train(); opt.zero_grad(); out=m(TR); loss=0
        for t in TR:
            idx=tri[t]; loss=loss+F.binary_cross_entropy_with_logits(out[t][idx],lab_t[t][idx].float(),pos_weight=POSW)
        loss.backward(); opt.step()
        if ep%5==0:
            m.eval(); yv,pv=collect(m,VA); vp=average_precision_score(yv,pv)
            if vp>best: best=vp; bs={k:v.clone() for k,v in m.state_dict().items()};bad=0
            else: bad+=1
            if bad>12: break
    if bs: m.load_state_dict(bs)
    m.eval(); yt,pt=collect(m,TE); yv,pv=collect(m,VA)
    return pt,pv,yt
t0=time.time(); PT=[];PV=[]
for sd in range(NSEED):
    pt,pv,yt=train_one(sd); PT.append(pt); PV.append(pv); print(f"  [{VAR} s{sd}] roc={roc_auc_score(yte,pt):.4f} pr={average_precision_score(yte,pt):.4f}",file=sys.stderr)
pbag=np.mean(PT,0)
# reference baselines (for honest comparison only; NOT part of the model)
def pool(ts,arr):
    xs=[];ys=[]
    for t in ts: a=np.where(active[t]&(label[t]>=0))[0]; xs.append(arr[t][a]); ys.append(label[t][a])
    return np.concatenate(xs),np.concatenate(ys)
# graph-exposure features for the +graph baselines (same as benchmark.py)
def magg(v,ei,w):
    if ei.size(1)==0: return torch.zeros(N)
    s,d=ei; num=scatter((v[s]*w),d,0,dim_size=N,reduce='sum'); den=scatter(w,d,0,dim_size=N,reduce='sum').clamp(min=1e-6); return num/den
EXP=np.zeros((Tall,N,len(REL)*2),np.float32)
for t in range(Tall):
    for ci,c in enumerate(REL):
        ei,w=A[t][c]; EXP[t,:,ci]=magg(torch.from_numpy(rn[t]),ei,w).numpy()
        tl=max(t-1,0); ei2,w2=A[tl][c]; EXP[t,:,len(REL)+ci]=magg(torch.from_numpy(rn[tl]),ei2,w2).numpy()
GF=np.concatenate([Xz,OWN,EXP],2).astype(np.float32)
Xtr_g,ytr2=pool(TR,GF); Xte_g,_=pool(TE,GF)
rf=RandomForestClassifier(n_estimators=400,min_samples_leaf=5,class_weight="balanced_subsample",n_jobs=4,random_state=0).fit(Xtr_g,ytr2)
pRF=rf.predict_proba(Xte_g)[:,1]
xg=xgb.XGBClassifier(n_estimators=150,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,min_child_weight=5,reg_lambda=2.0,scale_pos_weight=spw,n_jobs=4,tree_method="hist").fit(Xtr_g,ytr2)
pXG=xg.predict_proba(Xte_g)[:,1]
BASE={"RF+graph":pRF,"XGB+graph":pXG}; bb=max(BASE,key=lambda k:roc_auc_score(yte,BASE[k]))
def boot(pa,pb,B=3000):
    rng=np.random.default_rng(0);n=len(yte);d=[]
    for _ in range(B):
        idx=rng.integers(0,n,n)
        if 0<yte[idx].sum()<len(idx): d.append(roc_auc_score(yte[idx],pa[idx])-roc_auc_score(yte[idx],pb[idx]))
    d=np.array(d);return {"mean":round(float(d.mean()),4),"ci":[round(float(np.percentile(d,2.5)),4),round(float(np.percentile(d,97.5)),4)],"p":round(float((d>0).mean()),4)}
RES={"variant":VAR,"mode":MODE,"label":LABEL,"knobs":{"H":H,"L":L,"KPROP":KPROP,"AGG":AGG,"TEMPORAL":TEMPORAL,"NSEED":NSEED},
     "puregraph":metr(yte,pbag),"RF+graph":metr(yte,pRF),"XGB+graph":metr(yte,pXG),
     "vs_strongest_base":bb,"boot_vs_"+bb:boot(pbag,BASE[bb]),"runtime_sec":round(time.time()-t0,1)}
os.makedirs("data/ext/puregraph",exist_ok=True)
json.dump(RES,open(f"data/ext/puregraph/{VAR}_{MODE}_{LABEL}.json","w"),indent=2)
np.savez(f"data/ext/puregraph/{VAR}_{MODE}_{LABEL}_preds.npz",y=yte,pg=pbag,rf=pRF,xgb=pXG)
print(json.dumps({k:RES[k] for k in ["variant","puregraph","RF+graph","XGB+graph","vs_strongest_base","boot_vs_"+bb,"runtime_sec"]},ensure_ascii=False,indent=2))
