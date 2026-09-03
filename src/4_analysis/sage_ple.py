#!/usr/bin/env python3
"""Reviewer-cell ablation: does the ENCODING alone rescue a standard GNN?
Runs benchmark.py's GraphSAGE (2-layer SAGEConv, merged multiplex) but with ACE-v2's inputs:
  CELL=ple        -> [femb(PLE(Xz)), OWN]                       (encoding-only question)
  CELL=ple_expenc -> [femb(PLE(Xz)), OWN, EXPENC(exposures)]    (full ACE-v2 input set, SAGE machinery)
If SAGE+PLE ~= ACE-v2 the contribution is the encoding; if it stays below, the relation-gated
residual architecture earns its place. 3-seed bag, val+test recorded.
ENV: MODE=recent LABEL=severe CELL=ple NSEED=3 THREADS=3
"""
import json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch_geometric.nn as pygnn
from torch_geometric.utils import scatter
from sklearn.metrics import roc_auc_score, average_precision_score

torch.set_num_threads(int(os.environ.get("THREADS","3")))
MODE=os.environ.get("MODE","recent"); LABEL=os.environ.get("LABEL","severe")
CELL=os.environ.get("CELL","ple"); NSEED=int(os.environ.get("NSEED","3"))
NB=32; FEMB=12; NEB=8
LKEY={"severe":"label_severe"}.get(LABEL,"label"); RKEY={"severe":"restated_now_severe"}.get(LABEL,"restated_now")
G=torch.load("data/ext/graph.pt",weights_only=False); YEARS=G["years"]; yidx={y:i for i,y in enumerate(YEARS)}
X=G["X"].numpy().copy(); active=G["active"].numpy(); label=G[LKEY].numpy(); rn=G[RKEY].numpy().astype(np.float32)
Tall,N,Fdim=X.shape; snaps=G["snapshots"]
REL=["partner","office","auditor","board","ownership"] if MODE=="recent" else ["office","auditor","board","ownership"]
if MODE=="panel": USE=[yidx[y] for y in range(2005,2020)]; TR=USE[:11]; VA=[USE[11]]; TE=USE[12:15]
else: USE=[yidx[y] for y in range(2017,2023)]; TR=USE[:3]; VA=[USE[3]]; TE=USE[4:6]
print(f"[sage+{CELL}] {MODE}/{LABEL} test={[YEARS[i] for i in TE]}",file=sys.stderr)

tm=np.zeros((Tall,N),bool)
for t in TR: tm[t]=active[t]&(label[t]>=0)
Xtr0=X[tm]; lo=np.nanpercentile(Xtr0,1,0); hi=np.nanpercentile(Xtr0,99,0); med=np.nanmedian(Xtr0,0)
Xc=np.clip(X,lo,hi); ix=np.where(np.isnan(Xc)); Xc[ix]=np.take(med,ix[2])
mu=np.nanmean(np.clip(Xtr0,lo,hi),0); sd=np.nanstd(np.clip(Xtr0,lo,hi),0); sd[sd<1e-6]=1
Xz=((Xc-mu)/sd).astype(np.float32)
knots=[]
for j in range(Fdim):
    q=np.quantile(Xz[tm][:,j],np.linspace(0,1,NB+1)); q[0]=-1e9; q[-1]=1e9
    knots.append(np.maximum.accumulate(q).astype(np.float32))
def ple(Xs):
    parts=[]
    for j in range(Fdim):
        kb=knots[j]; x=Xs[:,j:j+1]
        parts.append(np.clip((x-kb[:-1][None,:])/(kb[1:][None,:]-kb[:-1][None,:]+1e-9),0.,1.).astype(np.float32))
    return np.concatenate(parts,1)
PLE_t={t:torch.from_numpy(ple(Xz[t])) for t in USE}
OWN=np.zeros((Tall,N,2),np.float32)
for t in range(Tall): OWN[t,:,0]=rn[t]; OWN[t,:,1]=rn[max(t-1,0)]
OWN_t=torch.from_numpy(OWN)
def adj(t,c): ei,w=snaps[t][c]; return ei,w
def mean_agg(v,ei,w):
    if ei.size(1)==0: return torch.zeros_like(v)
    s,d=ei; num=scatter(v[s]*w,d,0,dim_size=N,reduce='sum'); den=scatter(w,d,0,dim_size=N,reduce='sum').clamp(min=1e-6); return num/den
EXPE_t={}
if CELL=="ple_expenc":
    ALL=([r for r in REL if r in ("partner","office","auditor")])+["board","ownership"]
    rnt=[torch.from_numpy(rn[t]) for t in range(Tall)]
    EXP=np.zeros((Tall,N,len(ALL)*2),np.float32)
    for t in range(Tall):
        for ci,c in enumerate(ALL):
            ei,w=adj(t,c); EXP[t,:,ci]=mean_agg(rnt[t],ei,w).numpy()
            tl=max(t-1,0); eil,wl=adj(tl,c); EXP[t,:,len(ALL)+ci]=mean_agg(rnt[tl],eil,wl).numpy()
    ks=[]
    for j in range(EXP.shape[2]):
        v=EXP[:,:,j][tm]; vp=v[v>1e-9]
        if len(vp)<50: q=np.linspace(1e-9,1.0,NEB+1)
        else: q=np.quantile(vp,np.linspace(0,1,NEB+1)); q[0]=1e-9; q[-1]=max(float(q[-1]),1.0)+1e-6
        ks.append(np.maximum.accumulate(q).astype(np.float32))
    def expenc(E):
        parts=[]
        for j in range(E.shape[1]):
            x=E[:,j:j+1]; kb=ks[j]; flag=(x>1e-9).astype(np.float32)
            ramps=np.clip((x-kb[:-1][None,:])/(kb[1:][None,:]-kb[:-1][None,:]+1e-9),0.,1.).astype(np.float32)
            parts.append(np.concatenate([flag,ramps*flag],1))
        return np.concatenate(parts,1)
    EXPE_t={t:torch.from_numpy(expenc(EXP[t])) for t in USE}
merged=[]
for t in range(Tall):
    eis=[];ews=[]
    for c in REL: ei,w=adj(t,c); eis.append(ei); ews.append(w)
    merged.append((torch.cat(eis,1),torch.cat(ews)))
lab_t=torch.from_numpy(label)
ytr=np.concatenate([label[t][active[t]&(label[t]>=0)] for t in TR]); spw=float((ytr==0).sum()/max((ytr==1).sum(),1))
POSW=torch.tensor([spw])
def metr(y,p):
    o=np.argsort(-p)
    return {"roc":round(float(roc_auc_score(y,p)),4),"pr":round(float(average_precision_score(y,p)),4),
            "recall@10%":round(float(y[o[:max(1,len(p)//10)]].sum()/max(y.sum(),1)),4)}

aux=2+(EXPE_t[USE[0]].shape[1] if CELL=="ple_expenc" else 0)
class SAGEPLE(nn.Module):
    def __init__(s,H=64,drop=0.4):
        super().__init__(); s.femb=nn.Linear(NB,FEMB); s.drop=nn.Dropout(drop)
        s.c1=pygnn.SAGEConv(Fdim*FEMB+aux,H); s.c2=pygnn.SAGEConv(H,H); s.head=nn.Linear(H,1)
    def forward(s,t):
        p=s.femb(PLE_t[t].view(N,Fdim,NB)).reshape(N,Fdim*FEMB)
        cols=[p,OWN_t[t]]+([EXPE_t[t]] if CELL=="ple_expenc" else [])
        x=torch.cat(cols,1); ei,_=merged[t]
        h=F.relu(s.c1(x,ei)); h=s.drop(h); h=F.relu(s.c2(h,ei))
        return s.head(h).squeeze(1)
def train_one(seed):
    torch.manual_seed(seed); m=SAGEPLE(); opt=torch.optim.Adam(m.parameters(),lr=5e-3,weight_decay=5e-4)
    tri={t:torch.from_numpy(np.where(active[t]&(label[t]>=0))[0]).long() for t in TR}
    vai={t:np.where(active[t]&(label[t]>=0))[0] for t in VA}; best=-1;bs=None;bad=0
    for ep in range(200):
        m.train(); opt.zero_grad(); loss=0
        for t in TR: o=m(t); idx=tri[t]; loss=loss+F.binary_cross_entropy_with_logits(o[idx],lab_t[t][idx].float(),pos_weight=POSW)
        loss.backward(); opt.step()
        if ep%5==0:
            m.eval(); ys=[];ps=[]
            with torch.no_grad():
                for t in VA: o=m(t); a=vai[t]; ps.append(torch.sigmoid(o[a]).numpy()); ys.append(label[t][a])
            vp=average_precision_score(np.concatenate(ys),np.concatenate(ps))
            if vp>best: best=vp; bs={k:v.clone() for k,v in m.state_dict().items()};bad=0
            else: bad+=1
            if bad>8: break
    if bs: m.load_state_dict(bs)
    m.eval()
    def coll(ts):
        ys=[];ps=[]
        with torch.no_grad():
            for t in ts: o=m(t); a=np.where(active[t]&(label[t]>=0))[0]; ps.append(torch.sigmoid(o[a]).numpy()); ys.append(label[t][a])
        return np.concatenate(ys),np.concatenate(ps)
    yv,pv=coll(VA); yt,pt=coll(TE); return pv,pt,yv,yt,best
t0=time.time(); PV=[];PT=[]
for sd in range(NSEED):
    pv,pt,yv,yt,bv=train_one(sd); PV.append(pv); PT.append(pt)
    print(f"  [s{sd}] val_pr={bv:.4f} test={metr(yt,pt)}",file=sys.stderr)
RES={"cell":f"SAGE+{CELL}","mode":MODE,"label":LABEL,"nseed":NSEED,
     "val_bag":metr(yv,np.mean(PV,0)),"test_bag":metr(yt,np.mean(PT,0)),"runtime_sec":round(time.time()-t0,1)}
os.makedirs("data/ext/pure/v2",exist_ok=True)
json.dump(RES,open(f"data/ext/pure/v2/sage_{CELL}_{MODE}_{LABEL}.json","w"),indent=2)
print(json.dumps(RES,indent=1))
