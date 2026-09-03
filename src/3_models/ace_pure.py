#!/usr/bin/env python3
"""ACE-GNN (PURE) — the paper's headline model, with NO gradient boosting anywhere.
A clean re-implementation of the graph-only ACE architecture extracted from ace_experiment.py:
  - inputs  = own event history (restated_now, t & t-1) + neighbour exposure per relation (t & t-1)
              [NO firm financial ratios in the graph-only headline; NO XGBoost; NO GBDT-leaf]
  - encoder = relation-gated attention over {auditor-hierarchy, board, ownership}
  - hierarchy = monotone auditor levels  w_partner >= w_office >= w_auditor  (imposed by design)
  - collective inference = APPNP-style propagation of the PREDICTED logit over the auditor graph
  - seed-bagged for variance reduction
No ensemble, no meta-blend, no tabular fusion -> a single pure GNN, reproducible from this file alone.

ENV: MODE=recent|panel  LABEL=label|severe  NSEED=6  ABLATE=0|1  USEFEAT=0|1  OUT=...
Run from repo root:  MODE=recent LABEL=severe ABLATE=1 python3 src/3_models/ace_pure.py
"""
import json, sys, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.utils import scatter, softmax as gsoftmax
from sklearn.metrics import roc_auc_score, average_precision_score

torch.manual_seed(0); np.random.seed(0); DEV="cpu"
MODE=os.environ.get("MODE","recent"); LABEL_MODE=os.environ.get("LABEL","severe")
NSEED=int(os.environ.get("NSEED","6")); ABLATE=os.environ.get("ABLATE","0")=="1"
USEFEAT=os.environ.get("USEFEAT","0")=="1"   # headline = graph-only (False); True = +financial ratios ablation
OUT=os.environ.get("OUT",f"data/ext/pure/ace_pure_{MODE}_{LABEL_MODE}.json")
LKEY={"adverse":"label_adverse","severe":"label_severe"}.get(LABEL_MODE,"label")
RKEY={"adverse":"restated_now_adverse","severe":"restated_now_severe"}.get(LABEL_MODE,"restated_now")

G=torch.load("data/ext/graph.pt", weights_only=False)
YEARS=G["years"]; FEAT=G["feat_names"]; yidx={y:i for i,y in enumerate(YEARS)}
X=G["X"].numpy().copy(); active=G["active"].numpy(); label=G[LKEY].numpy()
rn=G[RKEY].numpy().astype(np.float32)
Tall,N,Fdim=X.shape; snaps=G["snapshots"]
AUD=["partner","office","auditor"] if MODE=="recent" else ["office","auditor"]
OTH=["board","ownership"]; ALL=AUD+OTH
if MODE=="panel": USE=[yidx[y] for y in range(2005,2020)]; TR=USE[:11]; VA=[USE[11]]; TE=USE[12:15]
else: USE=[yidx[y] for y in range(2017,2023)]; TR=USE[:3]; VA=[USE[3]]; TE=USE[4:6]
print(f"[ACE-pure] MODE={MODE} label={LABEL_MODE} usefeat={USEFEAT} nseed={NSEED} "
      f"train={[YEARS[i] for i in TR]} test={[YEARS[i] for i in TE]} aud={AUD}",file=sys.stderr)

# ---- standardize financial ratios (only used if USEFEAT) ----
tm=np.zeros((Tall,N),bool)
for t in TR: tm[t]=active[t]&(label[t]>=0)
Xtr0=X[tm]; lo=np.nanpercentile(Xtr0,1,0); hi=np.nanpercentile(Xtr0,99,0); med=np.nanmedian(Xtr0,0)
Xc=np.clip(X,lo,hi); ix=np.where(np.isnan(Xc)); Xc[ix]=np.take(med,ix[2])
mu=np.nanmean(np.clip(Xtr0,lo,hi),0); sd=np.nanstd(np.clip(Xtr0,lo,hi),0); sd[sd<1e-6]=1
Xz=((Xc-mu)/sd).astype(np.float32)
XL=np.stack([Xz[max(t-1,0)] for t in range(Tall)],0)

def adj(t,c): ei,w=snaps[t][c]; return ei.to(DEV),w.to(DEV)
def mean_agg(vec,ei,w):
    if ei.size(1)==0: return torch.zeros_like(vec)
    src,dst=ei; v=vec if vec.dim()>1 else vec.unsqueeze(1)
    num=scatter(v[src]*w.unsqueeze(1),dst,0,dim_size=N,reduce='sum')
    den=scatter(w,dst,0,dim_size=N,reduce='sum').clamp(min=1e-6)
    o=num/den.unsqueeze(1); return o if vec.dim()>1 else o.squeeze(1)

# ---- exposure features (neighbour restated_now per relation, t & t-1) + own event history ----
rnt=[torch.from_numpy(rn[t]) for t in range(Tall)]
EXP=np.zeros((Tall,N,len(ALL)*2),np.float32)
for t in range(Tall):
    for ci,c in enumerate(ALL):
        ei,w=adj(t,c); EXP[t,:,ci]=mean_agg(rnt[t],ei,w).numpy()
        tl=max(t-1,0); eil,wl=adj(tl,c); EXP[t,:,len(ALL)+ci]=mean_agg(rnt[tl],eil,wl).numpy()
OWN=np.zeros((Tall,N,2),np.float32)
for t in range(Tall): OWN[t,:,0]=rn[t]; OWN[t,:,1]=rn[max(t-1,0)]

Xz_t=torch.from_numpy(Xz); XL_t=torch.from_numpy(XL)
OWN_t=torch.from_numpy(OWN); EXP_t=torch.from_numpy(EXP); lab_t=torch.from_numpy(label)
A=[{c:adj(t,c) for c in ALL} for t in range(Tall)]

def metr(y,p):
    o=np.argsort(-p)
    return {"roc":round(float(roc_auc_score(y,p)),4),"pr":round(float(average_precision_score(y,p)),4),
            "recall@10%":round(float(y[o[:max(1,len(p)//10)]].sum()/max(y.sum(),1)),4)}

class ACEpure(nn.Module):
    """Graph-only ACE: relation-gated attention + monotone auditor hierarchy + collective inference.
    No GBDT-leaf embedding, no tabular fusion."""
    def __init__(s,H=96,layers=2,collective_steps=5,drop=0.3,use_feat=False,
                 attn=True,monotone=True):
        super().__init__(); s.H=H; s.cs=collective_steps; s.use_feat=use_feat
        s.attn_on=attn; s.monotone=monotone
        fin=(Fdim*2 if use_feat else 0)+2+len(ALL)*2
        s.featenc=nn.Sequential(nn.Linear(fin,H),nn.ReLU())
        s.lvl=nn.Parameter(torch.zeros(len(AUD)))
        s.nchan=1+len(OTH)
        s.asrc=nn.ModuleList([nn.Linear(H,1) for _ in range(s.nchan)])
        s.adst=nn.ModuleList([nn.Linear(H,1) for _ in range(s.nchan)])
        s.lin=nn.ModuleList([nn.Linear(H,H) for _ in range(s.nchan)])
        s.gate=nn.Linear(H,s.nchan)
        s.upd=nn.ModuleList([nn.Linear(2*H,H) for _ in range(layers)]); s.layers=layers
        s.drop=nn.Dropout(drop)
        s.head=nn.Sequential(nn.Linear(2*H,H),nn.ReLU(),nn.Dropout(drop),nn.Linear(H,1))
        s.beta=nn.Parameter(torch.tensor(0.0))
    def lvlw(s):
        if not s.monotone:                       # free (non-monotone) ablation
            w=F.softplus(s.lvl); return w/w.max().clamp(min=1e-6)
        sp=F.softplus(s.lvl); c=torch.cumsum(sp.flip(0),0).flip(0); return c/c.max().clamp(min=1e-6)
    def msg(s,k,h,ei,w):
        if ei.size(1)==0: return torch.zeros_like(h)
        src,dst=ei; hs=s.lin[k](h)
        if not s.attn_on:                        # mean-aggregation ablation
            return scatter(hs[src]*w.unsqueeze(1),dst,0,dim_size=N,reduce='sum') / \
                   scatter(w,dst,0,dim_size=N,reduce='sum').clamp(min=1e-6).unsqueeze(1)
        e=F.leaky_relu(s.asrc[k](hs)[src]+s.adst[k](hs)[dst]).squeeze(-1)+torch.log(w.clamp(min=1e-6))
        al=gsoftmax(e,dst,num_nodes=N)
        return scatter(al.unsqueeze(1)*hs[src],dst,0,dim_size=N,reduce='sum')
    def aud_msg(s,h,t):
        w=s.lvlw(); m=0
        for li,c in enumerate(AUD):
            ei,ew=A[t][c]; m=m+w[li]*s.msg(0,h,ei,ew)
        return m
    def forward(s,t):
        inp=([Xz_t[t],XL_t[t]] if s.use_feat else [])+[OWN_t[t],EXP_t[t]]
        h0=F.relu(s.featenc(torch.cat(inp,1))); h=h0
        for l in range(s.layers):
            chans=[s.aud_msg(h,t)]+[s.msg(1+j,h,*A[t][c]) for j,c in enumerate(OTH)]
            g=torch.softmax(s.gate(h),1)
            agg=sum(g[:,k:k+1]*chans[k] for k in range(s.nchan))
            h=F.relu(s.upd[l](torch.cat([h,agg],1))); h=s.drop(h)
        y=s.head(torch.cat([h0,h],1)).squeeze(1)
        # collective inference: propagate predicted logit over auditor hierarchy (APPNP-style)
        w=s.lvlw(); b=torch.sigmoid(s.beta)*0.6; yk=y
        for _ in range(s.cs):
            m=0
            for li,c in enumerate(AUD): m=m+w[li]*mean_agg(yk,*A[t][c])
            wsum=float(w.detach().sum()); yk=(1-b)*y+b*(m/max(wsum,1e-6))
        return y, yk

spw=None
def train_ace(seed=0,**kw):
    global spw
    torch.manual_seed(seed); m=ACEpure(**kw)
    opt=torch.optim.Adam(m.parameters(),lr=3e-3,weight_decay=2e-4)
    ytr=np.concatenate([label[t][active[t]&(label[t]>=0)] for t in TR])
    spw=float((ytr==0).sum()/max((ytr==1).sum(),1)); POSW=torch.tensor([spw])
    tri={t:torch.from_numpy(np.where(active[t]&(label[t]>=0))[0]).long() for t in TR}
    vai={t:np.where(active[t]&(label[t]>=0))[0] for t in VA}; best=-1; bs=None; bad=0
    for ep in range(350):
        m.train(); opt.zero_grad(); loss=0
        for t in TR:
            y,yk=m(t); idx=tri[t]
            loss=loss+F.binary_cross_entropy_with_logits(yk[idx],lab_t[t][idx].float(),pos_weight=POSW) \
                     +0.3*F.binary_cross_entropy_with_logits(y[idx],lab_t[t][idx].float(),pos_weight=POSW)
        loss.backward(); opt.step()
        if ep%5==0:
            m.eval(); ys=[];ps=[]
            with torch.no_grad():
                for t in VA: _,yk=m(t); a=vai[t]; ps.append(torch.sigmoid(yk[a]).numpy()); ys.append(label[t][a])
            vp=average_precision_score(np.concatenate(ys),np.concatenate(ps))
            if vp>best: best=vp; bs={k:v.clone() for k,v in m.state_dict().items()}; bad=0
            else: bad+=1
            if bad>12: break
    if bs: m.load_state_dict(bs)
    m.eval(); return m,best

def collect(m,ts):
    ys=[];ps=[]
    with torch.no_grad():
        for t in ts:
            _,yk=m(t); a=np.where(active[t]&(label[t]>=0))[0]
            ps.append(torch.sigmoid(yk[a]).numpy()); ys.append(label[t][a])
    return np.concatenate(ys),np.concatenate(ps)

def bag(nseed,tag="ace-pure",**kw):
    te=[];va=[];lw=[]
    for sd in range(nseed):
        m,bv=train_ace(seed=sd,**kw)
        y_te,p_te=collect(m,TE); y_va,p_va=collect(m,VA)
        te.append(p_te); va.append(p_va); lw.append([float(m.lvlw()[i]) for i in range(len(AUD))])
        print(f"  [{tag} s{sd}] vp={bv:.4f} roc={metr(y_te,p_te)['roc']} pr={metr(y_te,p_te)['pr']}",file=sys.stderr)
    return np.mean(te,0),np.mean(va,0),y_te,y_va,np.mean(lw,0)

t0=time.time()
RES={"mode":MODE,"label":LABEL_MODE,"test_years":[YEARS[i] for i in TE],
     "model":"ACE-GNN (pure, graph-only)" if not USEFEAT else "ACE-GNN (pure, +financials)",
     "nseed":NSEED,"results":{}}

# ---- headline pure model ----
p_te,p_va,y_te,y_va,lw=bag(NSEED,tag="ACE-pure",use_feat=USEFEAT,attn=True,monotone=True)
RES["results"]["ACE-GNN(pure)"]=metr(y_te,p_te)
RES["results"]["ACE-GNN(pure)"]["aud_level_weights"]={c:round(float(lw[i]),3) for i,c in enumerate(AUD)}
np.savez(OUT.replace('.json','_preds.npz'),y=y_te,p=p_te)

# ---- optional graph-only ablation (component drop), fewer seeds for speed ----
if ABLATE:
    na=max(3,NSEED//2)
    for tag,kw in [("-collective(K=0)",dict(collective_steps=0)),
                   ("-attention(mean)",dict(attn=False)),
                   ("-monotone(free)",dict(monotone=False)),
                   ("+financials",dict(use_feat=True))]:
        pt,_,yt,_,_=bag(na,tag=tag,use_feat=kw.get("use_feat",USEFEAT),
                        attn=kw.get("attn",True),monotone=kw.get("monotone",True),
                        collective_steps=kw.get("collective_steps",5))
        RES["results"][tag]=metr(yt,pt)
    # relation drop-one would require masking A; left to ablation.py (graph-only base)

RES["runtime_sec"]=round(time.time()-t0,1)
os.makedirs(os.path.dirname(OUT),exist_ok=True)
json.dump(RES,open(OUT,"w"),indent=2,ensure_ascii=False)
print(f"\n=== ACE-pure {MODE}/{LABEL_MODE} test {[YEARS[i] for i in TE]} ===")
for k,v in RES["results"].items():
    print(f"{k:22s} ROC={v.get('roc')} AUPRC={v.get('pr')} r@10%={v.get('recall@10%')}")
print(f"[done {RES['runtime_sec']}s -> {OUT}]")
