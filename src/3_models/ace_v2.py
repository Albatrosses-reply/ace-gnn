#!/usr/bin/env python3
"""ACE-v2 — ground-up redesign of the pure model around what THIS data rewards.

Evidence driving the design (all from saved runs in this repo):
  (1) PLE-encoded financials alone (PLEG stage-A, no graph)  -> PR 0.485   [tabular-neural, strong]
  (2) exposure+MP+collective alone (ace_pure, no financials) -> PR 0.471   [contagion, strong]
  (3) raw-z financials added to (2)                          -> PR 0.432   [encoding artifact, hurts]
  (4) RF on the SAME info set [Xz,OWN(,EXP)]                 -> PR 0.496 / 0.506 [the bar to beat]
=> The two signal sources are complementary but were never combined in ONE pure network.
ACE-v2 = PLE numerical embeddings (tree-like thresholds, Gorishniy) + own event history + explicit
multi-relation exposure features + BatchEnsemble (TabM-style) residual MLP encoder + optional
graph-residual attention message passing + collective inference (APPNP on the predicted logit over
the auditor hierarchy). 100%% neural, end-to-end, no trees anywhere.

HONEST PROTOCOL: variants are screened on VALIDATION AUPRC only; test is reported for transparency
but selection never uses it. JSON records val+test for every variant.

ENV: MODE=recent|panel LABEL=severe|label TAG=name NSEED=2 THREADS=3
     USE_EXP=1 USE_EXP2=0 COLLECTIVE=1 HIER=free|mono MP=0
     NB=32 FEMB=12 TABM=8 H=192 ENC_DEPTH=2 EPMAX=400 LR=1e-3 DROP=0.15 POSW_SQRT=1
Run from repo root:  TAG=C_col MODE=recent LABEL=severe python3 src/3_models/ace_v2.py
"""
import json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.utils import scatter, softmax as gsoftmax
from sklearn.metrics import roc_auc_score, average_precision_score

torch.set_num_threads(int(os.environ.get("THREADS","3")))
torch.manual_seed(0); np.random.seed(0)
MODE=os.environ.get("MODE","recent"); LABEL=os.environ.get("LABEL","severe")
TAG=os.environ.get("TAG","v2"); NSEED=int(os.environ.get("NSEED","2"))
USE_EXP=int(os.environ.get("USE_EXP","1")); USE_EXP2=int(os.environ.get("USE_EXP2","0"))
COLLECTIVE=int(os.environ.get("COLLECTIVE","1")); HIER=os.environ.get("HIER","free"); MP=int(os.environ.get("MP","0"))
NB=int(os.environ.get("NB","32")); FEMB=int(os.environ.get("FEMB","12")); TABM=int(os.environ.get("TABM","8"))
H=int(os.environ.get("H","192")); ENC_DEPTH=int(os.environ.get("ENC_DEPTH","2")); EPMAX=int(os.environ.get("EPMAX","400"))
LR=float(os.environ.get("LR","1e-3")); DROP=float(os.environ.get("DROP","0.15")); POSW_SQRT=int(os.environ.get("POSW_SQRT","1"))
KPROP=int(os.environ.get("KPROP","5"))
# v3 levers (defaults reproduce FINAL_L exactly):
BINS=os.environ.get("BINS","quantile")   # quantile | tree  (target-aware tree bins, Gorishniy et al.)
LAGS=int(os.environ.get("LAGS","2"))     # own-history / exposure lags (2 = t, t-1)
MTL=int(os.environ.get("MTL","0"))       # multi-task aux heads on the other two labels
RANKW=float(os.environ.get("RANKW","0")) # pairwise rank-loss weight mixed with BCE
WD=float(os.environ.get("WD","3e-5"))    # weight decay (variance reduction for noisy labels)
LS=float(os.environ.get("LS","0"))       # label smoothing toward train prior
AUXW=float(os.environ.get("AUXW","0.3")) # MTL aux-loss weight (1.0 anchors encoder on clean labels)
GCE=float(os.environ.get("GCE","0"))     # generalized cross-entropy q (0=off; ~0.7 = noise-robust, MAE-like)
MPKIND=os.environ.get("MPKIND","attn")   # attn | ltq  (learned-threshold quantile aggregation of neighbour hidden states)
LTQ_DP=int(os.environ.get("LTQ_DP","8")); LTQ_K=int(os.environ.get("LTQ_K","8"))
OUT=os.environ.get("OUT",f"data/ext/pure/v2/{TAG}_{MODE}_{LABEL}.json")
LKEY={"adverse":"label_adverse","severe":"label_severe"}.get(LABEL,"label")
RKEY={"adverse":"restated_now_adverse","severe":"restated_now_severe"}.get(LABEL,"restated_now")

GRAPH=os.environ.get("GRAPH","data/ext/graph.pt")  # variant graphs for sensitivity (e.g. graph_own3.pt)
G=torch.load(GRAPH,weights_only=False); YEARS=G["years"]; yidx={y:i for i,y in enumerate(YEARS)}
X=G["X"].numpy().copy(); active=G["active"].numpy(); label=G[LKEY].numpy().copy(); rn=G[RKEY].numpy().astype(np.float32)
AUXKEYS=[k for k in ["label","label_adverse","label_severe"] if k!=LKEY] if MTL else []
auxlab={k:G[k].numpy().copy() for k in AUXKEYS}
Tall,N,Fdim=X.shape; snaps=G["snapshots"]
AUD=["partner","office","auditor"] if MODE=="recent" else ["office","auditor"]
OTH=["board","ownership"]; ALL=AUD+OTH
if MODE=="panel": USE=[yidx[y] for y in range(2005,2020)]; TR=USE[:11]; VA=[USE[11]]; TE=USE[12:15]
else: USE=[yidx[y] for y in range(2017,2023)]; TR=USE[:3]; VA=[USE[3]]; TE=USE[4:6]
print(f"[ACE-v2:{TAG}] MODE={MODE} label={LABEL} exp={USE_EXP} exp2={USE_EXP2} col={COLLECTIVE} hier={HIER} mp={MP} "
      f"tabm={TABM} H={H} depth={ENC_DEPTH} nseed={NSEED} test={[YEARS[i] for i in TE]}",file=sys.stderr)
# PLACEBO leakage check: permute TRAIN+VALIDATION labels (everything the model fits/selects on);
# features (incl. exposures, PLE knots) and TEST labels untouched. No leakage => test ROC ~ 0.5.
if os.environ.get("PLACEBO"):
    _prng=np.random.default_rng(0)
    for _t in TR+VA:
        _a=np.where(active[_t]&(label[_t]>=0))[0]; label[_t,_a]=_prng.permutation(label[_t,_a])
        for _k in AUXKEYS:
            _a2=np.where(active[_t]&(auxlab[_k][_t]>=0))[0]; auxlab[_k][_t,_a2]=_prng.permutation(auxlab[_k][_t,_a2])
    print("[PLACEBO] train+val labels permuted (features/test untouched)",file=sys.stderr)

# ---- train-stat winsorize/impute/zscore (no leakage) ----
tm=np.zeros((Tall,N),bool)
for t in TR: tm[t]=active[t]&(label[t]>=0)
Xtr0=X[tm]; lo=np.nanpercentile(Xtr0,1,0); hi=np.nanpercentile(Xtr0,99,0); med=np.nanmedian(Xtr0,0)
Xc=np.clip(X,lo,hi); ix=np.where(np.isnan(Xc)); Xc[ix]=np.take(med,ix[2])
mu=np.nanmean(np.clip(Xtr0,lo,hi),0); sd=np.nanstd(np.clip(Xtr0,lo,hi),0); sd[sd<1e-6]=1
Xz=((Xc-mu)/sd).astype(np.float32)

# ---- PLE: piecewise-linear ramps, bins from TRAIN years only ----
# BINS=quantile: marginal quantile knots. BINS=tree: target-aware knots = decision-tree split
# points fitted per feature on TRAIN (the thresholds GBDTs actually exploit; Gorishniy et al.).
ytr_pool=np.concatenate([label[t][active[t]&(label[t]>=0)] for t in TR])
def _tree_knots(xcol,y,nb,lo=-1e9,hi=1e9):
    from sklearn.tree import DecisionTreeClassifier
    dt=DecisionTreeClassifier(max_leaf_nodes=nb,min_samples_leaf=max(64,len(y)//(nb*4)),random_state=0).fit(xcol.reshape(-1,1),y)
    thr=sorted(set(float(v) for f,v in zip(dt.tree_.feature,dt.tree_.threshold) if f!=-2))
    need=nb-1
    if len(thr)<need:                                  # supplement with marginal quantiles
        for q in np.quantile(xcol,np.linspace(0,1,need+2)[1:-1]):
            if len(thr)>=need: break
            if all(abs(q-t0)>1e-9 for t0 in thr): thr.append(float(q))
    thr=sorted(thr)[:need]
    while len(thr)<need:                               # discrete columns: pad with degenerate
        thr.append((thr[-1] if thr else 0.0)+1e-6*(len(thr)+1))   # zero-width filler knots
    return np.maximum.accumulate(np.array([lo]+sorted(thr)+[hi],dtype=np.float32))
Xtr_flat=Xz[tm]; knots=[]
for j in range(Fdim):
    if BINS=="tree": knots.append(_tree_knots(Xtr_flat[:,j],ytr_pool,NB))
    else:
        q=np.quantile(Xtr_flat[:,j],np.linspace(0,1,NB+1)); q[0]=-1e9; q[-1]=1e9
        knots.append(np.maximum.accumulate(q).astype(np.float32))
def ple(Xs):
    parts=[]
    for j in range(Fdim):
        kb=knots[j]; x=Xs[:,j:j+1]
        parts.append(np.clip((x-kb[:-1][None,:])/(kb[1:][None,:]-kb[:-1][None,:]+1e-9),0.,1.).astype(np.float32))
    return np.concatenate(parts,1)
PLE_t={t:torch.from_numpy(ple(Xz[t])) for t in USE}

def adj(t,c): ei,w=snaps[t][c]; return ei,w
def mean_agg(vec,ei,w):
    if ei.size(1)==0: return torch.zeros_like(vec)
    s,d=ei; v=vec if vec.dim()>1 else vec.unsqueeze(1)
    num=scatter(v[s]*w.unsqueeze(1),d,0,dim_size=N,reduce='sum'); den=scatter(w,d,0,dim_size=N,reduce='sum').clamp(min=1e-6)
    o=num/den.unsqueeze(1); return o if vec.dim()>1 else o.squeeze(1)
def max_agg(vec,ei,w):
    if ei.size(1)==0: return torch.zeros_like(vec)
    s,d=ei; out=scatter(vec[s],d,0,dim_size=N,reduce='max')
    deg=scatter(torch.ones_like(w),d,0,dim_size=N,reduce='sum'); return out*(deg>0).float()

# ---- explicit contagion features (LAGS lags; LAGS=2 reproduces the t,t-1 layout exactly) ----
OWN=np.zeros((Tall,N,LAGS),np.float32)
for t in range(Tall):
    for l in range(LAGS): OWN[t,:,l]=rn[max(t-l,0)]
rnt=[torch.from_numpy(rn[t]) for t in range(Tall)]
EXP=np.zeros((Tall,N,len(ALL)*LAGS),np.float32)     # 1-hop mean exposure at lags 0..LAGS-1
for t in range(Tall):
    for l in range(LAGS):
        tl=max(t-l,0)
        for ci,c in enumerate(ALL):
            ei,w=adj(tl,c); EXP[t,:,l*len(ALL)+ci]=mean_agg(rnt[tl],ei,w).numpy()
EXP2=np.zeros((Tall,N,len(AUD)*2),np.float32)       # auditor rels: 2-hop mean + 1-hop MAX exposure
for t in range(Tall):
    for ci,c in enumerate(AUD):
        ei,w=adj(t,c); e1=mean_agg(rnt[t],ei,w)
        EXP2[t,:,ci]=mean_agg(e1,ei,w).numpy()                       # 2-hop (mean of mean)
        EXP2[t,:,len(AUD)+ci]=max_agg(rnt[t],ei,w).numpy()           # any risky peer
OWN_t=torch.from_numpy(OWN); EXP_t=torch.from_numpy(EXP); EXP2_t=torch.from_numpy(EXP2)
lab_t=torch.from_numpy(label)
A=[{c:adj(t,c) for c in ALL} for t in range(Tall)]

# ---- EXPENC: exposure features are zero-inflated; raw values hurt an MLP. Encode each column as
# [1(x>0), PLE ramps over POSITIVE-part train quantiles] so the net gets tree-like thresholds. ----
EXPENC=int(os.environ.get("EXPENC","0")); NEB=int(os.environ.get("NEB","8"))
def fit_expenc(M):
    ks=[]
    for j in range(M.shape[2]):
        v=M[:,:,j][tm]; vp=v[v>1e-9]
        if BINS=="tree" and len(vp)>=200:
            ks.append(_tree_knots(v,ytr_pool,NEB,lo=1e-9,hi=max(float(vp.max()),1.0)+1e-6))
        elif len(vp)<50: ks.append(np.maximum.accumulate(np.linspace(1e-9,1.0,NEB+1)).astype(np.float32))
        else:
            q=np.quantile(vp,np.linspace(0,1,NEB+1)); q[0]=1e-9; q[-1]=max(float(q[-1]),1.0)+1e-6
            ks.append(np.maximum.accumulate(q).astype(np.float32))
    return ks
def apply_expenc(E,ks):
    parts=[]
    for j in range(E.shape[1]):
        x=E[:,j:j+1]; kb=ks[j]; flag=(x>1e-9).astype(np.float32)
        ramps=np.clip((x-kb[:-1][None,:])/(kb[1:][None,:]-kb[:-1][None,:]+1e-9),0.,1.).astype(np.float32)
        parts.append(np.concatenate([flag,ramps*flag],1))
    return np.concatenate(parts,1)
EXPE_t={}; EXP2E_t={}
if EXPENC:
    eknots=fit_expenc(EXP)
    EXPE_t={t:torch.from_numpy(apply_expenc(EXP[t],eknots)) for t in USE}
    if USE_EXP2:
        e2knots=fit_expenc(EXP2)
        EXP2E_t={t:torch.from_numpy(apply_expenc(EXP2[t],e2knots)) for t in USE}

ytr=np.concatenate([label[t][active[t]&(label[t]>=0)] for t in TR]); spw=float((ytr==0).sum()/max((ytr==1).sum(),1))
POSW=torch.tensor([float(np.sqrt(spw)) if POSW_SQRT else spw])
PRIOR=float((ytr==1).mean())             # train positive rate (label-smoothing target)
def gce_loss(logits,y01,q):              # generalized cross-entropy (Zhang & Sabuncu '18): robust to label noise
    p=torch.sigmoid(logits); pt=torch.where(y01>0.5,p,1-p).clamp(min=1e-6)
    w=torch.where(y01>0.5,POSW.to(p.dtype),torch.ones_like(p))
    return ((1-pt.pow(q))/q*w).sum()/w.sum().clamp(min=1e-6)
auxlab_t={k:torch.from_numpy(auxlab[k]) for k in AUXKEYS}; AUXPOSW={}; AUXIDX={}
for k in AUXKEYS:
    ya=np.concatenate([auxlab[k][t][active[t]&(auxlab[k][t]>=0)] for t in TR])
    sw=float((ya==0).sum()/max((ya==1).sum(),1))
    AUXPOSW[k]=torch.tensor([float(np.sqrt(sw)) if POSW_SQRT else sw])
    AUXIDX[k]={t:torch.from_numpy(np.where(active[t]&(auxlab[k][t]>=0))[0]).long() for t in TR}
def metr(y,p):
    o=np.argsort(-p)
    return {"roc":round(float(roc_auc_score(y,p)),4),"pr":round(float(average_precision_score(y,p)),4),
            "recall@10%":round(float(y[o[:max(1,len(p)//10)]].sum()/max(y.sum(),1)),4)}

expdim=(EXP.shape[2]*(1+NEB)) if EXPENC else EXP.shape[2]
exp2dim=(EXP2.shape[2]*(1+NEB)) if EXPENC else EXP2.shape[2]
aux_dim=LAGS+(expdim if USE_EXP else 0)+(exp2dim if USE_EXP2 else 0)
K_ENS=max(TABM,1)
class BELinear(nn.Module):                            # BatchEnsemble: shared W + per-member rank-1 adapters
    def __init__(s,din,dout):
        super().__init__(); s.W=nn.Linear(din,dout)
        s.r=nn.Parameter(torch.ones(K_ENS,din)+0.05*torch.randn(K_ENS,din))
        s.sd=nn.Parameter(torch.ones(K_ENS,dout)+0.05*torch.randn(K_ENS,dout))
        s.bm=nn.Parameter(torch.zeros(K_ENS,dout))
    def forward(s,x): return s.W(x*s.r.unsqueeze(1))*s.sd.unsqueeze(1)+s.bm.unsqueeze(1)

class ACEv2(nn.Module):
    def __init__(s):
        super().__init__()
        s.femb=nn.Linear(NB,FEMB)                     # shared per-feature numerical embedding over PLE ramps
        din=Fdim*FEMB+aux_dim
        s.inp=BELinear(din,H)
        s.ln=nn.ModuleList([nn.LayerNorm(H) for _ in range(ENC_DEPTH)])
        s.f1=nn.ModuleList([BELinear(H,2*H) for _ in range(ENC_DEPTH)])
        s.f2=nn.ModuleList([BELinear(2*H,H) for _ in range(ENC_DEPTH)])
        s.hln=nn.LayerNorm(H); s.head=BELinear(H,1)
        s.lvl=nn.Parameter(torch.zeros(len(AUD)))     # auditor-level weights for collective/MP
        s.beta=nn.Parameter(torch.tensor(0.0))
        if MP:                                        # graph-residual attention layer on pooled hidden
            s.nchan=1+len(OTH)
            s.asrc=nn.ModuleList([nn.Linear(H,1) for _ in range(s.nchan)])
            s.adst=nn.ModuleList([nn.Linear(H,1) for _ in range(s.nchan)])
            s.lin=nn.ModuleList([nn.Linear(H,H) for _ in range(s.nchan)])
            s.gate=nn.Linear(H,s.nchan)
            s.gh=nn.Sequential(nn.LayerNorm(H),nn.Linear(H,1))
            s.gamma=nn.Parameter(torch.tensor(0.2))
            if MPKIND=="ltq":                          # learned-threshold quantile aggregation over neighbour hidden states
                s.ltqproj=nn.ModuleList([nn.Linear(H,LTQ_DP) for _ in range(s.nchan)])
                s.ltqtau=nn.Parameter(torch.linspace(-2,2,LTQ_K).repeat(s.nchan,LTQ_DP,1)+0.01*torch.randn(s.nchan,LTQ_DP,LTQ_K))
                s.ltqlogs=nn.Parameter(torch.zeros(s.nchan,LTQ_DP,LTQ_K))
                s.ltqout=nn.ModuleList([nn.Linear(LTQ_DP*LTQ_K,H) for _ in range(s.nchan)])
        s.drop=nn.Dropout(DROP)
        if MTL: s.auxheads=nn.ModuleList([nn.Linear(H,1) for _ in AUXKEYS])  # created LAST: default RNG path unchanged
    def lvlw(s):
        if HIER=="mono":
            sp=F.softplus(s.lvl); c=torch.cumsum(sp.flip(0),0).flip(0); return c/c.max().clamp(min=1e-6)
        w=F.softplus(s.lvl); return w/w.max().clamp(min=1e-6)
    def feat(s,t):
        p=s.femb(PLE_t[t].view(N,Fdim,NB)).reshape(N,Fdim*FEMB)
        cols=[p,OWN_t[t]]+([EXPE_t[t] if EXPENC else EXP_t[t]] if USE_EXP else [])+([EXP2E_t[t] if EXPENC else EXP2_t[t]] if USE_EXP2 else [])
        return torch.cat(cols,1)
    def attn(s,k,h,ei,w):
        if ei.size(1)==0: return torch.zeros_like(h)
        src,dst=ei; hs=s.lin[k](h)
        e=F.leaky_relu(s.asrc[k](hs)[src]+s.adst[k](hs)[dst]).squeeze(-1)+torch.log(w.clamp(min=1e-6))
        al=gsoftmax(e,dst,num_nodes=N)
        return scatter(al.unsqueeze(1)*hs[src],dst,0,dim_size=N,reduce='sum')
    def ltq_agg(s,k,h,ei,w):                          # soft fraction of neighbours above learned thresholds, per probe
        if ei.size(1)==0: return torch.zeros_like(h)
        src,dst=ei; ms=s.ltqproj[k](h)[src]           # E x Dp
        tau=s.ltqtau[k].unsqueeze(0); sc=F.softplus(s.ltqlogs[k]).unsqueeze(0).clamp(min=1e-3)
        ind=torch.sigmoid((ms.unsqueeze(-1)-tau)/sc).reshape(ms.size(0),-1)   # E x Dp*K
        num=scatter(ind*w.unsqueeze(1),dst,0,dim_size=N,reduce='sum')
        den=scatter(w,dst,0,dim_size=N,reduce='sum').clamp(min=1e-6).unsqueeze(1)
        return s.ltqout[k](num/den)
    def forward(s,t):
        x=s.feat(t).unsqueeze(0).expand(K_ENS,N,-1)
        h=s.inp(x)
        for l in range(ENC_DEPTH):
            h=h+s.f2[l](s.drop(F.gelu(s.f1[l](s.ln[l](h)))))
        z=s.head(s.hln(h)).squeeze(-1).mean(0)        # tabular-neural logit
        hm=s.hln(h).mean(0)
        if MTL: s._hm=hm                              # pooled hidden for aux heads
        if MP:                                        # graph residual on pooled hidden state
            w=s.lvlw(); audm=0; agg_fn=s.ltq_agg if MPKIND=="ltq" else s.attn
            for li,c in enumerate(AUD):
                ei,ew=A[t][c]; audm=audm+w[li]*agg_fn(0,hm,ei,ew)
            chans=[audm]+[agg_fn(1+j,hm,*A[t][c]) for j,c in enumerate(OTH)]
            g=torch.softmax(s.gate(hm),1)
            agg=sum(g[:,k:k+1]*chans[k] for k in range(s.nchan))
            z=z+s.gamma*s.gh(agg).squeeze(1)
        if not COLLECTIVE: return z,z
        w=s.lvlw(); b=torch.sigmoid(s.beta)*0.6; yk=z
        for _ in range(KPROP):
            m=0
            for li,c in enumerate(AUD): m=m+w[li]*mean_agg(yk,*A[t][c])
            yk=(1-b)*z+b*(m/w.sum().clamp(min=1e-6))
        return z,yk
    def aux_logits(s): return [hd(s._hm).squeeze(1) for hd in s.auxheads]

def collect(m,ts):
    ys=[];ps=[]
    with torch.no_grad():
        for t in ts:
            _,yk=m(t); a=np.where(active[t]&(label[t]>=0))[0]
            ps.append(torch.sigmoid(yk[a]).numpy()); ys.append(label[t][a])
    return np.concatenate(ys),np.concatenate(ps)

def train_one(seed):
    torch.manual_seed(seed); m=ACEv2()
    opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=WD)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPMAX)
    tri={t:torch.from_numpy(np.where(active[t]&(label[t]>=0))[0]).long() for t in TR}
    best=-1; bs=None; bad=0
    for ep in range(EPMAX):
        m.train(); opt.zero_grad(); loss=0
        for t in TR:
            z,yk=m(t); idx=tri[t]; y01=lab_t[t][idx].float(); yt=y01
            if LS>0: yt=yt*(1-LS)+LS*PRIOR            # smooth toward train prior (noisy-label setting)
            if GCE>0: loss=loss+gce_loss(yk[idx],y01,GCE)   # noise-robust main loss (overrides BCE)
            else: loss=loss+F.binary_cross_entropy_with_logits(yk[idx],yt,pos_weight=POSW)
            if COLLECTIVE: loss=loss+0.3*F.binary_cross_entropy_with_logits(z[idx],yt,pos_weight=POSW)
            if MTL:
                als=m.aux_logits()
                for ai,k in enumerate(AUXKEYS):
                    ia=AUXIDX[k][t]
                    if len(ia): loss=loss+AUXW*F.binary_cross_entropy_with_logits(als[ai][ia],auxlab_t[k][t][ia].float(),pos_weight=AUXPOSW[k])
            if RANKW>0:
                yi=lab_t[t][idx]; pos=idx[yi==1]; neg=idx[yi==0]
                if len(pos)>0 and len(neg)>0:
                    npair=min(len(pos)*8,40000)
                    pi=pos[torch.randint(len(pos),(npair,))]; nj=neg[torch.randint(len(neg),(npair,))]
                    loss=loss+RANKW*F.softplus(yk[nj]-yk[pi]).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5.0); opt.step(); sched.step()
        if ep%5==0:
            m.eval(); yv,pv=collect(m,VA); vp=average_precision_score(yv,pv)
            if vp>best+1e-4: best=vp; bs={k:v.clone() for k,v in m.state_dict().items()}; bad=0
            else: bad+=1
            if bad>12: break
    if bs: m.load_state_dict(bs)
    m.eval(); return m,best

t0=time.time(); pte=[];pva=[];seed_rows=[];lw_acc=[]
for sd in range(NSEED):
    m,bv=train_one(sd)
    y_te,p_te=collect(m,TE); y_va,p_va=collect(m,VA)
    pte.append(p_te); pva.append(p_va)
    seed_rows.append({"seed":sd,"val_pr":round(float(bv),4),"test":metr(y_te,p_te)})
    lw_acc.append([float(m.lvlw()[i]) for i in range(len(AUD))])
    print(f"  [{TAG} s{sd}] val_pr={bv:.4f} test_roc={seed_rows[-1]['test']['roc']} test_pr={seed_rows[-1]['test']['pr']}",file=sys.stderr)
pte_b=np.mean(pte,0); pva_b=np.mean(pva,0)
RES={"tag":TAG,"mode":MODE,"label":LABEL,"test_years":[YEARS[i] for i in TE],
     "knobs":{"USE_EXP":USE_EXP,"USE_EXP2":USE_EXP2,"COLLECTIVE":COLLECTIVE,"HIER":HIER,"MP":MP,
              "NB":NB,"FEMB":FEMB,"TABM":TABM,"H":H,"ENC_DEPTH":ENC_DEPTH,"NSEED":NSEED,"EPMAX":EPMAX,
              "LR":LR,"DROP":DROP,"POSW_SQRT":POSW_SQRT,"KPROP":KPROP,
              "BINS":BINS,"LAGS":LAGS,"MTL":MTL,"RANKW":RANKW,"EXPENC":EXPENC,"NEB":NEB,
              "WD":WD,"LS":LS,"AUXW":AUXW},
     "val_bag":metr(y_va,pva_b),"test_bag":metr(y_te,pte_b),
     "per_seed":seed_rows,"aud_level_weights":{c:round(float(np.mean([l[i] for l in lw_acc])),3) for i,c in enumerate(AUD)},
     "runtime_sec":round(time.time()-t0,1)}
os.makedirs(os.path.dirname(OUT),exist_ok=True)
json.dump(RES,open(OUT,"w"),indent=2,ensure_ascii=False)
np.savez(OUT.replace(".json","_preds.npz"),y=y_te,p=pte_b,yv=y_va,pv=pva_b)
print(f"\n=== ACE-v2[{TAG}] {MODE}/{LABEL} ===")
print(f"VAL  bag: {RES['val_bag']}")
print(f"TEST bag: {RES['test_bag']}")
print(f"[done {RES['runtime_sec']}s -> {OUT}]")
