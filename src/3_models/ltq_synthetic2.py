#!/usr/bin/env python3
"""LTQ-MP DECISIVE representational test (deterministic regression, no label noise).

Pre-registered bar (set BEFORE running): LTQ must beat PNA by >=0.05 test R^2 on BOTH
threshold-dependent targets (count_above_c, frac_above_tail). Otherwise the mechanism is
NOT validated and we abandon it.

Fairness: ALL aggregators get degree (PNA degree-scalers; others get log-deg appended; LTQ gets
fraction AND count = fraction*deg). Same head capacity. Same training. Only pooling differs.
Targets are exact functions of the neighbour-signal multiset -> tests representational capacity.
Run:  python3 src/3_models/ltq_synthetic2.py
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.utils import scatter

N=5000; AVGDEG=20; D=8; K=10; H=64; EPOCHS=400; SEEDS=[0,1,2]
P_INFLATE=0.30; C=1.0; C_TAIL=1.7      # denser positive signal so counts/quantiles have variance
TARGETS=["mean","sum","count_c","frac_tail"]

def make_graph(seed):
    rng=np.random.default_rng(seed); M=N*AVGDEG//2
    s=rng.integers(0,N,M); d=rng.integers(0,N,M); m=s!=d; s,d=s[m],d[m]
    src=np.concatenate([s,d]); dst=np.concatenate([d,s])       # symmetric, NO self-loop (target is over true neighbours)
    return torch.tensor(np.stack([src,dst]),dtype=torch.long)

def make(ei,seed):
    rng=np.random.default_rng(seed+1000)
    sig=((rng.random(N)<P_INFLATE)*np.abs(rng.normal(0,1.2,N))).astype(np.float32)
    src,dst=ei.numpy()
    from collections import defaultdict
    nb=defaultdict(list)
    for a,b in zip(src,dst): nb[b].append(sig[a])
    tg={t:np.zeros(N,np.float32) for t in TARGETS}
    for v in range(N):
        vals=np.array(nb[v]) if nb[v] else np.array([0.0],np.float32)
        tg["mean"][v]=vals.mean(); tg["sum"][v]=vals.sum()
        tg["count_c"][v]=float((vals>C).sum()); tg["frac_tail"][v]=float((vals>C_TAIL).mean())
    deg=np.array([len(nb[v]) for v in range(N)],np.float32)
    return torch.tensor(sig).unsqueeze(1), {t:torch.tensor(tg[t]) for t in TARGETS}, torch.tensor(deg)

class AggReg(nn.Module):
    def __init__(s,kind):
        super().__init__(); s.kind=kind; s.phi=nn.Linear(1,D)
        if kind=="ltq":
            s.tau=nn.Parameter(torch.linspace(-1.5,2.5,K).repeat(D,1)+0.01*torch.randn(D,K))
            s.logs=nn.Parameter(torch.zeros(D,K)); outdim=D*K*2+1     # frac sketch, count sketch, log-deg
        elif kind=="pna": outdim=D*3*3+1                              # {mean,max,min} x {1,log-deg,1/log-deg} + log-deg
        else: outdim=D+1                                             # agg + log-deg
        s.head=nn.Sequential(nn.Linear(outdim,H),nn.ReLU(),nn.Linear(H,H),nn.ReLU(),nn.Linear(H,1))
    def forward(s,x,ei,deg):
        src,dst=ei; m=s.phi(x); ld=torch.log1p(deg).unsqueeze(1)
        if s.kind=="mean": agg=scatter(m[src],dst,0,dim_size=N,reduce='mean'); feat=torch.cat([agg,ld],1)
        elif s.kind=="max": agg=scatter(m[src],dst,0,dim_size=N,reduce='max'); feat=torch.cat([agg,ld],1)
        elif s.kind=="pna":
            me=scatter(m[src],dst,0,dim_size=N,reduce='mean'); mx=scatter(m[src],dst,0,dim_size=N,reduce='max')
            mn=scatter(m[src],dst,0,dim_size=N,reduce='min'); base=torch.cat([me,mx,mn],1)
            sc=torch.cat([base,base*ld,base/ld.clamp(min=1e-3)],1); feat=torch.cat([sc,ld],1)
        else:
            ms=m[src]; ind=torch.sigmoid((ms.unsqueeze(-1)-s.tau.unsqueeze(0))/F.softplus(s.logs).unsqueeze(0).clamp(min=1e-3))
            frac=scatter(ind.reshape(ms.size(0),D*K),dst,0,dim_size=N,reduce='mean')   # fraction above thresholds
            cnt=frac*deg.unsqueeze(1)                                                   # count = fraction * degree
            feat=torch.cat([frac,cnt,ld],1)
        return s.head(feat).squeeze(1)

def r2(y,p):
    y=y.numpy(); p=p.numpy(); return 1-((y-p)**2).sum()/(((y-y.mean())**2).sum()+1e-9)

def run(kind,x,ei,deg,y,tr,va,te,seed):
    torch.manual_seed(seed); m=AggReg(kind); opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=1e-5)
    mu,sd=y[tr].mean(),y[tr].std().clamp(min=1e-6); yz=(y-mu)/sd
    best=-1e9;bs=None;bad=0
    for ep in range(EPOCHS):
        m.train();opt.zero_grad(); o=m(x,ei,deg); loss=F.mse_loss(o[tr],yz[tr]); loss.backward();opt.step()
        if ep%5==0:
            m.eval()
            with torch.no_grad(): rv=r2(yz[va],m(x,ei,deg)[va])
            if rv>best+1e-4: best=rv;bs={k:v.clone() for k,v in m.state_dict().items()};bad=0
            else: bad+=1
            if bad>10: break
    if bs: m.load_state_dict(bs)
    m.eval()
    with torch.no_grad(): return r2(yz[te],m(x,ei,deg)[te])

if __name__=="__main__":
    res={t:{k:[] for k in ["mean","max","pna","ltq"]} for t in TARGETS}
    for seed in SEEDS:
        ei=make_graph(seed); x,tg,deg=make(ei,seed)
        idx=np.arange(N); rng=np.random.default_rng(seed); rng.shuffle(idx)
        tr,va,te=idx[:3000],idx[3000:3800],idx[3800:]
        for t in TARGETS:
            for kind in ["mean","max","pna","ltq"]:
                res[t][kind].append(run(kind,x,ei,deg,tg[t],tr,va,te,seed))
        print(f"[seed {seed}] done",flush=True)
    print("\n=== Test R^2 (mean over %d seeds) — aggregator x target ==="%len(SEEDS))
    print(f"{'target':10s} | {'mean':>11s} {'max':>11s} {'pna':>11s} {'ltq':>11s}   LTQ-PNA")
    for t in TARGETS:
        g=lambda k:np.mean(res[t][k]); row=f"{t:10s} |"
        for k in ["mean","max","pna","ltq"]: row+=f"  {g(k):+.3f}"
        gap=g('ltq')-g('pna'); flag=" PASS" if (t in ("count_c","frac_tail") and gap>=0.05) else ""
        print(row+f"   {gap:+.3f}{flag}")
    import json
    json.dump({t:{k:[round(float(z),4) for z in v] for k,v in d.items()} for t,d in res.items()},
              open("data/ext/pure/v2/ltq_synthetic2.json","w"),indent=2)
    print("\nBAR: LTQ-PNA >= +0.05 on count_c AND frac_tail to validate. saved -> ltq_synthetic2.json")
