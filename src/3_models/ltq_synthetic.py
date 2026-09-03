#!/usr/bin/env python3
"""LTQ-MP go/no-go: controlled synthetic test isolating the AGGREGATOR.

Hypothesis (the new mechanism's reason to exist):
  When a node's target depends on the TAIL of a zero-inflated neighbour-signal distribution
  (e.g. "any neighbour above c", "at least k neighbours above c", a high quantile), MEAN
  aggregation is information-destroying as degree grows, while a learned threshold-quantile
  sketch (LTQ) recovers the right statistic. PNA's FIXED {mean,max,std} cannot represent a
  learned count/mid-quantile, so LTQ should be the only aggregator strong across ALL regimes.

Fair protocol: every aggregator shares the same per-neighbour embedding phi and the same head
capacity; ONLY the pooling operator differs. Self-loops added for all (equal footing).
Reports test ROC-AUC per (regime x aggregator), mean over seeds.
Run:  python3 src/3_models/ltq_synthetic.py
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.utils import scatter
from sklearn.metrics import roc_auc_score

DEV="cpu"; N=4000; AVGDEG=12; D=8; K=8; H=64; EPOCHS=150; SEEDS=[0,1,2]
P_INFLATE=0.12          # signal is positive in ~12% of nodes (zero-inflated)
C=1.0                   # tail threshold on the raw signal
REGIMES=["mean","any","count2","q75"]

def make_graph(seed):
    rng=np.random.default_rng(seed)
    M=N*AVGDEG//2
    s=rng.integers(0,N,M); d=rng.integers(0,N,M); m=s!=d; s,d=s[m],d[m]
    src=np.concatenate([s,d,np.arange(N)]); dst=np.concatenate([d,s,np.arange(N)])  # symmetric + self-loops
    return torch.tensor(np.stack([src,dst]),dtype=torch.long)

def make_signal_labels(ei,seed):
    rng=np.random.default_rng(seed+1000)
    sig=((rng.random(N)<P_INFLATE)*np.abs(rng.normal(0,1,N))).astype(np.float32)
    src,dst=ei.numpy()
    # gather neighbour signals per node
    from collections import defaultdict
    nb=defaultdict(list)
    for a,b in zip(src,dst):
        if a!=b: nb[b].append(sig[a])
    labels={}
    for reg in REGIMES:
        stat=np.zeros(N,np.float32)
        for v in range(N):
            vals=np.array(nb[v]) if nb[v] else np.array([0.0],np.float32)
            if reg=="mean":   stat[v]=vals.mean()
            elif reg=="any":  stat[v]=float((vals>C).sum()>=1)
            elif reg=="count2": stat[v]=float((vals>C).sum()>=2)
            elif reg=="q75":  stat[v]=np.quantile(vals,0.75)
        thr=np.quantile(stat,0.86)
        prob=1/(1+np.exp(-(stat-thr-1e-6)*6))
        y=(rng.random(N)<prob).astype(np.int64)
        labels[reg]=y
    return torch.tensor(sig).unsqueeze(1), labels

class AggGNN(nn.Module):
    def __init__(s,kind):
        super().__init__(); s.kind=kind
        s.phi=nn.Linear(1,D)
        if kind=="ltq":
            s.tau=nn.Parameter(torch.linspace(-1,2,K).repeat(D,1)+0.01*torch.randn(D,K))
            s.logs=nn.Parameter(torch.zeros(D,K))
            outdim=D*K
        elif kind=="pna": outdim=3*D
        else: outdim=D
        s.head=nn.Sequential(nn.Linear(outdim,H),nn.ReLU(),nn.Dropout(0.1),nn.Linear(H,1))
    def forward(s,x,ei):
        src,dst=ei; m=s.phi(x)                       # N x D per-neighbour embedding
        if s.kind=="mean": agg=scatter(m[src],dst,0,dim_size=N,reduce='mean')
        elif s.kind=="max": agg=scatter(m[src],dst,0,dim_size=N,reduce='max')
        elif s.kind=="pna":
            me=scatter(m[src],dst,0,dim_size=N,reduce='mean')
            mx=scatter(m[src],dst,0,dim_size=N,reduce='max')
            mn=scatter(m[src],dst,0,dim_size=N,reduce='min')
            agg=torch.cat([me,mx,mn-me],1)           # mean, max, spread (PNA-style fixed bank)
        else:                                        # LTQ: soft fraction of neighbours above each learned threshold
            ms=m[src]                                # E x D
            ind=torch.sigmoid((ms.unsqueeze(-1)-s.tau.unsqueeze(0))/F.softplus(s.logs).unsqueeze(0).clamp(min=1e-3))
            agg=scatter(ind.reshape(ms.size(0),D*K),dst,0,dim_size=N,reduce='mean')
        return s.head(agg).squeeze(1)

def run_one(kind,x,ei,y,tr,va,te,seed):
    torch.manual_seed(seed); m=AggGNN(kind)
    opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=1e-4)
    yt=torch.tensor(y,dtype=torch.float32); pw=torch.tensor([(y[tr]==0).sum()/max((y[tr]==1).sum(),1)])
    best=-1;bs=None;bad=0
    for ep in range(EPOCHS):
        m.train();opt.zero_grad()
        o=m(x,ei); loss=F.binary_cross_entropy_with_logits(o[tr],yt[tr],pos_weight=pw)
        loss.backward();opt.step()
        if ep%5==0:
            m.eval()
            with torch.no_grad(): pv=torch.sigmoid(m(x,ei)[va]).numpy()
            try: a=roc_auc_score(y[va],pv)
            except: a=0
            if a>best+1e-4: best=a;bs={k:v.clone() for k,v in m.state_dict().items()};bad=0
            else: bad+=1
            if bad>8: break
    if bs: m.load_state_dict(bs)
    m.eval()
    with torch.no_grad(): pt=torch.sigmoid(m(x,ei)[te]).numpy()
    return roc_auc_score(y[te],pt)

if __name__=="__main__":
    res={reg:{k:[] for k in ["mean","max","pna","ltq"]} for reg in REGIMES}
    for seed in SEEDS:
        ei=make_graph(seed); x,labels=make_signal_labels(ei,seed)
        idx=np.arange(N); rng=np.random.default_rng(seed)
        rng.shuffle(idx); tr,va,te=idx[:2400],idx[2400:3000],idx[3000:]
        for reg in REGIMES:
            y=labels[reg]
            for kind in ["mean","max","pna","ltq"]:
                res[reg][kind].append(run_one(kind,x,ei,y,tr,va,te,seed))
        print(f"[seed {seed}] done",flush=True)
    print("\n=== Test ROC-AUC (mean over %d seeds) — aggregator x regime ==="%len(SEEDS))
    print(f"{'regime':8s} | {'mean':>12s} {'max':>12s} {'pna':>12s} {'ltq':>12s}")
    for reg in REGIMES:
        row=f"{reg:8s} |"
        for k in ["mean","max","pna","ltq"]:
            v=np.array(res[reg][k]); row+=f"  {v.mean():.3f}±{v.std():.3f}"
        # mark winner
        best=max(["mean","max","pna","ltq"],key=lambda k:np.mean(res[reg][k]))
        print(row+f"   <- best: {best}")
    import json
    json.dump({reg:{k:[round(float(z),4) for z in v] for k,v in d.items()} for reg,d in res.items()},
              open("data/ext/pure/v2/ltq_synthetic.json","w"),indent=2)
    print("\nsaved -> data/ext/pure/v2/ltq_synthetic.json")
