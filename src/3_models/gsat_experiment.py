#!/usr/bin/env python3
"""GSAT — Graph Stochastic Attention (Miao et al., ICML'22) as a SELF-INTERPRETABLE GNN baseline,
run on the accounting testbed under the IDENTICAL protocol as the standard GNN baselines (benchmark.py):
node feat=[Xz,own], merged multiplex, BCE+pos_weight, early-stop on val PR-AUC, recent/severe.

GSAT learns a per-edge Bernoulli attention p_e via an information bottleneck (KL to a prior r), samples an
edge mask (Gumbel-sigmoid) during training, and predicts from the masked graph. The learned p_e ARE the
explanation (which edges matter). We aggregate p_e by relation type to compare GSAT's edge-level explanation
with ACE-GNN's layer-level channel readout (auditor/office vs board/ownership).
Run from repo root:  MODE=recent LABEL=severe python3 src/3_models/gsat_experiment.py
"""
import json, os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.utils import scatter, softmax as gsoftmax
from sklearn.metrics import roc_auc_score, average_precision_score
torch.manual_seed(0); np.random.seed(0)
MODE=os.environ.get("MODE","recent"); LABEL=os.environ.get("LABEL","severe")
LKEY={"adverse":"label_adverse","severe":"label_severe"}.get(LABEL,"label")
RKEY={"adverse":"restated_now_adverse","severe":"restated_now_severe"}.get(LABEL,"restated_now")
G=torch.load("data/ext/graph.pt",weights_only=False); YEARS=G["years"]; yidx={y:i for i,y in enumerate(YEARS)}
X=G["X"].numpy().copy(); active=G["active"].numpy(); label=G[LKEY].numpy(); rn=G[RKEY].numpy().astype(np.float32)
Tall,N,Fdim=X.shape; snaps=G["snapshots"]
REL=["partner","office","auditor","board","ownership"] if MODE=="recent" else ["office","auditor","board","ownership"]
if MODE=="panel": USE=[yidx[y] for y in range(2005,2020)];TR=USE[:11];VA=[USE[11]];TE=USE[12:15]
else: USE=[yidx[y] for y in range(2017,2023)];TR=USE[:3];VA=[USE[3]];TE=USE[4:6]
# train-stat winsorize/impute/zscore (no leakage)
tm=np.zeros((Tall,N),bool)
for t in TR: tm[t]=active[t]&(label[t]>=0)
Xtr0=X[tm]; lo=np.nanpercentile(Xtr0,1,0); hi=np.nanpercentile(Xtr0,99,0); med=np.nanmedian(Xtr0,0)
Xc=np.clip(X,lo,hi); ix=np.where(np.isnan(Xc)); Xc[ix]=np.take(med,ix[2])
mu=np.nanmean(np.clip(Xtr0,lo,hi),0); sd=np.nanstd(np.clip(Xtr0,lo,hi),0); sd[sd<1e-6]=1
Xz=((Xc-mu)/sd).astype(np.float32)
OWN=np.zeros((Tall,N,2),np.float32)
for t in range(Tall): OWN[t,:,0]=rn[t]; OWN[t,:,1]=rn[max(t-1,0)]
NF=np.concatenate([Xz,OWN],2).astype(np.float32); NF_t=torch.from_numpy(NF); lab_t=torch.from_numpy(label); nfeat=NF.shape[2]
def adj(t,c): ei,w=snaps[t][c]; return ei,w
merged=[]; rtype=[]
for t in range(Tall):
    eis=[];ets=[]
    for ri,c in enumerate(REL):
        ei,w=adj(t,c); eis.append(ei); ets.append(torch.full((ei.size(1),),ri,dtype=torch.long))
    merged.append(torch.cat(eis,1) if eis else torch.zeros((2,0),dtype=torch.long))
    rtype.append(torch.cat(ets) if ets else torch.zeros(0,dtype=torch.long))
ytr=np.concatenate([label[t][active[t]&(label[t]>=0)] for t in TR]); spw=float((ytr==0).sum()/max((ytr==1).sum(),1))
yte=np.concatenate([label[t][active[t]&(label[t]>=0)] for t in TE])
print(f"[gsat] MODE={MODE} label={LABEL} rels={REL} nfeat={nfeat} spw={spw:.1f} test={[YEARS[i] for i in TE]}",file=sys.stderr)

H=64
class GSAT(nn.Module):
    def __init__(s,drop=0.4):
        super().__init__()
        s.enc=nn.Sequential(nn.Linear(nfeat,H),nn.ReLU(),nn.Dropout(drop))           # node encoder
        s.att=nn.Sequential(nn.Linear(2*H,H),nn.ReLU(),nn.Linear(H,1))               # per-edge attention logit
        s.l2=nn.Linear(H,H); s.head=nn.Linear(H,1); s.drop=nn.Dropout(drop)
    def edge_p(s,h,ei):                                                              # Bernoulli attention p_e
        src,dst=ei; return s.att(torch.cat([h[src],h[dst]],1)).squeeze(-1)
    def mp(s,h,ei,z):                                                               # masked mean message passing
        src,dst=ei; num=scatter((h[src])*z.unsqueeze(1),dst,0,dim_size=N,reduce='sum')
        den=scatter(z,dst,0,dim_size=N,reduce='sum').clamp(min=1e-6); return num/den.unsqueeze(1)
    def forward(s,t,sample=True,temp=1.0):
        ei=merged[t]; h=s.enc(NF_t[t]); logit=s.edge_p(h,ei); p=torch.sigmoid(logit)
        if sample and s.training:
            u=torch.rand_like(logit).clamp(1e-6,1-1e-6); g=torch.log(u)-torch.log(1-u)
            z=torch.sigmoid((logit+g)/temp)
        else: z=p
        m=s.mp(h,ei,z); h2=F.relu(s.l2(m)); h2=s.drop(h2)
        return s.head(h+ h2 if h.shape==h2.shape else h2).squeeze(1), p
R_PRIOR=float(os.environ.get("R_PRIOR","0.5")); BETA=float(os.environ.get("BETA","0.1")); POSW=torch.tensor([spw])
def kl_ib(p):                                                                       # KL(Bern(p)||Bern(r))
    r=R_PRIOR; p=p.clamp(1e-6,1-1e-6)
    return (p*torch.log(p/r)+(1-p)*torch.log((1-p)/(1-r))).mean()
def run(seed=0):
    torch.manual_seed(seed); m=GSAT(); opt=torch.optim.Adam(m.parameters(),lr=5e-3,weight_decay=5e-4)
    tri={t:torch.from_numpy(np.where(active[t]&(label[t]>=0))[0]).long() for t in TR}
    vai={t:np.where(active[t]&(label[t]>=0))[0] for t in VA}; best=-1;bs=None;bad=0
    for ep in range(200):
        m.train(); opt.zero_grad(); loss=0
        for t in TR:
            o,p=m(t,sample=True); idx=tri[t]
            loss=loss+F.binary_cross_entropy_with_logits(o[idx],lab_t[t][idx].float(),pos_weight=POSW)+BETA*kl_ib(p)
        loss.backward(); opt.step()
        if ep%5==0:
            m.eval(); ys=[];ps=[]
            with torch.no_grad():
                for t in VA: o,_=m(t,sample=False); a=vai[t]; ps.append(torch.sigmoid(o[a]).numpy()); ys.append(label[t][a])
            vp=average_precision_score(np.concatenate(ys),np.concatenate(ps))
            if vp>best: best=vp; bs={k:v.clone() for k,v in m.state_dict().items()};bad=0
            else: bad+=1
            if bad>8: break
    if bs: m.load_state_dict(bs)
    m.eval(); ys=[];ps=[]; rel_att={c:[] for c in REL}
    with torch.no_grad():
        for t in TE:
            o,p=m(t,sample=False); a=np.where(active[t]&(label[t]>=0))[0]; ps.append(torch.sigmoid(o[a]).numpy()); ys.append(label[t][a])
            for ri,c in enumerate(REL):
                mask=(rtype[t]==ri);
                if mask.sum()>0: rel_att[c].append(float(p[mask].mean()))
    p_te=np.concatenate(ps); y_=np.concatenate(ys)
    return p_te,y_,{c:(float(np.mean(v)) if v else 0.0) for c,v in rel_att.items()}
P=[];RA=[]
for sd in range(int(os.environ.get("NSEED","3"))):
    p,y_,ra=run(sd); P.append(p); RA.append(ra); print(f"  [gsat s{sd}] roc={roc_auc_score(yte,p):.4f}",file=sys.stderr)
pm=np.mean(P,0)
relatt={c:round(float(np.mean([r[c] for r in RA])),3) for c in REL}
o=np.argsort(-pm); r10=float(yte[o[:max(1,len(pm)//10)]].sum()/max(yte.sum(),1))
np.savez(f"data/ext/gsat_{MODE}_{LABEL}_preds.npz",y=yte,gsat=pm)
out={"mode":MODE,"label":LABEL,"model":"GSAT","roc":round(float(roc_auc_score(yte,pm)),4),
     "pr":round(float(average_precision_score(yte,pm)),4),"recall@10%":round(r10,4),
     "edge_attention_by_relation":relatt,"beta":BETA,"r_prior":R_PRIOR,"n_test":int(len(yte))}
json.dump(out,open(f"data/ext/gsat_{MODE}_{LABEL}.json","w"),indent=2)
print(json.dumps(out,ensure_ascii=False,indent=2))
