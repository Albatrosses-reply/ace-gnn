#!/usr/bin/env python3
"""Architecture component ablation + relation drop-one for the ACE-GNN (standalone, no XGB base,
so each component's contribution to the graph model is isolated). 3-seed mean±std. ENV MODE/LABEL."""
import json, sys, os, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.utils import scatter, softmax as gsoftmax
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
torch.manual_seed(0); np.random.seed(0); DEV="cpu"
MODE=os.environ.get("MODE","recent"); LABEL_MODE=os.environ.get("LABEL","severe")
OUT=os.environ.get("OUT",f"data/ext/ablation_{MODE}_{LABEL_MODE}.json"); NSEED=int(os.environ.get("NSEED","3"))
LKEY={"adverse":"label_adverse","severe":"label_severe"}.get(LABEL_MODE,"label")
RKEY={"adverse":"restated_now_adverse","severe":"restated_now_severe"}.get(LABEL_MODE,"restated_now")
G=torch.load("data/ext/graph.pt", weights_only=False)
YEARS=G["years"]; FEAT=G["feat_names"]; yidx={y:i for i,y in enumerate(YEARS)}
X=G["X"].numpy().copy(); active=G["active"].numpy(); label=G[LKEY].numpy(); rn=G[RKEY].numpy().astype(np.float32)
Tall,N,Fdim=X.shape; snaps=G["snapshots"]
AUD=["partner","office","auditor"] if MODE=="recent" else ["office","auditor"]; OTH=["board","ownership"]; ALL=AUD+OTH
if MODE=="panel": USE=[yidx[y] for y in range(2005,2020)]; TR=USE[:11]; VA=[USE[11]]; TE=USE[12:15]
else: USE=[yidx[y] for y in range(2017,2023)]; TR=USE[:3]; VA=[USE[3]]; TE=USE[4:6]
tm=np.zeros((Tall,N),bool)
for t in TR: tm[t]=active[t]&(label[t]>=0)
Xtr0=X[tm]; lo=np.nanpercentile(Xtr0,1,0); hi=np.nanpercentile(Xtr0,99,0); med=np.nanmedian(Xtr0,0)
Xc=np.clip(X,lo,hi); ix=np.where(np.isnan(Xc)); Xc[ix]=np.take(med,ix[2])
mu=np.nanmean(np.clip(Xtr0,lo,hi),0); sd=np.nanstd(np.clip(Xtr0,lo,hi),0); sd[sd<1e-6]=1
Xz=((Xc-mu)/sd).astype(np.float32)
def adj(t,c): ei,w=snaps[t][c]; return ei.to(DEV),w.to(DEV)
def mean_agg(vec,ei,w):
    if ei.size(1)==0: return torch.zeros_like(vec)
    src,dst=ei; v=vec if vec.dim()>1 else vec.unsqueeze(1)
    num=scatter(v[src]*w.unsqueeze(1),dst,0,dim_size=N,reduce='sum'); den=scatter(w,dst,0,dim_size=N,reduce='sum').clamp(min=1e-6)
    o=num/den.unsqueeze(1); return o if vec.dim()>1 else o.squeeze(1)
rnt=[torch.from_numpy(rn[t]) for t in range(Tall)]
EXP=np.zeros((Tall,N,len(ALL)*2),np.float32)
for t in range(Tall):
    for ci,c in enumerate(ALL):
        ei,w=adj(t,c); EXP[t,:,ci]=mean_agg(rnt[t],ei,w).numpy()
        tl=max(t-1,0); eil,wl=adj(tl,c); EXP[t,:,len(ALL)+ci]=mean_agg(rnt[tl],eil,wl).numpy()
OWN=np.zeros((Tall,N,2),np.float32)
for t in range(Tall): OWN[t,:,0]=rn[t]; OWN[t,:,1]=rn[max(t-1,0)]
XL=np.stack([Xz[max(t-1,0)] for t in range(Tall)],0)
GF=np.concatenate([Xz,OWN,EXP],2).astype(np.float32)
def pool(ts,arr):
    xs=[];ys=[]
    for t in ts:
        a=np.where(active[t]&(label[t]>=0))[0]; xs.append(arr[t][a]); ys.append(label[t][a])
    return np.concatenate(xs),np.concatenate(ys)
Xtr_g,ytr=pool(TR,GF); spw=float((ytr==0).sum()/max((ytr==1).sum(),1))
xgb_g=xgb.XGBClassifier(n_estimators=150,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,min_child_weight=5,reg_lambda=2.0,scale_pos_weight=spw,eval_metric="aucpr",n_jobs=4,tree_method="hist").fit(Xtr_g,ytr)
LV=np.stack([xgb_g.apply(GF[t]) for t in range(Tall)],0).astype(np.int64); NTREE=LV.shape[2]; LC=int(LV.max())+1; off=(np.arange(NTREE)*LC)
Xz_t=torch.from_numpy(Xz); XL_t=torch.from_numpy(XL); OWN_t=torch.from_numpy(OWN); EXP_t=torch.from_numpy(EXP); LE_t=torch.from_numpy(LV+off[None,None,:]); lab_t=torch.from_numpy(label)
A=[{c:adj(t,c) for c in ALL} for t in range(Tall)]
def metr(y,p): return {"roc":float(roc_auc_score(y,p)),"pr":float(average_precision_score(y,p))}

class ACE(nn.Module):
    def __init__(s,H=96,emb=8,layers=2,K=5,use_attn=True,mono=True,use_leaf=True,use_feat=True,drop_rel=None,drop=0.3):
        super().__init__(); s.K=K; s.use_attn=use_attn; s.mono=mono; s.use_leaf=use_leaf; s.use_feat=use_feat; s.drop_rel=drop_rel
        s.aud=[c for c in AUD if c!=drop_rel]; s.oth=[c for c in OTH if c!=drop_rel]; s.nchan=1+len(s.oth)
        s.leaf=nn.Embedding(NTREE*LC,emb); s.leafproj=nn.Sequential(nn.Linear(NTREE*emb,2*H),nn.ReLU(),nn.Dropout(drop),nn.Linear(2*H,H))
        fin=(Fdim*2 if use_feat else 0)+2+len(ALL)*2; s.featenc=nn.Sequential(nn.Linear(fin,H),nn.ReLU())
        s.lvl=nn.Parameter(torch.zeros(max(len(s.aud),1)))
        s.asrc=nn.ModuleList([nn.Linear(H,1) for _ in range(s.nchan)]); s.adst=nn.ModuleList([nn.Linear(H,1) for _ in range(s.nchan)])
        s.lin=nn.ModuleList([nn.Linear(H,H) for _ in range(s.nchan)]); s.gate=nn.Linear(H,s.nchan)
        s.upd=nn.ModuleList([nn.Linear(2*H,H) for _ in range(layers)]); s.layers=layers; s.dp=nn.Dropout(drop)
        s.head=nn.Sequential(nn.Linear(2*H,H),nn.ReLU(),nn.Dropout(drop),nn.Linear(H,1)); s.beta=nn.Parameter(torch.tensor(0.0))
    def lvlw(s):
        sp=F.softplus(s.lvl)
        return (torch.cumsum(sp.flip(0),0).flip(0)/torch.cumsum(sp.flip(0),0).flip(0).max().clamp(min=1e-6)) if s.mono else (sp/sp.sum().clamp(min=1e-6))
    def aggr(s,k,h,t,c):
        ei,w=A[t][c]
        if ei.size(1)==0: return torch.zeros_like(s.lin[k](h))
        hs=s.lin[k](h)
        if not s.use_attn: return mean_agg(hs,ei,w)
        src,dst=ei; e=F.leaky_relu(s.asrc[k](hs)[src]+s.adst[k](hs)[dst]).squeeze(-1)+torch.log(w.clamp(min=1e-6))
        al=gsoftmax(e,dst,num_nodes=N); return scatter(al.unsqueeze(1)*hs[src],dst,0,dim_size=N,reduce='sum')
    def forward(s,t):
        inp=([Xz_t[t],XL_t[t]] if s.use_feat else [])+[OWN_t[t],EXP_t[t]]
        h0=s.featenc(torch.cat(inp,1))
        if s.use_leaf: h0=h0+s.leafproj(s.leaf(LE_t[t]).reshape(N,-1))
        h0=F.relu(h0); h=h0
        for l in range(s.layers):
            w=s.lvlw(); audm=sum(w[i]*s.aggr(0,h,t,c) for i,c in enumerate(s.aud)) if s.aud else torch.zeros_like(h)
            chans=[audm]+[s.aggr(1+j,h,t,c) for j,c in enumerate(s.oth)]
            g=torch.softmax(s.gate(h),1); agg=sum(g[:,k:k+1]*chans[k] for k in range(s.nchan))
            h=F.relu(s.upd[l](torch.cat([h,agg],1))); h=s.dp(h)
        y=s.head(torch.cat([h0,h],1)).squeeze(1)
        if s.K==0: return y
        w=s.lvlw(); b=torch.sigmoid(s.beta)*0.6; yk=y
        for _ in range(s.K):
            msg=sum(w[i]*mean_agg(yk,*A[t][c]) for i,c in enumerate(s.aud)) if s.aud else 0.0
            yk=(1-b)*y+b*(msg/float(w.detach().sum()) if s.aud else y)
        return yk
POSW=torch.tensor([spw])
def train_one(seed,**kw):
    torch.manual_seed(seed); m=ACE(**kw); opt=torch.optim.Adam(m.parameters(),lr=3e-3,weight_decay=2e-4)
    tri={t:torch.from_numpy(np.where(active[t]&(label[t]>=0))[0]).long() for t in TR}; vai={t:np.where(active[t]&(label[t]>=0))[0] for t in VA}
    best=-1;bs=None;bad=0
    for ep in range(300):
        m.train(); opt.zero_grad(); loss=0
        for t in TR: o=m(t); idx=tri[t]; loss=loss+F.binary_cross_entropy_with_logits(o[idx],lab_t[t][idx].float(),pos_weight=POSW)
        loss.backward(); opt.step()
        if ep%5==0:
            m.eval(); ys=[];ps=[]
            with torch.no_grad():
                for t in VA: o=m(t); a=vai[t]; ps.append(torch.sigmoid(o[a]).numpy()); ys.append(label[t][a])
            vp=average_precision_score(np.concatenate(ys),np.concatenate(ps))
            if vp>best: best=vp;bs={k:v.clone() for k,v in m.state_dict().items()};bad=0
            else: bad+=1
            if bad>10: break
    if bs: m.load_state_dict(bs)
    m.eval(); ys=[];ps=[]
    with torch.no_grad():
        for t in TE: o=m(t); a=np.where(active[t]&(label[t]>=0))[0]; ps.append(torch.sigmoid(o[a]).numpy()); ys.append(label[t][a])
    return np.concatenate(ys),np.concatenate(ps)
def run(name,**kw):
    rocs=[];prs=[]
    for sd in range(NSEED): y,p=train_one(sd,**kw); rocs.append(metr(y,p)["roc"]); prs.append(metr(y,p)["pr"])
    r={"roc_mean":round(float(np.mean(rocs)),4),"roc_std":round(float(np.std(rocs)),4),"pr_mean":round(float(np.mean(prs)),4)}
    print(f"  {name:30s} ROC={r['roc_mean']:.4f}±{r['roc_std']:.4f} PR={r['pr_mean']:.4f}",file=sys.stderr); return r

RES={"mode":MODE,"label":LABEL_MODE,"nseed":NSEED,"component_ablation":{},"relation_drop":{}}
print("[component ablation]",file=sys.stderr)
RES["component_ablation"]["FULL"]=run("FULL (all components)")
RES["component_ablation"]["-collective(K=0)"]=run("-collective inference",K=0)
RES["component_ablation"]["-attention(mean)"]=run("-attention (mean-pool)",use_attn=False)
RES["component_ablation"]["-monotone(free wts)"]=run("-monotone hierarchy",mono=False)
RES["component_ablation"]["-GBDTleaf"]=run("-GBDT leaf emb",use_leaf=False)
RES["component_ablation"]["graph-only(-features)"]=run("graph-only (no feat/leaf)",use_leaf=False,use_feat=False)
print("[relation drop-one]",file=sys.stderr)
for c in ALL:
    RES["relation_drop"][f"-{c}"]=run(f"drop {c}",drop_rel=c)
json.dump(RES,open(OUT,"w"),indent=2,ensure_ascii=False)
print(f"[done -> {OUT}]",file=sys.stderr)
