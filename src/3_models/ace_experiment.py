#!/usr/bin/env python3
"""ACE-GNN: Attention + Collective inference + Ensemble — engineered to BEAT XGB+graph.
Differentiators vs hand-crafted exposure features fed to XGBoost:
  (1) strong tabular arm: CONCATENATED GBDT-leaf embeddings (not summed) -> matches XGB on features.
  (2) attention aggregation over the monotone hierarchical auditor ties (partner>office>auditor) + board + ownership.
  (3) COLLECTIVE INFERENCE: iteratively propagate the model's *predicted risk* over the auditor graph
      (APPNP on logits) -> spreads latent risk to firms whose peers are predicted-risky even with NO
      observed problem yet. XGB (independent, observed-features-only) structurally cannot do this.
  (4) decorrelated 2-tower ENSEMBLE: logistic meta-blend of ACE-GNN and XGB+graph (val-fit).
ENV: MODE=recent|panel  LABEL=label|adverse|severe  OUT=...
"""
import json, sys, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.utils import scatter, softmax as gsoftmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
torch.manual_seed(0); np.random.seed(0); DEV="cpu"
MODE=os.environ.get("MODE","recent"); LABEL_MODE=os.environ.get("LABEL","severe")
OUT=os.environ.get("OUT",f"data/ext/ace_{MODE}_{LABEL_MODE}.json")
LKEY={"adverse":"label_adverse","severe":"label_severe"}.get(LABEL_MODE,"label")
RKEY={"adverse":"restated_now_adverse","severe":"restated_now_severe"}.get(LABEL_MODE,"restated_now")
G=torch.load("data/ext/graph.pt", weights_only=False)
YEARS=G["years"]; FEAT=G["feat_names"]; yidx={y:i for i,y in enumerate(YEARS)}
X=G["X"].numpy().copy(); active=G["active"].numpy(); label=G[LKEY].numpy(); rn=G[RKEY].numpy().astype(np.float32)
Tall,N,Fdim=X.shape; snaps=G["snapshots"]
AUD=["partner","office","auditor"] if MODE=="recent" else ["office","auditor"]
OTH=["board","ownership"]; ALL=AUD+OTH
if MODE=="panel": USE=[yidx[y] for y in range(2005,2020)]; TR=USE[:11]; VA=[USE[11]]; TE=USE[12:15]
else: USE=[yidx[y] for y in range(2017,2023)]; TR=USE[:3]; VA=[USE[3]]; TE=USE[4:6]
print(f"[ACE] MODE={MODE} label={LABEL_MODE} train={[YEARS[i] for i in TR]} test={[YEARS[i] for i in TE]} aud={AUD}",file=sys.stderr)
# PLACEBO leakage check: permute TRAIN+VALIDATION labels (everything the model fits/selects on);
# features (incl. exposure o_t) and TEST labels are untouched. If the ACE pipeline carries no
# look-ahead leakage, test ROC must collapse to ~0.5 (val is shuffled so early-stopping cannot peek).
if os.environ.get("PLACEBO"):
    _prng=np.random.default_rng(0)
    for t in TR+VA:
        a=np.where(active[t]&(label[t]>=0))[0]; label[t,a]=_prng.permutation(label[t,a])
    print("[PLACEBO] train+val labels permuted (features/test untouched)",file=sys.stderr)

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
    xs=[];ys=[];idxs=[]
    for t in ts:
        a=np.where(active[t]&(label[t]>=0))[0]; xs.append(arr[t][a]); ys.append(label[t][a]); idxs.append((t,a))
    return np.concatenate(xs),np.concatenate(ys),idxs
Xtr_f,ytr,_=pool(TR,Xz); Xte_f,yte,_=pool(TE,Xz); Xva_f,yva,_=pool(VA,Xz)
Xtr_g,_,_=pool(TR,GF); Xte_g,_,_=pool(TE,GF); Xva_g,_,_=pool(VA,GF)
spw=float((ytr==0).sum()/max((ytr==1).sum(),1))
def metr(y,p):
    o=np.argsort(-p); return {"roc":round(float(roc_auc_score(y,p)),4),"pr":round(float(average_precision_score(y,p)),4),
      "recall@10%":round(float(y[o[:max(1,len(p)//10)]].sum()/max(y.sum(),1)),4)}
def mkxgb(n): return xgb.XGBClassifier(n_estimators=n,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,
    min_child_weight=5,reg_lambda=2.0,scale_pos_weight=spw,eval_metric="aucpr",n_jobs=4,tree_method="hist")
RES={"mode":MODE,"label":LABEL_MODE,"test_years":[YEARS[i] for i in TE],"results":{}}
RES["results"]["XGB_feat"]=metr(yte,mkxgb(400).fit(Xtr_f,ytr).predict_proba(Xte_f)[:,1])
xgb_g=mkxgb(150).fit(Xtr_g,ytr); pXg_te=xgb_g.predict_proba(Xte_g)[:,1]; pXg_va=xgb_g.predict_proba(Xva_g)[:,1]
RES["results"]["XGB_feat+graph"]=metr(yte,pXg_te)
# additional STRONG tabular+graph base learners (so the ensemble incorporates the best baselines)
from sklearn.ensemble import RandomForestClassifier as _RF
import lightgbm as _lgb
rf_g=_RF(n_estimators=400,min_samples_leaf=5,class_weight="balanced_subsample",n_jobs=4,random_state=0).fit(Xtr_g,ytr)
pRf_te=rf_g.predict_proba(Xte_g)[:,1]; pRf_va=rf_g.predict_proba(Xva_g)[:,1]; RES["results"]["RF_feat+graph"]=metr(yte,pRf_te)
lgb_g=_lgb.LGBMClassifier(n_estimators=400,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,reg_lambda=2.0,scale_pos_weight=spw,n_jobs=4,verbose=-1,random_state=0).fit(Xtr_g,ytr)
pLg_te=lgb_g.predict_proba(Xte_g)[:,1]; pLg_va=lgb_g.predict_proba(Xva_g)[:,1]; RES["results"]["LGB_feat+graph"]=metr(yte,pLg_te)
# strongest single baseline = the hardest competitor on TEST (conservative bar: beating the strongest
# baseline is the toughest comparison, so this cannot flatter ACE). We also report ACE vs all three +graph
# bases below, so the verdict never depends on which single competitor is named.
_basep={"XGB_feat+graph":pXg_te,"RF_feat+graph":pRf_te,"LGB_feat+graph":pLg_te}
BEST_BASE=max(_basep,key=lambda k:roc_auc_score(yte,_basep[k])); pBEST_te=_basep[BEST_BASE]
print(f"[base+graph] XGB={RES['results']['XGB_feat+graph']['roc']} RF={RES['results']['RF_feat+graph']['roc']} LGB={RES['results']['LGB_feat+graph']['roc']} | strongest={BEST_BASE}",file=sys.stderr)
def logit(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
LV=np.stack([xgb_g.apply(GF[t]) for t in range(Tall)],0).astype(np.int64); NTREE=LV.shape[2]; LC=int(LV.max())+1
off=(np.arange(NTREE)*LC).astype(np.int64)
print(f"[base] XGBf={RES['results']['XGB_feat']['roc']} XGBg={RES['results']['XGB_feat+graph']['roc']}",file=sys.stderr)

Xz_t=torch.from_numpy(Xz); XL_t=torch.from_numpy(XL); OWN_t=torch.from_numpy(OWN); EXP_t=torch.from_numpy(EXP)
LE_t=torch.from_numpy(LV+off[None,None,:]); lab_t=torch.from_numpy(label)
A=[{c:adj(t,c) for c in ALL} for t in range(Tall)]

class ACE(nn.Module):
    def __init__(s,H=96,emb=8,layers=2,collective_steps=5,drop=0.3,use_leaf=True,use_feat=True):
        super().__init__(); s.H=H; s.cs=collective_steps; s.use_leaf=use_leaf; s.use_feat=use_feat
        s.leaf=nn.Embedding(NTREE*LC,emb); s.leafproj=nn.Sequential(nn.Linear(NTREE*emb,2*H),nn.ReLU(),nn.Dropout(drop),nn.Linear(2*H,H))
        fin=(Fdim*2 if use_feat else 0)+2+len(ALL)*2
        s.featenc=nn.Sequential(nn.Linear(fin,H),nn.ReLU())
        s.lvl=nn.Parameter(torch.zeros(len(AUD)))
        # attention params per relation channel (aud-hier counts as one channel)
        s.nchan=1+len(OTH); s.asrc=nn.ModuleList([nn.Linear(H,1) for _ in range(s.nchan)]); s.adst=nn.ModuleList([nn.Linear(H,1) for _ in range(s.nchan)])
        s.lin=nn.ModuleList([nn.Linear(H,H) for _ in range(s.nchan)])
        s.gate=nn.Linear(H,s.nchan); s.upd=nn.ModuleList([nn.Linear(2*H,H) for _ in range(layers)]); s.layers=layers; s.drop=nn.Dropout(drop)
        s.head=nn.Sequential(nn.Linear(2*H,H),nn.ReLU(),nn.Dropout(drop),nn.Linear(H,1))
        s.beta=nn.Parameter(torch.tensor(0.0))
    def lvlw(s):
        sp=F.softplus(s.lvl); c=torch.cumsum(sp.flip(0),0).flip(0); return c/c.max().clamp(min=1e-6)
    def attn(s,k,h,ei,w):
        if ei.size(1)==0: return torch.zeros_like(h)
        src,dst=ei; hs=s.lin[k](h)
        e=F.leaky_relu(s.asrc[k](hs)[src]+s.adst[k](hs)[dst]).squeeze(-1)+torch.log(w.clamp(min=1e-6))
        al=gsoftmax(e,dst,num_nodes=N)
        return scatter(al.unsqueeze(1)*hs[src],dst,0,dim_size=N,reduce='sum')
    def aud_attn(s,h,t):
        w=s.lvlw(); m=0
        for li,c in enumerate(AUD):
            ei,ew=A[t][c]; m=m+w[li]*s.attn(0,h,ei,ew)
        return m
    def forward(s,t):
        inp=([Xz_t[t],XL_t[t]] if s.use_feat else [])+[OWN_t[t],EXP_t[t]]
        h0=s.featenc(torch.cat(inp,1))
        if s.use_leaf: h0=h0+s.leafproj(s.leaf(LE_t[t]).reshape(N,-1))
        h0=F.relu(h0)
        h=h0
        for l in range(s.layers):
            chans=[s.aud_attn(h,t)]+[s.attn(1+j,h,*A[t][c]) for j,c in enumerate(OTH)]
            g=torch.softmax(s.gate(h),1)
            agg=sum(g[:,k:k+1]*chans[k] for k in range(s.nchan))
            h=F.relu(s.upd[l](torch.cat([h,agg],1))); h=s.drop(h)
        y=s.head(torch.cat([h0,h],1)).squeeze(1)
        # collective inference: propagate predicted logit over aud-hier (mean), APPNP-style
        w=s.lvlw(); b=torch.sigmoid(s.beta)*0.6
        yk=y
        for _ in range(s.cs):
            msg=0
            for li,c in enumerate(AUD): msg=msg+w[li]*mean_agg(yk,*A[t][c])
            wsum=float(w.detach().sum())
            yk=(1-b)*y+b*(msg/max(wsum,1e-6))
        return y, yk
    def forward_logit(s,t):  # final = base logit + collective refinement
        y,yk=s.forward(t); return y+yk-y.detach()*0+ (yk-y)  # = yk ; keep graph grad

POSW=torch.tensor([spw])
def get_scores(m,ts):
    sc={}
    with torch.no_grad():
        for t in ts: _,yk=m(t); sc[t]=yk
    return sc
def train_ace(seed=0,**kw):
    torch.manual_seed(seed); m=ACE(**kw); opt=torch.optim.Adam(m.parameters(),lr=3e-3,weight_decay=2e-4)
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
        for t in ts: _,yk=m(t); a=np.where(active[t]&(label[t]>=0))[0]; ps.append(torch.sigmoid(yk[a]).numpy()); ys.append(label[t][a])
    return np.concatenate(ys),np.concatenate(ps)
# ---- bag 3 diverse members: full(leaf+feat), raw(feat), graphonly(no feat/leaf, max-decorrelated) ----
NSEED=6
def bag(use_leaf,use_feat,tag):
    te=[];va=[];lw=[]
    for sd in range(NSEED):
        m,bv=train_ace(seed=sd,use_leaf=use_leaf,use_feat=use_feat)
        y_te,p_te=collect(m,TE); y_va,p_va=collect(m,VA)
        te.append(p_te); va.append(p_va); lw.append([float(m.lvlw()[i]) for i in range(len(AUD))])
        print(f"  [{tag} s{sd}] vp={bv:.4f} roc={metr(y_te,p_te)['roc']}",file=sys.stderr)
    return np.mean(te,0),np.mean(va,0),y_te,y_va,np.mean(lw,0)
t0=time.time()
pF_te,pF_va,yte_,yva_,lwF=bag(True,True,"full")
pR_te,pR_va,_,_,_=bag(False,True,"raw")
pG_te,pG_va,_,_,_=bag(False,False,"graphonly")
RES["results"]["XGB_feat+graph"]=metr(yte_,pXg_te)
RES["results"]["ACE_full_bag"]=metr(yte_,pF_te); RES["results"]["ACE_full_bag"]["aud_level_weights"]={c:round(float(lwF[i]),3) for i,c in enumerate(AUD)}
RES["results"]["ACE_raw_bag"]=metr(yte_,pR_te)
RES["results"]["ACE_graphonly_bag"]=metr(yte_,pG_te)
# helper: downside-safe meta-ensemble over a set of (val,test) prob columns; blend vs strongest base
def safe_meta(cols_va, cols_te, fallback_va, fallback_te):
    Zva=np.column_stack([logit(c) for c in cols_va]); Zte=np.column_stack([logit(c) for c in cols_te])
    mt=LogisticRegression(max_iter=2000,C=1.0).fit(Zva,yva_); pv=mt.predict_proba(Zva)[:,1]; pt=mt.predict_proba(Zte)[:,1]
    gs=np.linspace(0,1,21); gb=max(gs,key=lambda g: roc_auc_score(yva_, g*pv+(1-g)*fallback_va))
    return gb*pt+(1-gb)*fallback_te, [round(float(c),3) for c in mt.coef_[0]], round(float(gb),2)
# (a) strong TABULAR-ONLY ensemble (no GNN) — does the GNN add anything beyond this?
ptab,_,_=safe_meta([pXg_va,pRf_va,pLg_va],[pXg_te,pRf_te,pLg_te],pRf_va,pRf_te)
RES["results"]["ENS_tabular(XGB+RF+LGB)"]=metr(yte_,ptab)
# (b) FULL ensemble: strong tabular base learners + decorrelated GNN members
mem_va=[pXg_va,pRf_va,pLg_va,pF_va,pR_va,pG_va]; mem_te=[pXg_te,pRf_te,pLg_te,pF_te,pR_te,pG_te]
pens,mcoef,gbest=safe_meta(mem_va,mem_te,pRf_va,pRf_te)   # fallback to strongest base (RF+graph)
RES["results"]["ENSEMBLE_safe"]=metr(yte_,pens); RES["results"]["ENSEMBLE_safe"]["blend_gamma"]=gbest
RES["results"]["ENSEMBLE_safe"]["meta_members"]=["XGB+g","RF+g","LGB+g","ACE-full","ACE-raw","ACE-graphonly"]
RES["results"]["ENSEMBLE_safe"]["meta_coef"]=mcoef
# (c) parameter-free RANK-AVERAGE combiner over the same six members — no validation tuning, no meta-overfit
from scipy.stats import rankdata
def rank_avg(cols): return np.mean([rankdata(c)/len(c) for c in cols],0)
prank=rank_avg(mem_te)
RES["results"]["ENSEMBLE_rankavg"]=metr(yte_,prank)
RES["pred_corr"]={"xgb_vs_full":round(float(np.corrcoef(pXg_te,pF_te)[0,1]),3),
                  "xgb_vs_graphonly":round(float(np.corrcoef(pXg_te,pG_te)[0,1]),3),
                  "RFg_vs_graphonly":round(float(np.corrcoef(pRf_te,pG_te)[0,1]),3)}
def boot(pa,pb,B=3000):
    rng=np.random.default_rng(0); n=len(yte_); d=[]
    for _ in range(B):
        idx=rng.integers(0,n,n)
        if 0<yte_[idx].sum()<len(idx): d.append(roc_auc_score(yte_[idx],pa[idx])-roc_auc_score(yte_[idx],pb[idx]))
    d=np.array(d); return {"mean":round(float(d.mean()),4),"ci95":[round(float(np.percentile(d,2.5)),4),round(float(np.percentile(d,97.5)),4)],"p_gt_0":round(float((d>0).mean()),4)}
# decisive tests: ENSEMBLE vs strongest single baseline, and vs tabular-only ensemble (does GNN add?)
RES["bootstrap"]={"ENS_vs_"+BEST_BASE:boot(pens,pBEST_te),
                  "ENS_vs_tabularENS":boot(pens,ptab),
                  "ENS_vs_XGB+graph":boot(pens,pXg_te),
                  "RANKAVG_vs_"+BEST_BASE:boot(prank,pBEST_te)}
RES["strongest_single_baseline"]=BEST_BASE
RES["runtime_sec"]=round(time.time()-t0,1)
np.savez(OUT.replace('.json','_preds.npz'),y=yte_,xgb=pXg_te,rf=pRf_te,lgb=pLg_te,full=pF_te,raw=pR_te,graphonly=pG_te,tabens=ptab,ens=pens,rankavg=prank)
json.dump(RES,open(OUT,"w"),indent=2,ensure_ascii=False)
print(f"\n=== {MODE}/{LABEL_MODE} test {[YEARS[i] for i in TE]} ===")
for k,v in RES["results"].items(): print(f"{k:32s} ROC={v.get('roc')} PR={v.get('pr')} r@10%={v.get('recall@10%')}")
print(f"strongest single baseline = {BEST_BASE}")
for k,v in RES["bootstrap"].items(): print(f"  {k}: gain={v['mean']:+.4f} CI{v['ci95']} p={v['p_gt_0']}")
print(f"[done {RES['runtime_sec']}s -> {OUT}]")
