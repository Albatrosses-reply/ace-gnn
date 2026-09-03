#!/usr/bin/env python3
"""PLE-AuditorSAGE-APPNP — pure-graph redesign (with Codex). Key lever: PIECEWISE-LINEAR ENCODING (PLE) of the
numerical ratios (Gorishniy et al., numerical embeddings / FT-Transformer) to give a NEURAL encoder tree-like
threshold basis functions -> close the feature-encoding gap WITHOUT trees. Graph is a SMALL residual correction:
auditor-only message passing + APPNP collective inference on the LOGIT.

STAGED (env STAGE=A|B|C): A=PLE encoder only (tabular-neural ceiling; must clear ~0.81 or graph cannot rescue);
B=A + APPNP-on-logit over auditor graph; C=B + auditor-only residual SAGE.
ENV: MODE=recent|panel LABEL=severe STAGE=A B=32(bins) H=256 ENC_DEPTH=3 NSEED=5 KPROP=6 ALPHA=0.2 GAMMA=0.2 EDGE_DROP=0.2
Run from repo root: STAGE=A python3 src/3_models/pleg_experiment.py
"""
import json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.utils import scatter
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
torch.manual_seed(0); np.random.seed(0)
MODE=os.environ.get("MODE","recent"); LABEL=os.environ.get("LABEL","severe"); STAGE=os.environ.get("STAGE","A")
NB=int(os.environ.get("B","32")); H=int(os.environ.get("H","256")); ENC_DEPTH=int(os.environ.get("ENC_DEPTH","3"))
NSEED=int(os.environ.get("NSEED","5")); KPROP=int(os.environ.get("KPROP","6")); ALPHA=float(os.environ.get("ALPHA","0.2"))
GAMMA=float(os.environ.get("GAMMA","0.2")); EDGE_DROP=float(os.environ.get("EDGE_DROP","0.2")); SAGE_L=int(os.environ.get("SAGE_L","1"))
TABM=int(os.environ.get("TABM","16")); FEMB=int(os.environ.get("FEMB","16")); POSW_SQRT=int(os.environ.get("POSW_SQRT","1")); EPMAX=int(os.environ.get("EPMAX","600"))
LKEY={"adverse":"label_adverse","severe":"label_severe"}.get(LABEL,"label")
RKEY={"adverse":"restated_now_adverse","severe":"restated_now_severe"}.get(LABEL,"restated_now")
G=torch.load("data/ext/graph.pt",weights_only=False); YEARS=G["years"]; yidx={y:i for i,y in enumerate(YEARS)}
X=G["X"].numpy().copy(); active=G["active"].numpy(); label=G[LKEY].numpy(); rn=G[RKEY].numpy().astype(np.float32)
Tall,N,Fdim=X.shape; snaps=G["snapshots"]
AUD=["partner","office","auditor"] if MODE=="recent" else ["office","auditor"]   # auditor-only graph (drop board/ownership noise)
if MODE=="panel": USE=[yidx[y] for y in range(2005,2020)];TR=USE[:11];VA=[USE[11]];TE=USE[12:15]
else: USE=[yidx[y] for y in range(2017,2023)];TR=USE[:3];VA=[USE[3]];TE=USE[4:6]
print(f"[pleg STAGE={STAGE}] MODE={MODE} label={LABEL} B={NB} H={H} encL={ENC_DEPTH} K={KPROP} a={ALPHA} g={GAMMA} edrop={EDGE_DROP} aud={AUD} test={[YEARS[i] for i in TE]}",file=sys.stderr)
# train-stat winsorize/impute/zscore
tm=np.zeros((Tall,N),bool)
for t in TR: tm[t]=active[t]&(label[t]>=0)
Xtr0=X[tm]; lo=np.nanpercentile(Xtr0,1,0); hi=np.nanpercentile(Xtr0,99,0); med=np.nanmedian(Xtr0,0)
Xc=np.clip(X,lo,hi); ix=np.where(np.isnan(Xc)); Xc[ix]=np.take(med,ix[2])
mu=np.nanmean(np.clip(Xtr0,lo,hi),0); sd=np.nanstd(np.clip(Xtr0,lo,hi),0); sd[sd<1e-6]=1
Xz=((Xc-mu)/sd).astype(np.float32)
OWN=np.zeros((Tall,N,2),np.float32)
for t in range(Tall): OWN[t,:,0]=rn[t]; OWN[t,:,1]=rn[max(t-1,0)]
# ---- PLE: piecewise-linear quantile encoding (bins from TRAIN years only) ----
Xtr_flat=Xz[tm]                                   # (Ntrain, F) z-scored
knots=[]
for j in range(Fdim):
    q=np.quantile(Xtr_flat[:,j],np.linspace(0,1,NB+1)); q[0]=-1e9; q[-1]=1e9
    q=np.maximum.accumulate(q)                    # monotone
    knots.append(q.astype(np.float32))
def ple(Xslice):                                  # (n,F) -> (n, F*NB) piecewise-linear ramps
    parts=[]
    for j in range(Fdim):
        kb=knots[j]; x=Xslice[:,j:j+1]
        left=kb[:-1][None,:]; right=kb[1:][None,:]
        enc=np.clip((x-left)/(right-left+1e-9),0.0,1.0)
        parts.append(enc.astype(np.float32))
    return np.concatenate(parts,1)
PLE=np.zeros((Tall,N,Fdim*NB),np.float32)
for t in USE: PLE[t]=ple(Xz[t])
PLE_t=torch.from_numpy(PLE.astype(np.float32)); OWN_t=torch.from_numpy(OWN.astype(np.float32))
enc_in_dim=(Fdim*FEMB if FEMB else Fdim*NB)+2
lab_t=torch.from_numpy(label)
def adj(t,c): ei,w=snaps[t][c]; return ei,w
# merged auditor adjacency per year (row-normalized for APPNP/SAGE)
AUDADJ=[]
for t in range(Tall):
    eis=[];ews=[]
    for c in AUD:
        ei,w=adj(t,c); eis.append(ei); ews.append(w)
    ei=torch.cat(eis,1) if eis else torch.zeros((2,0),dtype=torch.long); w=torch.cat(ews) if ews else torch.zeros(0)
    AUDADJ.append((ei,w))
ytr=np.concatenate([label[t][active[t]&(label[t]>=0)] for t in TR]); spw=float((ytr==0).sum()/max((ytr==1).sum(),1))
yte=np.concatenate([label[t][active[t]&(label[t]>=0)] for t in TE]); POSW=torch.tensor([float(np.sqrt(spw)) if POSW_SQRT else spw])
def metr(y,p):
    o=np.argsort(-p);return {"roc":round(float(roc_auc_score(y,p)),4),"pr":round(float(average_precision_score(y,p)),4),"recall@10%":round(float(y[o[:max(1,len(p)//10)]].sum()/max(y.sum(),1)),4)}
def propagate(z,ei,w,K,alpha,edrop,training):
    if ei.size(1)==0: return z
    if training and edrop>0:
        keep=torch.rand(ei.size(1))>edrop; ei=ei[:,keep]; w=w[keep]
    s,d=ei; deg=scatter(w,d,0,dim_size=N,reduce='sum').clamp(min=1e-6)
    zk=z
    for _ in range(K):
        m=scatter(zk[s]*w,d,0,dim_size=N,reduce='sum')/deg
        zk=alpha*z+(1-alpha)*m
    return zk

K_ENS=max(TABM,1)
class BELinear(nn.Module):                                  # BatchEnsemble: shared W + per-member rank-1 adapters (TabM core)
    def __init__(s,din,dout):
        super().__init__(); s.W=nn.Linear(din,dout)
        s.r=nn.Parameter(torch.ones(K_ENS,din)+0.05*torch.randn(K_ENS,din))
        s.sd=nn.Parameter(torch.ones(K_ENS,dout)+0.05*torch.randn(K_ENS,dout))
        s.bm=nn.Parameter(torch.zeros(K_ENS,dout))
    def forward(s,x):                                       # x:(K,N,din)->(K,N,dout)
        return s.W(x*s.r.unsqueeze(1))*s.sd.unsqueeze(1)+s.bm.unsqueeze(1)
class PLEG(nn.Module):
    def __init__(s,drop=0.1):
        super().__init__()
        if FEMB: s.femb=nn.Linear(NB,FEMB)                  # shared per-feature numerical embedding
        s.inp=BELinear(enc_in_dim,H)
        s.ln=nn.ModuleList([nn.LayerNorm(H) for _ in range(ENC_DEPTH)])
        s.f1=nn.ModuleList([BELinear(H,2*H) for _ in range(ENC_DEPTH)])
        s.f2=nn.ModuleList([BELinear(2*H,H) for _ in range(ENC_DEPTH)])
        s.hln=nn.LayerNorm(H); s.head=BELinear(H,1)
        if STAGE in ("B","C"): s.gamma=nn.Parameter(torch.tensor(float(GAMMA)))
        if STAGE=="C":
            s.sage=nn.ModuleList([nn.Linear(2*H,H) for _ in range(SAGE_L)])
            s.head_graph=nn.Sequential(nn.LayerNorm(H),nn.Linear(H,1))
        s.drop=nn.Dropout(drop)
    def feat(s,t):
        if FEMB:
            p=PLE_t[t].view(N,Fdim,NB); p=s.femb(p).reshape(N,Fdim*FEMB)
            return torch.cat([p,OWN_t[t]],1)
        return torch.cat([PLE_t[t],OWN_t[t]],1)
    def encode(s,t):                                        # returns z_attr (mean over K members) and body h (mean)
        x=s.feat(t).unsqueeze(0).expand(K_ENS,N,-1)
        h=s.inp(x)
        for l in range(ENC_DEPTH):
            u=s.f2[l](s.drop(F.gelu(s.f1[l](s.ln[l](h))))); h=h+u
        z=s.head(s.hln(h)).squeeze(-1)                      # (K,N)
        return z.mean(0), h.mean(0)
    def forward(s,t,training=True):
        z_attr,h=s.encode(t)
        if STAGE=="A": return z_attr
        ei,w=AUDADJ[t]
        if STAGE=="B":
            return (1-s.gamma)*z_attr + s.gamma*propagate(z_attr,ei,w,KPROP,ALPHA,EDGE_DROP,training)
        hh=h; e2,w2=ei,w
        if training and EDGE_DROP>0:
            keep=torch.rand(e2.size(1))>EDGE_DROP; e2=e2[:,keep]; w2=w2[keep]
        sidx,didx=e2; deg=scatter(w2,didx,0,dim_size=N,reduce='sum').clamp(min=1e-6)
        for lin in s.sage:
            m=scatter(hh[sidx]*w2.unsqueeze(1),didx,0,dim_size=N,reduce='sum')/deg.unsqueeze(1)
            hh=F.gelu(lin(torch.cat([hh,m],1))); hh=s.drop(hh)
        z_base=z_attr+s.gamma*s.head_graph(hh).squeeze(1)
        return 0.7*z_base+0.3*propagate(z_base,ei,w,KPROP,ALPHA,EDGE_DROP,training)
def collect(m,ts,training=False):
    ys=[];ps=[]
    with torch.no_grad():
        for t in ts: a=np.where(active[t]&(label[t]>=0))[0]; o=m(t,training=False); ps.append(torch.sigmoid(o[a]).numpy()); ys.append(label[t][a])
    return np.concatenate(ys),np.concatenate(ps)
def train_one(seed):
    torch.manual_seed(seed); m=PLEG()
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=3e-5)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPMAX)
    tri={t:torch.from_numpy(np.where(active[t]&(label[t]>=0))[0]).long() for t in TR}
    best=-1;bs=None;bad=0
    for ep in range(EPMAX):
        m.train(); opt.zero_grad(); loss=0
        for t in TR:
            o=m(t,training=True); idx=tri[t]
            loss=loss+F.binary_cross_entropy_with_logits(o[idx],lab_t[t][idx].float(),pos_weight=POSW)
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5.0); opt.step(); sched.step()
        if ep%5==0:
            m.eval(); yv,pv=collect(m,VA)
            try: va=roc_auc_score(yv,pv)
            except: va=0
            if va>best+1e-4: best=va; bs={k:v.clone() for k,v in m.state_dict().items()};bad=0
            else: bad+=1
            if bad>14: break        # ~70 epochs patience
    if bs: m.load_state_dict(bs)
    m.eval(); yt,pt=collect(m,TE)
    return pt,yt
t0=time.time(); PT=[]
for sd in range(NSEED):
    pt,yt=train_one(sd); PT.append(pt); print(f"  [{STAGE} s{sd}] roc={roc_auc_score(yte,pt):.4f} pr={average_precision_score(yte,pt):.4f}",file=sys.stderr)
pbag=np.mean(PT,0)
# baselines for reference
def pool(ts,arr):
    xs=[];ys=[]
    for t in ts: a=np.where(active[t]&(label[t]>=0))[0]; xs.append(arr[t][a]); ys.append(label[t][a])
    return np.concatenate(xs),np.concatenate(ys)
def magg(v,ei,w):
    if ei.size(1)==0: return torch.zeros(N)
    s,d=ei;num=scatter((v[s]*w),d,0,dim_size=N,reduce='sum');den=scatter(w,d,0,dim_size=N,reduce='sum').clamp(min=1e-6);return num/den
REL=["partner","office","auditor","board","ownership"] if MODE=="recent" else ["office","auditor","board","ownership"]
EXP=np.zeros((Tall,N,len(REL)*2),np.float32)
for t in range(Tall):
    for ci,c in enumerate(REL):
        ei,w=adj(t,c); EXP[t,:,ci]=magg(torch.from_numpy(rn[t]),ei,w).numpy()
        tl=max(t-1,0); ei2,w2=adj(tl,c); EXP[t,:,len(REL)+ci]=magg(torch.from_numpy(rn[tl]),ei2,w2).numpy()
GF=np.concatenate([Xz,OWN,EXP],2).astype(np.float32); Xtr_g,ytr2=pool(TR,GF); Xte_g,_=pool(TE,GF)
Xtr_f,_=pool(TR,np.concatenate([Xz,OWN],2)); Xte_f,_=pool(TE,np.concatenate([Xz,OWN],2))
pRF=RandomForestClassifier(n_estimators=400,min_samples_leaf=5,class_weight="balanced_subsample",n_jobs=4,random_state=0).fit(Xtr_g,ytr2).predict_proba(Xte_g)[:,1]
pXG=xgb.XGBClassifier(n_estimators=150,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,min_child_weight=5,reg_lambda=2.0,scale_pos_weight=spw,n_jobs=4,tree_method="hist").fit(Xtr_g,ytr2).predict_proba(Xte_g)[:,1]
pRFf=RandomForestClassifier(n_estimators=400,min_samples_leaf=5,class_weight="balanced_subsample",n_jobs=4,random_state=0).fit(Xtr_f,ytr2).predict_proba(Xte_f)[:,1]
BASE={"RF+graph":pRF,"XGB+graph":pXG}; bb=max(BASE,key=lambda k:roc_auc_score(yte,BASE[k]))
def boot(pa,pb,B=3000):
    rng=np.random.default_rng(0);n=len(yte);d=[]
    for _ in range(B):
        idx=rng.integers(0,n,n)
        if 0<yte[idx].sum()<len(idx): d.append(roc_auc_score(yte[idx],pa[idx])-roc_auc_score(yte[idx],pb[idx]))
    d=np.array(d);return {"mean":round(float(d.mean()),4),"ci":[round(float(np.percentile(d,2.5)),4),round(float(np.percentile(d,97.5)),4)],"p":round(float((d>0).mean()),4)}
RES={"stage":STAGE,"mode":MODE,"label":LABEL,"knobs":{"B":NB,"H":H,"encL":ENC_DEPTH,"K":KPROP,"ALPHA":ALPHA,"GAMMA":GAMMA,"NSEED":NSEED,"EDGE_DROP":EDGE_DROP},
     "pleg":metr(yte,pbag),"pleg_seedmean":round(float(np.mean([roc_auc_score(yte,p) for p in PT])),4),"pleg_seedstd":round(float(np.std([roc_auc_score(yte,p) for p in PT])),4),
     "RF+graph":metr(yte,pRF),"XGB+graph":metr(yte,pXG),"RF_featonly":metr(yte,pRFf),
     "vs":bb,"boot_vs_"+bb:boot(pbag,BASE[bb]),"runtime_sec":round(time.time()-t0,1)}
os.makedirs("data/ext/pleg",exist_ok=True)
json.dump(RES,open(f"data/ext/pleg/stage{STAGE}_{MODE}_{LABEL}.json","w"),indent=2)
np.savez(f"data/ext/pleg/stage{STAGE}_{MODE}_{LABEL}_preds.npz",y=yte,pg=pbag,rf=pRF,xgb=pXG)
print(json.dumps({k:RES[k] for k in ["stage","pleg","pleg_seedmean","pleg_seedstd","RF+graph","RF_featonly","XGB+graph","vs","boot_vs_"+bb,"runtime_sec"]},ensure_ascii=False,indent=2))
