#!/usr/bin/env python3
"""15-baseline benchmark vs ACE-GNN ensemble (loaded from saved npz), identical split/test set.
Families: tabular(feat), tabular(feat+graph-exposure), standard GNNs(merged multiplex). Leaderboard
+ paired bootstrap of ACE's ROC gain vs EVERY baseline. ENV: MODE, LABEL."""
import json, sys, os, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.utils import scatter
import torch_geometric.nn as pygnn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb, lightgbm as lgb, catboost as cb
torch.manual_seed(0); np.random.seed(0); DEV="cpu"
MODE=os.environ.get("MODE","recent"); LABEL_MODE=os.environ.get("LABEL","severe")
OUT=os.environ.get("OUT",f"data/ext/bench_{MODE}_{LABEL_MODE}.json")
LKEY={"adverse":"label_adverse","severe":"label_severe"}.get(LABEL_MODE,"label")
RKEY={"adverse":"restated_now_adverse","severe":"restated_now_severe"}.get(LABEL_MODE,"restated_now")
G=torch.load("data/ext/graph.pt", weights_only=False)
YEARS=G["years"]; FEAT=G["feat_names"]; yidx={y:i for i,y in enumerate(YEARS)}
X=G["X"].numpy().copy(); active=G["active"].numpy(); label=G[LKEY].numpy(); rn=G[RKEY].numpy().astype(np.float32)
Tall,N,Fdim=X.shape; snaps=G["snapshots"]
REL=["partner","office","auditor","board","ownership"] if MODE=="recent" else ["office","auditor","board","ownership"]
if MODE=="panel": USE=[yidx[y] for y in range(2005,2020)]; TR=USE[:11]; VA=[USE[11]]; TE=USE[12:15]
else: USE=[yidx[y] for y in range(2017,2023)]; TR=USE[:3]; VA=[USE[3]]; TE=USE[4:6]
print(f"[bench] MODE={MODE} label={LABEL_MODE} test={[YEARS[i] for i in TE]} rels={REL}",file=sys.stderr)

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
EXP=np.zeros((Tall,N,len(REL)*2),np.float32)
for t in range(Tall):
    for ci,c in enumerate(REL):
        ei,w=adj(t,c); EXP[t,:,ci]=mean_agg(rnt[t],ei,w).numpy()
        tl=max(t-1,0); eil,wl=adj(tl,c); EXP[t,:,len(REL)+ci]=mean_agg(rnt[tl],eil,wl).numpy()
OWN=np.zeros((Tall,N,2),np.float32)
for t in range(Tall): OWN[t,:,0]=rn[t]; OWN[t,:,1]=rn[max(t-1,0)]
GF=np.concatenate([Xz,OWN,EXP],2).astype(np.float32)
def pool(ts,arr):
    xs=[];ys=[]
    for t in ts:
        a=np.where(active[t]&(label[t]>=0))[0]; xs.append(arr[t][a]); ys.append(label[t][a])
    return np.concatenate(xs),np.concatenate(ys)
Xtr_f,ytr=pool(TR,Xz); Xte_f,yte=pool(TE,Xz); Xva_f,yva=pool(VA,Xz)
Xtr_g,_=pool(TR,GF); Xte_g,_=pool(TE,GF)
spw=float((ytr==0).sum()/max((ytr==1).sum(),1))
def metr(y,p):
    o=np.argsort(-p); return {"roc":round(float(roc_auc_score(y,p)),4),"pr":round(float(average_precision_score(y,p)),4),
      "recall@10%":round(float(y[o[:max(1,len(p)//10)]].sum()/max(y.sum(),1)),4)}
PREDS={}; RES={"mode":MODE,"label":LABEL_MODE,"test_years":[YEARS[i] for i in TE],"n_test":int(len(yte)),"pos_test":int(yte.sum()),"leaderboard":{}}

# ---------- tabular baselines (features only) ----------
def add(name,p): PREDS[name]=p; RES["leaderboard"][name]=metr(yte,p)
add("LogisticRegression", LogisticRegression(max_iter=2000,class_weight="balanced").fit(Xtr_f,ytr).predict_proba(Xte_f)[:,1])
add("RandomForest", RandomForestClassifier(n_estimators=400,max_depth=None,min_samples_leaf=5,class_weight="balanced_subsample",n_jobs=4,random_state=0).fit(Xtr_f,ytr).predict_proba(Xte_f)[:,1])
add("ExtraTrees", ExtraTreesClassifier(n_estimators=400,min_samples_leaf=5,class_weight="balanced_subsample",n_jobs=4,random_state=0).fit(Xtr_f,ytr).predict_proba(Xte_f)[:,1])
add("HistGradBoosting", HistGradientBoostingClassifier(max_depth=4,learning_rate=0.05,l2_regularization=2.0,class_weight="balanced",random_state=0).fit(Xtr_f,ytr).predict_proba(Xte_f)[:,1])
add("LightGBM", lgb.LGBMClassifier(n_estimators=400,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,reg_lambda=2.0,min_child_samples=20,scale_pos_weight=spw,n_jobs=4,verbose=-1,random_state=0).fit(Xtr_f,ytr).predict_proba(Xte_f)[:,1])
add("CatBoost", cb.CatBoostClassifier(iterations=400,depth=4,learning_rate=0.05,l2_leaf_reg=3.0,scale_pos_weight=spw,verbose=0,random_seed=0).fit(Xtr_f,ytr).predict_proba(Xte_f)[:,1])
def mkxgb(n): return xgb.XGBClassifier(n_estimators=n,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,min_child_weight=5,reg_lambda=2.0,scale_pos_weight=spw,eval_metric="aucpr",n_jobs=4,tree_method="hist")
add("XGBoost", mkxgb(400).fit(Xtr_f,ytr).predict_proba(Xte_f)[:,1])
add("MLP", MLPClassifier(hidden_layer_sizes=(128,64),alpha=1e-3,max_iter=300,random_state=0).fit(Xtr_f,ytr).predict_proba(Xte_f)[:,1])
# ---------- tabular + graph-exposure features ----------
add("LogReg+graph", LogisticRegression(max_iter=2000,class_weight="balanced").fit(Xtr_g,ytr).predict_proba(Xte_g)[:,1])
add("RandomForest+graph", RandomForestClassifier(n_estimators=400,min_samples_leaf=5,class_weight="balanced_subsample",n_jobs=4,random_state=0).fit(Xtr_g,ytr).predict_proba(Xte_g)[:,1])
add("XGBoost+graph", mkxgb(150).fit(Xtr_g,ytr).predict_proba(Xte_g)[:,1])
print("[tabular done]",{k:RES["leaderboard"][k]["roc"] for k in RES["leaderboard"]},file=sys.stderr)

# ---------- standard GNN baselines (merged multiplex, node feat=[Xz,own]) ----------
NF=np.concatenate([Xz,OWN],2).astype(np.float32); NF_t=torch.from_numpy(NF); lab_t=torch.from_numpy(label)
nfeat=NF.shape[2]
merged=[]; rtype=[]
for t in range(Tall):
    eis=[];ews=[];ets=[]
    for ri,c in enumerate(REL):
        ei,w=adj(t,c); eis.append(ei); ews.append(w); ets.append(torch.full((ei.size(1),),ri,dtype=torch.long))
    merged.append((torch.cat(eis,1) if eis else torch.zeros((2,0),dtype=torch.long), torch.cat(ews) if ews else torch.zeros(0)))
    rtype.append(torch.cat(ets) if ets else torch.zeros(0,dtype=torch.long))
class GNNB(nn.Module):
    def __init__(s,kind,H=64,drop=0.4):
        super().__init__(); s.kind=kind; s.drop=nn.Dropout(drop)
        if kind=="GCN": s.c1=pygnn.GCNConv(nfeat,H,add_self_loops=True); s.c2=pygnn.GCNConv(H,H,add_self_loops=True)
        elif kind=="SAGE": s.c1=pygnn.SAGEConv(nfeat,H); s.c2=pygnn.SAGEConv(H,H)
        elif kind=="GAT": s.c1=pygnn.GATConv(nfeat,H//4,heads=4,dropout=drop); s.c2=pygnn.GATConv(H,H//4,heads=4,dropout=drop)
        elif kind=="APPNP": s.l1=nn.Linear(nfeat,H); s.l2=nn.Linear(H,H); s.prop=pygnn.APPNP(K=5,alpha=0.15)
        elif kind=="RGCN": s.c1=pygnn.RGCNConv(nfeat,H,len(REL)); s.c2=pygnn.RGCNConv(H,H,len(REL))
        s.head=nn.Linear(H,1)
    def forward(s,t):
        ei,ew=merged[t]; x=NF_t[t]
        if s.kind=="GCN": h=F.relu(s.c1(x,ei,ew)); h=s.drop(h); h=F.relu(s.c2(h,ei,ew))
        elif s.kind=="SAGE": h=F.relu(s.c1(x,ei)); h=s.drop(h); h=F.relu(s.c2(h,ei))
        elif s.kind=="GAT": h=F.elu(s.c1(x,ei)); h=s.drop(h); h=F.elu(s.c2(h,ei))
        elif s.kind=="APPNP": h=F.relu(s.l1(x)); h=s.drop(h); h=s.l2(h); h=s.prop(h,ei,ew)
        elif s.kind=="RGCN": h=F.relu(s.c1(x,ei,rtype[t])); h=s.drop(h); h=F.relu(s.c2(h,ei,rtype[t]))
        return s.head(h).squeeze(1)
POSW=torch.tensor([spw])
def train_gnn(kind):
    torch.manual_seed(0); m=GNNB(kind); opt=torch.optim.Adam(m.parameters(),lr=5e-3,weight_decay=5e-4)
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
    m.eval(); ys=[];ps=[]
    with torch.no_grad():
        for t in TE: o=m(t); a=np.where(active[t]&(label[t]>=0))[0]; ps.append(torch.sigmoid(o[a]).numpy()); ys.append(label[t][a])
    return np.concatenate(ps),np.concatenate(ys)
for kind in ["GCN","SAGE","GAT","APPNP","RGCN"]:
    p,yc=train_gnn(kind); assert np.array_equal(yc,yte); add(f"{kind}",p); print(f"[{kind}] roc={RES['leaderboard'][kind]['roc']}",file=sys.stderr)

# ---------- our ACE-GNN ensemble (load saved preds; verify alignment) ----------
npz=f"data/ext/ace3_{MODE}_{LABEL_MODE}_preds.npz"
if not os.path.exists(npz): npz=f"data/ext/ace2_{MODE}_{LABEL_MODE}_preds.npz"
if os.path.exists(npz):
    d=np.load(npz); assert np.array_equal(d["y"],yte), "test-set misalignment with saved ACE preds"
    add("ACE-GNN (ours)", d["ens"])
    RES["leaderboard"]["ACE-GNN (ours)"]["is_ours"]=True
else:
    print("[WARN] ACE preds npz missing",file=sys.stderr)

# ---------- leaderboard + bootstrap: ACE gain vs every baseline ----------
lb=sorted(RES["leaderboard"].items(), key=lambda kv:-kv[1]["roc"])
RES["ranking"]=[k for k,_ in lb]
if "ACE-GNN (ours)" in PREDS:
    pace=PREDS["ACE-GNN (ours)"]; rng=np.random.default_rng(0); n=len(yte)
    boot={}
    for name,p in PREDS.items():
        if name=="ACE-GNN (ours)": continue
        diffs=[]
        for _ in range(2000):
            idx=rng.integers(0,n,n)
            if 0<yte[idx].sum()<len(idx): diffs.append(roc_auc_score(yte[idx],pace[idx])-roc_auc_score(yte[idx],p[idx]))
        diffs=np.array(diffs)
        boot[name]={"gain":round(float(diffs.mean()),4),"ci95":[round(float(np.percentile(diffs,2.5)),4),round(float(np.percentile(diffs,97.5)),4)],"p_gt_0":round(float((diffs>0).mean()),3)}
    RES["ace_gain_vs_baseline"]=boot
json.dump(RES,open(OUT,"w"),indent=2,ensure_ascii=False)
print(f"\n=== {MODE}/{LABEL_MODE} leaderboard (test {[YEARS[i] for i in TE]}) ===")
for rank,(k,v) in enumerate(lb,1):
    g=RES.get("ace_gain_vs_baseline",{}).get(k)
    tag=" <-- OURS" if v.get("is_ours") else (f"  ACEΔ={g['gain']:+.4f} p={g['p_gt_0']}" if g else "")
    print(f"{rank:2d}. {k:22s} ROC={v['roc']:.4f} PR={v['pr']:.4f} r@10%={v['recall@10%']:.3f}{tag}")
print(f"[done -> {OUT}]")
