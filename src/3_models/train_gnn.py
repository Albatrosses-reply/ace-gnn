#!/usr/bin/env python3
"""Self-interpretable temporal multiplex GNN for restatement-contagion prediction.
Honest temporal split, baselines, ablations, interpretability. -> data/results.json"""
import json, sys, time, math, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.utils import scatter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

torch.manual_seed(0); np.random.seed(0)
DEV="cpu"
LABEL_MODE=os.environ.get("LABEL","label")   # "label"(any) | "adverse"(material) | "severe"(ICFR+fraud+AAER)
OUT=os.environ.get("OUT","data/results.json")
LKEY={"adverse":"label_adverse","severe":"label_severe"}.get(LABEL_MODE,"label")
RKEY={"adverse":"restated_now_adverse","severe":"restated_now_severe"}.get(LABEL_MODE,"restated_now")
G=torch.load("data/graph.pt", weights_only=False)
YEARS=G["years"]; FEAT=G["feat_names"]; REL=["auditor","office","board","ownership"]
X=G["X"].numpy().copy()            # (T,N,F)
active=G["active"].numpy()         # (T,N)
ffi=G["ffi"].numpy()               # (T,N)
label=G[LKEY].numpy()              # (T,N) in {-1,0,1}
restated_now=G[RKEY].numpy()
print(f"[label mode] {LABEL_MODE} -> {LKEY}",file=sys.stderr)
T,N,Fdim=X.shape
snaps=[{r:(ei.to(DEV),w.to(DEV)) for r,(ei,w) in s.items()} for s in G["snapshots"]]
print(f"[data] T={T} N={N} F={Fdim} rels={REL}",file=sys.stderr)

TRAIN_T=[0,1,2,3,4,5]; VAL_T=[6]; TEST_T=[7,8]   # 2010-15 | 2016 | 2017-18 (predict +1)

# ---------- preprocess features (TRAIN-only stats) ----------
def trainmask():
    m=np.zeros((T,N),bool)
    for t in TRAIN_T: m[t]=active[t]&(label[t]>=0)
    return m
tm=trainmask()
Xtr=X[tm]                                   # train firm-years × F
lo=np.nanpercentile(Xtr,1,axis=0); hi=np.nanpercentile(Xtr,99,axis=0)
med=np.nanmedian(Xtr,axis=0)
Xc=np.clip(X,lo,hi)
inds=np.where(np.isnan(Xc)); Xc[inds]=np.take(med,inds[2])
mu=np.nanmean(np.clip(Xtr,lo,hi),axis=0); sd=np.nanstd(np.clip(Xtr,lo,hi),axis=0); sd[sd<1e-6]=1
Xz=((Xc-mu)/sd).astype(np.float32)
Xz=torch.from_numpy(Xz).to(DEV)
ind_idx=np.clip(ffi,-1,None)+1              # -1->0 unknown, 1..12 -> 2..13
nIND=int(ind_idx.max())+1
ind_t=torch.from_numpy(ind_idx).long().to(DEV)
lab_t=torch.from_numpy(label).long().to(DEV)
act_t=torch.from_numpy(active).to(DEV)

def mask_for(ts):
    idx=[]
    for t in ts:
        a=np.where(active[t]&(label[t]>=0))[0]
        idx.append((t,torch.from_numpy(a).long()))
    return idx
TR=mask_for(TRAIN_T); VA=mask_for(VAL_T); TE=mask_for(TEST_T)
pos=sum(int((label[t][active[t]&(label[t]>=0)]==1).sum()) for t in TRAIN_T)
neg=sum(int((label[t][active[t]&(label[t]>=0)]==0).sum()) for t in TRAIN_T)
POSW=torch.tensor([neg/max(pos,1)],device=DEV); print(f"[train] pos={pos} neg={neg} posw={POSW.item():.2f}",file=sys.stderr)

# ---------- message passing ----------
def agg(x, ei, w):
    if ei.size(1)==0: return torch.zeros_like(x)
    src,dst=ei; m=x[src]*w.unsqueeze(1)
    num=scatter(m,dst,dim=0,dim_size=x.size(0),reduce='sum')
    den=scatter(w,dst,dim=0,dim_size=x.size(0),reduce='sum').clamp(min=1e-6)
    return num/den.unsqueeze(1)

class MultiplexLayer(nn.Module):
    def __init__(s,din,dout,rels):
        super().__init__(); s.rels=rels
        s.self_lin=nn.Linear(din,dout)
        s.rel_lin=nn.ModuleDict({r:nn.Linear(din,dout) for r in rels})
        s.att=nn.Linear(dout,1,bias=False)
    def forward(s,x,snap):
        reps=[s.self_lin(x)]; keys=["self"]
        for r in s.rels:
            ei,w=snap[r]; reps.append(agg(s.rel_lin[r](x),ei,w)); keys.append(r)
        H=torch.stack(reps,1)                       # N×R×d
        a=torch.softmax(s.att(torch.tanh(H)).squeeze(-1),dim=1)  # N×R
        out=(a.unsqueeze(-1)*H).sum(1)
        return F.relu(out), a, keys

class TempMultiGNN(nn.Module):
    def __init__(s,Fdim,H=64,rels=REL,temporal=True,drop=0.3):
        super().__init__(); s.temporal=temporal; s.rels=rels
        s.enc=nn.Linear(Fdim,H); s.ind=nn.Embedding(nIND,H)
        s.l1=MultiplexLayer(H,H,rels); s.l2=MultiplexLayer(H,H,rels)
        s.gru=nn.GRUCell(H,H); s.drop=nn.Dropout(drop); s.head=nn.Linear(H,1)
    def forward(s,upto):
        sh=torch.zeros(N,s.gru.hidden_size,device=DEV); out={}; att_acc={}
        for t in range(upto+1):
            h=F.relu(s.enc(Xz[t])+s.ind(ind_t[t]))
            h,a1,keys=s.l1(h,snaps[t]); h=s.drop(h)
            h,a2,_=s.l2(h,snaps[t])
            sh=s.gru(h,sh) if s.temporal else h
            out[t]=s.head(s.drop(sh)).squeeze(-1)
            att_acc[t]=(a1.detach(),a2.detach(),keys)
        return out,att_acc

def evaluate(scores_by_t, splits):
    ys=[]; ps=[]
    per={}
    for t,idx in splits:
        sc=scores_by_t[t][idx].detach().cpu().numpy()
        yy=label[t][idx.cpu().numpy()]
        ys.append(yy); ps.append(sc)
        if len(np.unique(yy))>1:
            per[YEARS[t]+1]={"n":int(len(yy)),"pos":int(yy.sum()),
                "roc":float(roc_auc_score(yy,sc)),"pr":float(average_precision_score(yy,sc))}
    y=np.concatenate(ys); p=np.concatenate(ps)
    order=np.argsort(-p)
    def rec_at(k):
        kk=max(1,int(len(p)*k)); top=order[:kk]
        return float(y[top].sum()/max(y.sum(),1))
    return {"roc":float(roc_auc_score(y,p)),"pr":float(average_precision_score(y,p)),
            "base_rate":float(y.mean()),"recall@5%":rec_at(.05),"recall@10%":rec_at(.10),
            "n":int(len(y)),"pos":int(y.sum()),"per_year":per}

def train_model(rels=REL,temporal=True,epochs=300,lr=5e-3,wd=1e-4,patience=40,tag=""):
    m=TempMultiGNN(Fdim,64,rels,temporal).to(DEV)
    opt=torch.optim.Adam(m.parameters(),lr=lr,weight_decay=wd)
    best=-1; best_state=None; bad=0
    upto_tr=max(TRAIN_T); upto_ev=max(TEST_T)
    for ep in range(epochs):
        m.train(); opt.zero_grad()
        out,_=m(upto_tr); loss=0
        for t,idx in TR:
            loss=loss+F.binary_cross_entropy_with_logits(out[t][idx],lab_t[t][idx].float(),pos_weight=POSW)
        loss.backward(); opt.step()
        if ep%5==0 or ep==epochs-1:
            m.eval()
            with torch.no_grad():
                out,_=m(upto_ev); sc={t:out[t] for t in out}
                va=evaluate(sc,VA)["pr"]
            if va>best: best=va; best_state={k:v.detach().clone() for k,v in m.state_dict().items()}; bad=0
            else: bad+=1
            if bad>patience//5: break
    if best_state: m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        out,att=m(max(TEST_T)); sc={t:out[t] for t in out}
    res={"val_pr":best,"test":evaluate(sc,TE),"val":evaluate(sc,VA)}
    return m,sc,att,res

# ---------- baselines: LR / MLP (features only) ----------
def pooled(ts):
    xs=[]; ys=[]
    for t in ts:
        a=np.where(active[t]&(label[t]>=0))[0]
        xs.append(Xz[t][a].cpu().numpy()); ys.append(label[t][a])
    return np.concatenate(xs),np.concatenate(ys)
Xtr_p,ytr_p=pooled(TRAIN_T); Xva_p,yva_p=pooled(VAL_T); Xte_p,yte_p=pooled(TEST_T)
lr=LogisticRegression(max_iter=2000,class_weight="balanced").fit(Xtr_p,ytr_p)
pte=lr.predict_proba(Xte_p)[:,1]
LR_RES={"roc":float(roc_auc_score(yte_p,pte)),"pr":float(average_precision_score(yte_p,pte)),
        "base_rate":float(yte_p.mean()),"n":int(len(yte_p)),"pos":int(yte_p.sum())}
order=np.argsort(-pte)
LR_RES["recall@10%"]=float(yte_p[order[:max(1,len(pte)//10)]].sum()/max(yte_p.sum(),1))
print(f"[LR] test ROC={LR_RES['roc']:.3f} PR={LR_RES['pr']:.3f}",file=sys.stderr)

# XGBoost (features only) — strong non-linear tabular baseline
import xgboost as xgb
spw=float((ytr_p==0).sum()/max((ytr_p==1).sum(),1))
xgbm=xgb.XGBClassifier(n_estimators=400,max_depth=4,learning_rate=0.04,subsample=0.8,
    colsample_bytree=0.8,min_child_weight=5,reg_lambda=2.0,scale_pos_weight=spw,
    eval_metric="aucpr",n_jobs=4,tree_method="hist")
xgbm.fit(Xtr_p,ytr_p)
ptx=xgbm.predict_proba(Xte_p)[:,1]
ox=np.argsort(-ptx)
XGB_RES={"roc":float(roc_auc_score(yte_p,ptx)),"pr":float(average_precision_score(yte_p,ptx)),
         "base_rate":float(yte_p.mean()),"n":int(len(yte_p)),"pos":int(yte_p.sum()),
         "recall@5%":float(yte_p[ox[:max(1,len(ptx)//20)]].sum()/max(yte_p.sum(),1)),
         "recall@10%":float(yte_p[ox[:max(1,len(ptx)//10)]].sum()/max(yte_p.sum(),1))}
print(f"[XGB] test ROC={XGB_RES['roc']:.3f} PR={XGB_RES['pr']:.3f} (spw={spw:.1f})",file=sys.stderr)

# ---------- contagion: descriptive ----------
def neighbor_restated(t,rel):
    ei,w=snaps[t][rel]
    if ei.size(1)==0: return np.zeros(N)
    rn=torch.from_numpy(restated_now[t].astype(np.float32)).to(DEV)
    src,dst=ei; cnt=scatter(rn[src],dst,dim=0,dim_size=N,reduce='sum')
    return cnt.cpu().numpy()
contagion={}
for rel in REL:
    rows=[]
    for t in TEST_T:
        a=active[t]&(label[t]>=0); nb=neighbor_restated(t,rel)
        exp=(nb>0)&a; nexp=(nb==0)&a
        rows.append((int(exp.sum()),float(label[t][exp].mean() if exp.sum() else float('nan')),
                     int(nexp.sum()),float(label[t][nexp].mean() if nexp.sum() else float('nan'))))
    ne=sum(r[0] for r in rows); pe=np.nanmean([r[1] for r in rows])
    nn_=sum(r[2] for r in rows); pn=np.nanmean([r[3] for r in rows])
    contagion[rel]={"n_with_restating_neighbor":ne,"P(restate_t+1|exposed)":round(float(pe),4),
                    "n_without":nn_,"P(restate_t+1|not_exposed)":round(float(pn),4),
                    "lift":round(float(pe/pn),2) if pn and not math.isnan(pn) else None}
print("[contagion]",json.dumps(contagion,ensure_ascii=False),file=sys.stderr)

# ---------- run models ----------
RESULTS={"config":{"years":YEARS,"train":[YEARS[t] for t in TRAIN_T],"val":[YEARS[t] for t in VAL_T],
                   "test":[YEARS[t] for t in TEST_T],"features":FEAT,"relations":REL,
                   "params":G["params"]},
         "label_pos_rate":{int(YEARS[t]+1):round(float(label[t][active[t]&(label[t]>=0)].mean()),4) for t in range(T-1)},
         "baselines":{"logistic_features_only":LR_RES,"xgboost_features_only":XGB_RES},
         "contagion_descriptive":contagion,"models":{}}

t0=time.time()
specs=[("temporal_multiplex_FULL",dict(rels=REL,temporal=True)),
       ("static_multiplex_noGRU",dict(rels=REL,temporal=False)),
       ("temporal_auditor_only",dict(rels=["auditor"],temporal=True)),
       ("temporal_office_only",dict(rels=["office"],temporal=True)),
       ("temporal_board_only",dict(rels=["board"],temporal=True)),
       ("temporal_ownership_only",dict(rels=["ownership"],temporal=True)),
       ("temporal_auditor_office",dict(rels=["auditor","office"],temporal=True))]
saved_att=None
for name,kw in specs:
    m,sc,att,res=train_model(tag=name,**kw)
    RESULTS["models"][name]=res
    print(f"[{name}] val_pr={res['val_pr']:.3f} TEST roc={res['test']['roc']:.3f} pr={res['test']['pr']:.3f} "
          f"r@10%={res['test']['recall@10%']:.3f}",file=sys.stderr)
    if name=="temporal_multiplex_FULL":
        # relation attention (avg over active test nodes, layer1)
        ratt={}
        for t in TEST_T:
            a1,a2,keys=att[t]; a=active[t]&(label[t]>=0)
            ratt[int(YEARS[t])]={k:round(float(a1[a][:,i].mean()),4) for i,k in enumerate(keys)}
        RESULTS["relation_attention_layer1"]=ratt
        # model-based contagion: predicted risk by exposure
        sc_np=torch.sigmoid(sc[TEST_T[0]]).detach().cpu().numpy()
RESULTS["runtime_sec"]=round(time.time()-t0,1)
RESULTS["label_mode"]=LABEL_MODE
json.dump(RESULTS,open(OUT,"w"),indent=2,ensure_ascii=False)
print(f"[DONE {RESULTS['runtime_sec']}s -> {OUT}]",file=sys.stderr)
# console summary
print("\n=== TEST (pooled 2018-2019) ===")
print(f"{'model':32s} {'ROC':>6s} {'PR-AUC':>7s} {'rec@10%':>8s}")
print(f"{'logistic (features only)':32s} {LR_RES['roc']:6.3f} {LR_RES['pr']:7.3f} {LR_RES['recall@10%']:8.3f}")
print(f"{'xgboost (features only)':32s} {XGB_RES['roc']:6.3f} {XGB_RES['pr']:7.3f} {XGB_RES['recall@10%']:8.3f}")
for name in RESULTS["models"]:
    r=RESULTS["models"][name]["test"]; print(f"{name:32s} {r['roc']:6.3f} {r['pr']:7.3f} {r['recall@10%']:8.3f}")
print(f"\nbase rate (test) = {LR_RES['base_rate']:.3f}")
print("\n=== CONTAGION (descriptive) ===")
for rel,c in contagion.items():
    print(f"  {rel:10s}: P(restate|exposed)={c['P(restate_t+1|exposed)']} vs not={c['P(restate_t+1|not_exposed)']} lift={c['lift']}")
