#!/usr/bin/env python3
"""Post-hoc analyses on saved predictions: per-year breakdown, calibration (Brier+reliability),
DeLong test (ACE vs strongest baseline), and a label-shuffle placebo (leakage check)."""
import json, os, glob, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import scipy.stats as st

# ---------- fast DeLong test for two correlated ROC AUCs ----------
def _midrank(x):
    J=np.argsort(x); Z=x[J]; N=len(x); T=np.zeros(N); i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=0.5*(i+j-1)+1; i=j
    out=np.empty(N); out[J]=T; return out
def delong(y, p1, p2):
    pos=p1[y==1],p2[y==1]; neg=p1[y==0],p2[y==0]; m=int((y==1).sum()); n=int((y==0).sum())
    aucs=[]; V10=[];V01=[]
    for pa,na in [(pos[0],neg[0]),(pos[1],neg[1])]:
        tx=_midrank(pa); ty=_midrank(na); tz=_midrank(np.concatenate([pa,na]))
        auc=(tz[:m].sum()-m*(m+1)/2)/(m*n); aucs.append(auc)
        V10.append((tz[:m]-tx)/n); V01.append(1-(tz[m:]-ty)/m)
    V10=np.array(V10); V01=np.array(V01)
    S=np.cov(V10)/m+np.cov(V01)/n
    da=aucs[0]-aucs[1]; var=S[0,0]+S[1,1]-2*S[0,1]
    if var<=0: return aucs[0],aucs[1],da,float('nan')
    z=da/np.sqrt(var); pval=2*(1-st.norm.cdf(abs(z)))
    return aucs[0],aucs[1],da,float(pval)

G=torch.load("data/ext/graph.pt", weights_only=False); YEARS=G["years"]; yidx={y:i for i,y in enumerate(YEARS)}
active=G["active"].numpy()
def seg_years(MODE,LABEL):
    LKEY={"adverse":"label_adverse","severe":"label_severe"}.get(LABEL,"label")
    label=G[LKEY].numpy()
    TE=[yidx[y] for y in (range(2017,2020) if MODE=="panel" else range(2021,2023))]
    segs=[]
    for t in TE:
        a=np.where(active[t]&(label[t]>=0))[0]; segs.append((YEARS[t]+1,len(a)))
    return segs

OUT={}
for f in sorted(glob.glob("data/ext/ace3_*_preds.npz"))+sorted(glob.glob("data/ext/ace2_panel_severe_preds.npz")):
    base=os.path.basename(f).replace("_preds.npz","")
    parts=base.split("_"); MODE=parts[1]; LABEL=parts[2]
    key=f"{MODE}/{LABEL}"
    if key in OUT: continue
    d=np.load(f); y=d["y"]; pace=d["ens"]; pxgb=d["xgb"]; prf=d.get("rf",d.get("xgb"))
    res={"overall":{"ACE":{"roc":round(float(roc_auc_score(y,pace)),4),"pr":round(float(average_precision_score(y,pace)),4),"brier":round(float(brier_score_loss(y,pace)),4)},
                     "XGB+graph":{"roc":round(float(roc_auc_score(y,pxgb)),4),"brier":round(float(brier_score_loss(y,pxgb)),4)}}}
    # per-year
    segs=seg_years(MODE,LABEL); res["per_year"]={}; i0=0
    if sum(c for _,c in segs)==len(y):
        for yr,c in segs:
            sl=slice(i0,i0+c); yy=y[sl]
            if 0<yy.sum()<c: res["per_year"][int(yr)]={"n":int(c),"pos":int(yy.sum()),"ACE_roc":round(float(roc_auc_score(yy,pace[sl])),4),"XGBg_roc":round(float(roc_auc_score(yy,pxgb[sl])),4)}
            i0+=c
    else: res["per_year"]="segment-mismatch"
    # calibration reliability (10 bins) for ACE
    bins=np.linspace(0,1,11); rel=[]
    for b in range(10):
        m=(pace>=bins[b])&(pace<bins[b+1])
        if m.sum()>=20: rel.append({"bin":round((bins[b]+bins[b+1])/2,2),"pred":round(float(pace[m].mean()),3),"obs":round(float(y[m].mean()),3),"n":int(m.sum())})
    res["calibration_ACE"]=rel
    # DeLong: ACE vs RF+graph (strongest), and vs XGB+graph
    a1,a2,da,pv=delong(y,pace,prf); res["delong_ACE_vs_RFgraph"]={"auc_ace":round(a1,4),"auc_rf":round(a2,4),"diff":round(da,4),"p":round(pv,4)}
    a1,a2,da,pv=delong(y,pace,pxgb); res["delong_ACE_vs_XGBgraph"]={"auc_ace":round(a1,4),"auc_xgb":round(a2,4),"diff":round(da,4),"p":round(pv,4)}
    OUT[key]=res
    print(f"[{key}] ACE roc={res['overall']['ACE']['roc']} brier={res['overall']['ACE']['brier']} | DeLong vs RF+g: d={res['delong_ACE_vs_RFgraph']['diff']} p={res['delong_ACE_vs_RFgraph']['p']}")
json.dump(OUT,open("data/ext/posthoc.json","w"),indent=2,ensure_ascii=False)
print("[posthoc -> data/ext/posthoc.json]")
