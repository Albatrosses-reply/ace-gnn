#!/usr/bin/env python3
"""Paired bootstrap of ACE-GNN against the strongest fair tabular learner (Table 7 / Sec. 6.3).
The same resampled test firm-years are scored under both models, so the comparison is genuinely paired.
Reads   data/ext/pure/v2/FINAL_L_{MODE}_{LABEL}_preds.npz        (ace_v2.py; keys y, p = seed-bagged test predictions)
        data/ext/pure/tabular_fair_{MODE}_{LABEL}_preds.npz      (tabular_fair.py; keys y, <model>)
Writes  data/ext/pure/v2/boot_FINALL_vs_fair{TAG}_{MODE}_{LABEL}.json with
        d_aupr / d_roc: {mean, ci (2.5/97.5 pct), p_gt0}  for  ACE minus baseline.
ENV: MODE=recent|panel  LABEL=severe|label  BASELINE=RandomForest|HistGradBoosting|...  TAG=RF|HG  B=3000  SEED=0
Run from repo root:  MODE=recent LABEL=severe BASELINE=RandomForest TAG=RF python3 src/4_analysis/paired_bootstrap.py
"""
import json, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
MODE=os.environ.get("MODE","recent"); LABEL=os.environ.get("LABEL","severe")
BASE=os.environ.get("BASELINE","RandomForest"); TAG=os.environ.get("TAG",{"RandomForest":"RF","HistGradBoosting":"HG"}.get(BASE,BASE))
B=int(os.environ.get("B","3000")); SEED=int(os.environ.get("SEED","0"))
ace=np.load(f"data/ext/pure/v2/FINAL_L_{MODE}_{LABEL}_preds.npz"); fair=np.load(f"data/ext/pure/tabular_fair_{MODE}_{LABEL}_preds.npz")
y=ace["y"]; assert np.array_equal(y,fair["y"]), "test-set misalignment between ACE and fair-tabular predictions"
pa=ace["p"]; pb=fair[BASE.replace(" ","_")]
rng=np.random.default_rng(SEED); n=len(y); d_pr=[]; d_roc=[]
for _ in range(B):
    idx=rng.integers(0,n,n); yi=y[idx]
    if 0<yi.sum()<n:
        d_pr.append(average_precision_score(yi,pa[idx])-average_precision_score(yi,pb[idx]))
        d_roc.append(roc_auc_score(yi,pa[idx])-roc_auc_score(yi,pb[idx]))
def summ(d):
    d=np.array(d); return {"mean":float(d.mean()),"ci":[float(np.percentile(d,2.5)),float(np.percentile(d,97.5))],"p_gt0":float((d>0).mean())}
out={"d_aupr":summ(d_pr),"d_roc":summ(d_roc),"baseline":BASE,"B":len(d_pr),"seed":SEED,"n_test":int(n)}
OUT=os.environ.get("OUT",f"data/ext/pure/v2/boot_FINALL_vs_fair{TAG}_{MODE}_{LABEL}.json")
json.dump(out,open(OUT,"w"),indent=2)
print(json.dumps(out),file=sys.stderr); print(f"[done -> {OUT}]",file=sys.stderr)
