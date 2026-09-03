#!/usr/bin/env python3
"""Assemble the pure-model leaderboard used by Table 6 (make_tables.py) for one setting from the saved
component results, without retraining anything:
  tabular (fair, [Xz,OWN] features)  <- data/ext/pure/tabular_fair_{MODE}_{LABEL}.json      (tabular_fair.py)
  standard GNNs (GCN/SAGE/GAT/APPNP/RGCN) <- data/ext/bench_{MODE}_{LABEL}.json               (benchmark.py)
  GSAT                                <- data/ext/gsat_{MODE}_{LABEL}.json                    (gsat_experiment.py)
  ACE-GNN (ours)                      <- data/ext/pure/v2/FINAL_L_{MODE}_{LABEL}.json test_bag (ace_v2.py, defaults)
Writes data/ext/pure/leaderboard_{SETTING}.json where SETTING = recent_severe | panel_severe | recent_any.
ENV: MODE=recent|panel  LABEL=severe|label
Run from repo root:  MODE=recent LABEL=severe python3 src/4_analysis/assemble_leaderboard.py
"""
import json, os, sys
MODE=os.environ.get("MODE","recent"); LABEL=os.environ.get("LABEL","severe")
SETTING=f"{MODE}_{'any' if LABEL=='label' else LABEL}"
def load(p):
    if not os.path.exists(p): sys.exit(f"missing {p}")
    return json.load(open(p))
fair=load(f"data/ext/pure/tabular_fair_{MODE}_{LABEL}.json")
_b=[f"data/ext/bench_{MODE}_{LABEL}.json", f"data/ext/bench_{SETTING}.json"]   # benchmark.py names the 'any' label file bench_recent_any.json
bench=load(next((p for p in _b if os.path.exists(p)), _b[0]))
_g=[f"data/ext/gsat_{MODE}_{LABEL}.json", f"data/ext/gsat_{SETTING}.json"]      # same naming quirk for gsat_experiment.py
gsat=load(next((p for p in _g if os.path.exists(p)), _g[0]))
ace=load(f"data/ext/pure/v2/FINAL_L_{MODE}_{LABEL}.json")
assert fair["n_test"]==bench["n_test"] and fair["pos_test"]==bench["pos_test"], "test sets differ between component files"
lb={}
for k in ["LogisticRegression","RandomForest","ExtraTrees","HistGradBoosting","LightGBM","CatBoost","XGBoost","MLP"]:
    lb[k]=dict(fair["leaderboard"][k])
for k in ["GCN","SAGE","GAT","APPNP","RGCN"]:
    lb[k]=dict(bench["leaderboard"][k])
lb["GSAT"]={m:gsat[m] for m in ["roc","pr","recall@10%"]}
lb["ACE-GNN (ours)"]={m:ace["test_bag"][m] for m in ["roc","pr","recall@10%"]}; lb["ACE-GNN (ours)"]["is_ours"]=True
out={"setting":SETTING,"test_years":bench["test_years"],"n_test":bench["n_test"],"pos_test":bench["pos_test"],
     "primary_metric":"AUPRC (pr) + recall@10%  [ROC reported but secondary: class prevalence ~ rare-event]",
     "model_family":["pure_tabular","standard_GNN","self_interpretable_GNN","ours"],
     "excluded_synthetic":["ACE-GNN (ours)","LogReg+graph","RandomForest+graph","XGBoost+graph"],
     "leaderboard_pure":lb,
     "ace_source":f"ace_v2.py FINAL_L ({'PLE+EXPENC+MP, no collective' if not ace['knobs'].get('COLLECTIVE') else 'with collective'}; {ace['knobs']['NSEED']}-seed bag; val-selected variant)",
     "tabular_source":"tabular_fair (features=[Xz,OWN], identical to GNN node features)"}
os.makedirs("data/ext/pure",exist_ok=True)
OUT=os.environ.get("OUT",f"data/ext/pure/leaderboard_{SETTING}.json")
json.dump(out,open(OUT,"w"),indent=2,ensure_ascii=False)
print(f"[done -> {OUT}]",file=sys.stderr)
for k,v in sorted(lb.items(),key=lambda kv:-kv[1]["pr"]): print(f"{k:22s} PR={v['pr']:.4f} r@10%={v['recall@10%']:.3f} ROC={v['roc']:.4f}")
