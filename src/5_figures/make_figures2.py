#!/usr/bin/env python3
"""Round-2 figures: contagion granularity gradient + model comparison (incl XGBoost) across labels."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams["font.family"]="DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"]=False
os.makedirs("figs",exist_ok=True)

R={"any":json.load(open("data/legacy/results.json")),
   "material":json.load(open("data/legacy/results_adverse.json")),
   "severe":json.load(open("data/legacy/results_severe.json"))}
PC=json.load(open("data/legacy/partner_contagion.json"))
LABS=["any","material","severe"]; LABKO={"any":"Any restatement","material":"Material","severe":"Severe (fraud/ICFR/AAER)"}

# ---- Figure A: contagion granularity gradient (HEADLINE) ----
chans=["partner","office","auditor","board","ownership"]
chko={"partner":"Partner","office":"Office","auditor":"Audit firm","board":"Board","ownership":"Common ownership"}
fig,ax=plt.subplots(figsize=(11,5.5))
x=np.arange(len(chans)); w=0.26; cols={"any":"#3b6fb6","material":"#8e44ad","severe":"#c0392b"}
for i,lab in enumerate(LABS):
    vals=[PC[lab][c]["lift"] if PC[lab][c]["lift"] else 0 for c in chans]
    bars=ax.bar(x+(i-1)*w,vals,w,label=LABKO[lab],color=cols[lab])
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+.02,f"{v:.2f}",ha="center",fontsize=7.5)
ax.axhline(1.0,ls="--",c="gray",lw=1.2,label="no contagion (lift = 1)")
ax.set_xticks(x); ax.set_xticklabels([chko[c] for c in chans]); ax.set_ylabel("Contagion lift  P(event | exposed) / P(event | unexposed)")
ax.set_title("Contagion lift by network channel",fontsize=12,fontweight="bold")
ax.legend(fontsize=8,ncol=2); ax.set_ylim(0,2.9)
plt.tight_layout(); plt.savefig("figs/fig3_contagion_gradient.png",dpi=150); plt.close()
print("wrote figs/fig3_contagion_gradient.png")

# ---- Figure B: model comparison incl XGBoost, across labels ----
mods=[("logistic_features_only","LR\n(feat)","base"),("xgboost_features_only","XGBoost\n(feat)","base"),
      ("temporal_multiplex_FULL","Temporal\nMultiplex","gnn"),("temporal_auditor_only","Auditor\nonly","gnn"),
      ("temporal_office_only","Office\nonly","gnn"),("temporal_board_only","Board\nonly","gnn"),
      ("temporal_ownership_only","Ownership\nonly","gnn")]
def getroc(res,key,kind): return (res["baselines"][key]["roc"] if kind=="base" else res["models"][key]["test"]["roc"])
def getpr(res,key,kind):  return (res["baselines"][key]["pr"]  if kind=="base" else res["models"][key]["test"]["pr"])
fig,ax=plt.subplots(1,2,figsize=(14,5)); fig.suptitle("Model performance (test 2018-2019) - incl. XGBoost, 3 labels",fontweight="bold")
x=np.arange(len(mods)); w=0.26
for i,lab in enumerate(LABS):
    roc=[getroc(R[lab],k,kind) for k,_,kind in mods]
    ax[0].bar(x+(i-1)*w,roc,w,label=LABKO[lab],color=cols[lab])
ax[0].axhline(0.5,ls="--",c="gray"); ax[0].set_xticks(x); ax[0].set_xticklabels([m[1] for m in mods],fontsize=8)
ax[0].set_ylim(0,0.72); ax[0].set_title("ROC-AUC"); ax[0].legend(fontsize=8)
for i,lab in enumerate(LABS):
    pr=[getpr(R[lab],k,kind) for k,_,kind in mods]
    ax[1].bar(x+(i-1)*w,pr,w,label=LABKO[lab],color=cols[lab])
ax[1].set_xticks(x); ax[1].set_xticklabels([m[1] for m in mods],fontsize=8)
ax[1].set_title("PR-AUC"); ax[1].legend(fontsize=8)
plt.tight_layout(rect=(0,0,1,0.94)); plt.savefig("figs/fig4_models_xgb.png",dpi=150); plt.close()
print("wrote figs/fig4_models_xgb.png")

# ---- relation attention (severe) ----
RA=R["severe"].get("relation_attention_layer1",{})
if RA:
    keys=["self","auditor","office","board","ownership"]
    yrs=sorted(RA.keys()); vals=np.array([[RA[y].get(k,0) for k in keys] for y in yrs]).mean(0)
    fig,ax=plt.subplots(figsize=(7,4.2))
    ax.bar(keys,vals,color=["#7f8c8d","#c0392b","#e67e22","#2980b9","#27ae60"])
    for i,v in enumerate(vals): ax.text(i,v+.005,f"{v:.3f}",ha="center",fontsize=9)
    ax.set_title("Learned relation attention (severe label, test mean)",fontweight="bold"); ax.set_ylabel("attention")
    plt.tight_layout(); plt.savefig("figs/fig5_attention_severe.png",dpi=150); plt.close()
    print("wrote figs/fig5_attention_severe.png")
print("\nSEVERE attention:",RA)
