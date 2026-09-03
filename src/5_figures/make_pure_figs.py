#!/usr/bin/env python3
"""Pure-model figures for the Information Sciences submission (single WRDS dataset, pure models only).
Reads canonical pure results from data/ext/pure/ and renders English figures into figs/ and
paper/Information_Sciences/figures/. Synthetic (+graph) / ensemble / external models are NOT shown.

Generates (skips any whose source JSON is missing):
  fig_pure_leaderboard_recent_severe.png   <- data/ext/pure/leaderboard_recent_severe.json
  fig_pure_leaderboard_panel_severe.png    <- data/ext/pure/leaderboard_panel_severe.json
  fig_pure_ablation.png                     <- data/ext/pure/ace_pure_recent_severe.json
  fig9_hierarchy_weights.png                <- data/ext/pure/ace_pure_recent_severe.json (aud_level_weights)
Run from repo root:  python3 src/5_figures/make_pure_figs.py
"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

OUTDIRS = ["figs", "paper/Information_Sciences/figures"]
def save(fig, name):
    for d in OUTDIRS:
        os.makedirs(d, exist_ok=True)
        fig.savefig(f"{d}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("wrote", name)

# colours by family
C_OURS="#c0392b"; C_GNN="#2980b9"; C_SELF="#8e44ad"; C_TAB="#95a5a6"
STD_GNN={"GCN","SAGE","GAT","APPNP","RGCN"}; SELF_GNN={"GSAT"}
def fam_color(name):
    if "ours" in name.lower() or name.startswith("ACE"): return C_OURS
    if name in STD_GNN: return C_GNN
    if name in SELF_GNN: return C_SELF
    return C_TAB

def leaderboard_fig(setting, title):
    fn=f"data/ext/pure/leaderboard_{setting}.json"
    if not os.path.exists(fn): print("skip leaderboard", setting); return
    d=json.load(open(fn)); lb=d["leaderboard_pure"]
    rows=sorted(lb.items(), key=lambda kv:(kv[1].get("pr") or 0))   # ascending -> best on top
    names=[k.replace(" (ours)","") for k,_ in rows]
    pr=[v.get("pr") for _,v in rows]; cols=[fam_color(k) for k,_ in rows]
    fig,ax=plt.subplots(figsize=(8,5.6))
    y=np.arange(len(names)); ax.barh(y,pr,color=cols)
    for i,v in enumerate(pr): ax.text(v+0.004,i,f"{v:.3f}",va="center",fontsize=8,
                                      fontweight=("bold" if cols[i]==C_OURS else "normal"))
    ax.set_yticks(y); ax.set_yticklabels(names,fontsize=9)
    ax.set_xlabel("AUPRC  (primary metric; positives are rare)")
    ax.set_xlim(0,max(pr)*1.16)
    n=d.get("n_test"); pos=d.get("pos_test"); ty=d.get("test_years")
    ax.set_title(f"{title}\nPure-model leaderboard  (test {ty}, n={n}, positives={pos})",
                 fontsize=10.5, fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_OURS,label="ACE-GNN (ours)"),Patch(color=C_GNN,label="Standard GNN"),
                       Patch(color=C_SELF,label="Self-interpretable GNN"),Patch(color=C_TAB,label="Tabular ML")],
              loc="lower right",fontsize=8,framealpha=0.9)
    fig.tight_layout(); save(fig, f"fig_pure_leaderboard_{setting}")

def ablation_fig():
    fn="data/ext/pure/ace_pure_recent_severe.json"
    if not os.path.exists(fn): print("skip ablation (ace_pure not ready)"); return
    d=json.load(open(fn))["results"]
    if "ACE-GNN(pure)" not in d: print("skip ablation (no headline)"); return
    base=d["ACE-GNN(pure)"]["pr"]
    order=[("ACE-GNN(pure)","Full ACE-GNN (pure)"),
           ("-collective(K=0)","- collective inference"),
           ("-attention(mean)","- relation attention (mean agg.)"),
           ("-monotone(free)","- monotone hierarchy (free wts)"),
           ("+financials","+ firm financial ratios")]
    rows=[(lab,d[k]["pr"]) for k,lab in order if k in d]
    if len(rows)<2: print("skip ablation (insufficient)"); return
    labs=[r[0] for r in rows]; prs=[r[1] for r in rows]
    cols=[C_OURS]+[ "#34495e" ]*(len(rows)-1)
    fig,ax=plt.subplots(figsize=(8,4.6))
    y=np.arange(len(labs))[::-1]; ax.barh(y,prs,color=cols)
    for i,v in zip(y,prs):
        ax.text(v+0.003,i,f"{v:.3f}",va="center",fontsize=9,fontweight="bold")
    ax.axvline(base,ls="--",lw=1,color=C_OURS,alpha=.6)
    ax.set_yticks(y); ax.set_yticklabels(labs,fontsize=9)
    ax.set_xlabel("AUPRC (recent/severe)"); ax.set_xlim(0,max(prs)*1.15)
    ax.set_title("Component ablation of ACE-GNN (graph-only)\n"
                 "architectural components are accuracy-neutral; adding firm financials hurts screening",
                 fontsize=10,fontweight="bold")
    fig.tight_layout(); save(fig,"fig_pure_ablation")

def hierarchy_fig():
    fn="data/ext/pure/ace_pure_recent_severe.json"
    if not os.path.exists(fn): print("skip fig9 (ace_pure not ready)"); return
    w=json.load(open(fn))["results"].get("ACE-GNN(pure)",{}).get("aud_level_weights")
    if not w: print("skip fig9 (no weights)"); return
    lv=["partner","office","auditor"]; labels=["Partner (L1)","Office (L2)","Audit firm (L3)"]
    vals=[w.get(k) for k in lv]
    fig,ax=plt.subplots(figsize=(7,4.4)); x=np.arange(3)
    ax.bar(x,vals,0.55,color=C_OURS)
    for i,v in enumerate(vals): ax.text(i,v+0.02,f"{v:.2f}",ha="center",fontsize=10,fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0,1.18)
    ax.set_ylabel("Auditor-level weight  w (monotone by design)")
    ax.set_title("Monotone auditor-hierarchy weights (recent/severe)\n"
                 "ordering partner >= office >= firm imposed by design; "
                 "independently corroborated by contagion lift / drop-one",
                 fontsize=10,fontweight="bold")
    fig.tight_layout(); save(fig,"fig9_hierarchy_weights")

if __name__=="__main__":
    leaderboard_fig("recent_severe","ACE-GNN tops every pure model on rare-event accounting-risk prediction")
    leaderboard_fig("panel_severe", "Result holds on the long-panel split (2017-2019 test)")
    ablation_fig()
    hierarchy_fig()
    print("done.")
