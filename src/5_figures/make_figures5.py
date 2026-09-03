#!/usr/bin/env python3
"""Round-5 figures (ACE-v2 with graph-only decorrelated member): wins vs XGB+graph + bootstrap sig."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; import numpy as np
for f in ["AppleGothic","Apple SD Gothic Neo","NanumGothic"]:
    try: matplotlib.rcParams["font.family"]=f; break
    except Exception: pass
matplotlib.rcParams["axes.unicode_minus"]=False; os.makedirs("figs",exist_ok=True)
ENS="ENSEMBLE_safe"
S={}
for m in ["recent_severe","panel_severe","recent_any","panel_any"]:
    p=f"data/ext/ace2_{m}.json"
    if os.path.exists(p): S[m]=json.load(open(p))
ko={"recent_severe":"최근/심각\n(2021-22)","panel_severe":"장기패널/심각\n(2018-19)","recent_any":"최근/전체정정\n(2021-22)","panel_any":"장기패널/전체정정"}
keys=[k for k in ["recent_severe","recent_any","panel_severe","panel_any"] if k in S]

# Fig12: XGB+graph vs ACE-v2 ENSEMBLE + bootstrap CI
x=np.arange(len(keys)); w=0.36
fig,ax=plt.subplots(figsize=(11,5.6))
xg=[S[k]["results"]["XGB_feat+graph"]["roc"] for k in keys]; en=[S[k]["results"][ENS]["roc"] for k in keys]
ax.bar(x-w/2,xg,w,label="XGB+graph (챔피언 베이스라인)",color="#7f8c8d")
ax.bar(x+w/2,en,w,label="ACE-GNN v2 앙상블 (제안, +graph-only 멤버)",color="#c0392b")
for i,k in enumerate(keys):
    g=S[k]["bootstrap_ROC_gain_vs_XGBgraph"]; ci=g["ci95"]; p=g["p_gt_0"]
    star="***" if p>=0.99 else ("**" if p>=0.975 else ("*" if p>=0.95 else "ns"))
    ax.text(i-w/2,xg[i]+.003,f"{xg[i]:.3f}",ha="center",fontsize=8); ax.text(i+w/2,en[i]+.003,f"{en[i]:.3f}",ha="center",fontsize=8,fontweight="bold")
    ax.annotate(f"+{g['mean']:.4f} {star}\nCI[{ci[0]:.3f},{ci[1]:.3f}]",xy=(i,max(xg[i],en[i])+.013),ha="center",fontsize=8,color="#c0392b")
ax.set_xticks(x); ax.set_xticklabels([ko[k] for k in keys]); ax.set_ylabel("ROC-AUC"); ax.set_ylim(0.55,0.88); ax.legend(loc="upper right")
ax.set_title("ACE-GNN v2 앙상블이 XGB+graph를 능가 (graph-only 탈상관 멤버 추가)\n부트스트랩 3000회 ROC 이득(±95% CI); *p≥.95 **p≥.975 ***p≥.99",fontsize=11,fontweight="bold")
plt.tight_layout(); plt.savefig("figs/fig12_ace_win.png",dpi=150); plt.close(); print("fig12 settings:",keys)

# Fig13: components + decorrelation (recent_severe)
if "recent_severe" in S:
    d=S["recent_severe"]["results"]; pc=S["recent_severe"]["pred_corr"]
    comps=["XGB_feat+graph","ACE_full_bag","ACE_raw_bag","ACE_graphonly_bag",ENS]
    cko=[f"XGB+graph",f"ACE-full\n(corr {pc['xgb_vs_full']})",f"ACE-raw\n(corr {pc['xgb_vs_raw']})",f"ACE-graphonly\n(corr {pc['xgb_vs_graphonly']})","ENSEMBLE\n(제안)"]
    v=[d[c]["roc"] for c in comps]; cols=["#7f8c8d","#e67e22","#2980b9","#27ae60","#c0392b"]
    fig,ax=plt.subplots(figsize=(9.5,5)); b=ax.bar(np.arange(len(comps)),v,color=cols)
    for bb,vv in zip(b,v): ax.text(bb.get_x()+bb.get_width()/2,vv+.003,f"{vv:.3f}",ha="center",fontsize=9,fontweight="bold")
    ax.axhline(d["XGB_feat+graph"]["roc"],ls="--",c="#7f8c8d",lw=1)
    ax.set_xticks(np.arange(len(comps))); ax.set_xticklabels(cko,fontsize=8.5); ax.set_ylim(0.55,0.86); ax.set_ylabel("ROC-AUC (recent/severe)")
    ax.set_title("왜 이기는가 — 탈상관(graph-only corr 0.74) 멤버가 앙상블을 끌어올림\n개별 모델은 XGB 이하라도, 상보적 결합으로 앙상블이 모두 상회",fontsize=10.5,fontweight="bold")
    plt.tight_layout(); plt.savefig("figs/fig13_ace_components.png",dpi=150); plt.close(); print("fig13")
