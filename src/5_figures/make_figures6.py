#!/usr/bin/env python3
"""Round-6/7 figures: 16-baseline leaderboard, component ablation, relation-drop, calibration, per-year."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; import numpy as np
for f in ["AppleGothic","Apple SD Gothic Neo","NanumGothic"]:
    try: matplotlib.rcParams["font.family"]=f; break
    except Exception: pass
matplotlib.rcParams["axes.unicode_minus"]=False; os.makedirs("figs",exist_ok=True)

# Fig14: leaderboard recent/severe (horizontal bar, ACE highlighted)
if os.path.exists("data/ext/bench_recent_severe.json"):
    d=json.load(open("data/ext/bench_recent_severe.json")); lb=d["leaderboard"]; rk=d["ranking"]
    names=rk[::-1]; rocs=[lb[k]["roc"] for k in names]; cols=["#c0392b" if lb[k].get("is_ours") else "#7f8c8d" for k in names]
    fig,ax=plt.subplots(figsize=(8,7)); ax.barh(range(len(names)),rocs,color=cols)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names,fontsize=8); ax.set_xlim(0.5,0.86); ax.set_xlabel("ROC-AUC")
    for i,(k,r) in enumerate(zip(names,rocs)): ax.text(r+.002,i,f"{r:.3f}",va="center",fontsize=7)
    ax.set_title("16-baseline 리더보드 — recent/severe (test 2021-22)\nACE-GNN(빨강) 1위, 전 baseline 부트스트랩 p=1.0로 유의 능가",fontsize=10.5,fontweight="bold")
    plt.tight_layout(); plt.savefig("figs/fig14_leaderboard_recent_severe.png",dpi=150); plt.close(); print("fig14")

# Fig15: component ablation + relation drop (recent/severe)
if os.path.exists("data/ext/ablation_recent_severe.json"):
    a=json.load(open("data/ext/ablation_recent_severe.json")); ca=a["component_ablation"]; rd=a["relation_drop"]
    full=ca["FULL"]["roc_mean"]
    fig,ax=plt.subplots(1,2,figsize=(13,5))
    # component
    order=["FULL","-GBDTleaf","-collective(K=0)","graph-only(-features)","-attention(mean)","-monotone(free wts)"]
    order=[k for k in order if k in ca]; vals=[ca[k]["roc_mean"] for k in order]; errs=[ca[k]["roc_std"] for k in order]
    cols=["#c0392b"]+["#3b6fb6"]*(len(order)-1)
    ax[0].bar(range(len(order)),vals,yerr=errs,color=cols,capsize=3)
    ax[0].set_xticks(range(len(order))); ax[0].set_xticklabels([k.replace("(","\n(") for k in order],fontsize=7.5,rotation=0)
    ax[0].axhline(full,ls="--",c="#c0392b",lw=1); ax[0].set_ylim(0.77,0.83); ax[0].set_ylabel("ROC-AUC (단일 ACE GNN, 3-seed)")
    for i,(v,e) in enumerate(zip(vals,errs)): ax[0].text(i,v+e+.001,f"{v:.3f}",ha="center",fontsize=7)
    ax[0].set_title("(a) 컴포넌트 ablation\nGBDT-leaf·collective inference가 핵심 기여",fontsize=10)
    # relation drop
    rk=list(rd.keys()); rv=[rd[k]["roc_mean"] for k in rk]; re_=[rd[k]["roc_std"] for k in rk]
    ax[1].bar(range(len(rk)),[full-x for x in rv],yerr=re_,color="#e67e22",capsize=3)
    ax[1].axhline(0,c="gray",lw=1); ax[1].set_xticks(range(len(rk))); ax[1].set_xticklabels([k.replace("-","drop\n") for k in rk],fontsize=8)
    ax[1].set_ylabel("ROC 감소 (FULL 대비)")
    for i,(v) in enumerate([full-x for x in rv]): ax[1].text(i,v+.0005,f"{v:+.4f}",ha="center",fontsize=7.5)
    ax[1].set_title("(b) relation drop-one\n각 관계 제거 시 ROC 감소(=채널 기여)",fontsize=10)
    fig.suptitle("ACE-GNN ablation (recent/severe)",fontweight="bold")
    plt.tight_layout(rect=(0,0,1,0.95)); plt.savefig("figs/fig15_ablation.png",dpi=150); plt.close(); print("fig15")

# Fig16: calibration reliability + per-year (from posthoc)
if os.path.exists("data/ext/posthoc.json"):
    ph=json.load(open("data/ext/posthoc.json"))
    fig,ax=plt.subplots(1,2,figsize=(12,4.8))
    for key,c in [("recent/severe","#c0392b"),("panel/severe","#2c7fb8")]:
        if key in ph and ph[key].get("calibration_ACE"):
            rel=ph[key]["calibration_ACE"]; ax[0].plot([r["pred"] for r in rel],[r["obs"] for r in rel],"o-",label=key,color=c)
    ax[0].plot([0,0.6],[0,0.6],"k--",lw=1,label="perfect"); ax[0].set_xlabel("예측확률"); ax[0].set_ylabel("실제 빈도"); ax[0].legend(fontsize=8)
    ax[0].set_title("(a) Calibration (reliability) — ACE",fontsize=10)
    # per-year
    width=0.35; offs=0
    for key,c in [("recent/severe","#c0392b"),("panel/severe","#2c7fb8")]:
        if key in ph and isinstance(ph[key].get("per_year"),dict):
            py=ph[key]["per_year"]; yrs=sorted(int(y) for y in py)
            ax[1].plot(yrs,[py[str(y)]["ACE_roc"] for y in yrs],"o-",label=f"{key} ACE",color=c)
            ax[1].plot(yrs,[py[str(y)]["XGBg_roc"] for y in yrs],"s--",label=f"{key} XGB+g",color=c,alpha=0.5)
    ax[1].set_xlabel("예측 대상연도"); ax[1].set_ylabel("ROC-AUC"); ax[1].legend(fontsize=7); ax[1].set_title("(b) 연도별 test ROC",fontsize=10)
    plt.tight_layout(); plt.savefig("figs/fig16_calibration_peryear.png",dpi=150); plt.close(); print("fig16")
print("figures done")
