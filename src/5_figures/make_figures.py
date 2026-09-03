#!/usr/bin/env python3
"""Figures for the restatement-contagion GNN experiment."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
for f in ["AppleGothic","Apple SD Gothic Neo","NanumGothic","Malgun Gothic"]:
    try:
        matplotlib.rcParams["font.family"]=f; break
    except Exception: pass
matplotlib.rcParams["axes.unicode_minus"]=False

R=json.load(open("data/results.json"))
ADV=json.load(open("data/results_adverse.json")) if os.path.exists("data/results_adverse.json") else None
os.makedirs("figs",exist_ok=True)
LRr=R["baselines"]["logistic_features_only"]
models=list(R["models"].keys())
nice={"temporal_multiplex_FULL":"Temporal\nMultiplex","static_multiplex_noGRU":"Static\nMultiplex",
      "temporal_auditor_only":"Auditor\nonly","temporal_board_only":"Board\nonly",
      "temporal_ownership_only":"Ownership\nonly","temporal_no_ownership":"Auditor+\nBoard"}

# ---------- Figure 1: 2x2 main ----------
fig,ax=plt.subplots(2,2,figsize=(13,9)); fig.suptitle("Restatement-Contagion Prediction via Self-Interpretable Temporal Multiplex GNN\n(US public firms, WRDS: Audit Analytics + BoardEx + 13F + Compustat; test = predict 2018-2019)",fontsize=11,fontweight="bold")

# (a) model ROC/PR bars
labels=["Logistic\n(features)"]+[nice[m] for m in models]
roc=[LRr["roc"]]+[R["models"][m]["test"]["roc"] for m in models]
pr =[LRr["pr"]] +[R["models"][m]["test"]["pr"]  for m in models]
x=np.arange(len(labels)); w=0.38
b1=ax[0,0].bar(x-w/2,roc,w,label="ROC-AUC",color="#3b6fb6")
b2=ax[0,0].bar(x+w/2,pr,w,label="PR-AUC",color="#e08a3c")
ax[0,0].axhline(0.5,ls="--",c="gray",lw=1); ax[0,0].axhline(LRr["base_rate"],ls=":",c="red",lw=1,label=f"base rate={LRr['base_rate']:.3f}")
ax[0,0].set_xticks(x); ax[0,0].set_xticklabels(labels,fontsize=8); ax[0,0].set_ylim(0,0.72)
ax[0,0].set_title("(a) Test performance by model",fontsize=10); ax[0,0].legend(fontsize=8,loc="upper right")
for b in list(b1)+list(b2): ax[0,0].text(b.get_x()+b.get_width()/2,b.get_height()+.005,f"{b.get_height():.3f}",ha="center",fontsize=6.5)

# (b) contagion lift
C=R["contagion_descriptive"]; rels=list(C.keys())
exp=[C[r]["P(restate_t+1|exposed)"] for r in rels]; nex=[C[r]["P(restate_t+1|not_exposed)"] for r in rels]
x=np.arange(len(rels))
ax[0,1].bar(x-w/2,exp,w,label="이웃이 당해 정정함(exposed)",color="#c0392b")
ax[0,1].bar(x+w/2,nex,w,label="비노출",color="#95a5a6")
for i,r in enumerate(rels):
    lift=C[r]["lift"]; ax[0,1].text(i,max(exp[i],nex[i])+.003,f"lift={lift}",ha="center",fontsize=8,fontweight="bold")
ax[0,1].set_xticks(x); ax[0,1].set_xticklabels(rels); ax[0,1].set_ylabel("P(정정 in t+1)")
ax[0,1].set_title("(b) 전염(contagion): 정정한 이웃 노출 효과",fontsize=10); ax[0,1].legend(fontsize=8)

# (c) relation attention
RA=R.get("relation_attention_layer1",{})
if RA:
    keys=["self","auditor","board","ownership"]
    yrs=sorted(RA.keys())
    vals=np.array([[RA[y][k] for k in keys] for y in yrs]).mean(0)
    cols=["#7f8c8d","#c0392b","#2980b9","#27ae60"]
    ax[1,0].bar(keys,vals,color=cols)
    for i,v in enumerate(vals): ax[1,0].text(i,v+.005,f"{v:.3f}",ha="center",fontsize=9)
    ax[1,0].set_title("(c) 학습된 관계 어텐션 (자기설명형, test 평균)",fontsize=10)
    ax[1,0].set_ylabel("attention weight")

# (d) per-year ROC + positive rate
full=R["models"]["temporal_multiplex_FULL"]
yr_roc={**{int(y):full["val"]["per_year"][y]["roc"] for y in full["val"]["per_year"]},
        **{int(y):full["test"]["per_year"][y]["roc"] for y in full["test"]["per_year"]}}
pr_rate=R["label_pos_rate"]
yrs=sorted(int(y) for y in pr_rate)
ax2=ax[1,1]; ax3=ax2.twinx()
ax2.plot(list(yr_roc.keys()),list(yr_roc.values()),"o-",c="#3b6fb6",label="ROC (full model)")
ax3.bar(yrs,[pr_rate[str(y)] for y in yrs],alpha=0.25,color="red",label="정정 양성률")
ax2.set_ylim(0.5,0.7); ax2.axhline(0.5,ls="--",c="gray",lw=1)
ax2.set_xlabel("예측 대상연도 (t+1)"); ax2.set_ylabel("ROC-AUC",color="#3b6fb6"); ax3.set_ylabel("양성률",color="red")
ax2.set_title("(d) 연도별 ROC & 정정 양성률",fontsize=10)
ax2.axvspan(2010.5,2016.5,alpha=0.05,color="green"); ax2.text(2013,0.515,"train",ha="center",fontsize=8,color="green")
ax2.axvspan(2016.5,2017.5,alpha=0.05,color="orange"); ax2.axvspan(2017.5,2019.5,alpha=0.08,color="blue"); ax2.text(2018.5,0.515,"test",ha="center",fontsize=8,color="blue")
plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig("figs/fig1_main.png",dpi=150); plt.close()
print("wrote figs/fig1_main.png")

# ---------- Figure 2: any vs material restatement ----------
if ADV:
    keymods=["temporal_multiplex_FULL","temporal_auditor_only"]
    fig,ax=plt.subplots(1,2,figsize=(12,4.5)); fig.suptitle("Robustness: any restatement vs material (adverse) restatement",fontweight="bold")
    # ROC compare
    grp=["LR (feat)","Temporal\nMultiplex","Auditor\nonly"]
    any_roc=[R["baselines"]["logistic_features_only"]["roc"]]+[R["models"][m]["test"]["roc"] for m in keymods]
    adv_roc=[ADV["baselines"]["logistic_features_only"]["roc"]]+[ADV["models"][m]["test"]["roc"] for m in keymods]
    x=np.arange(len(grp))
    ax[0].bar(x-0.2,any_roc,0.4,label="any restatement",color="#3b6fb6")
    ax[0].bar(x+0.2,adv_roc,0.4,label="material (adverse)",color="#8e44ad")
    ax[0].axhline(0.5,ls="--",c="gray"); ax[0].set_xticks(x); ax[0].set_xticklabels(grp,fontsize=9)
    ax[0].set_title("Test ROC-AUC"); ax[0].set_ylim(0,0.72); ax[0].legend(fontsize=8)
    for i,(a,b) in enumerate(zip(any_roc,adv_roc)):
        ax[0].text(i-0.2,a+.005,f"{a:.3f}",ha="center",fontsize=7); ax[0].text(i+0.2,b+.005,f"{b:.3f}",ha="center",fontsize=7)
    # contagion lift compare
    rels=list(R["contagion_descriptive"].keys())
    any_l=[R["contagion_descriptive"][r]["lift"] for r in rels]
    adv_l=[ADV["contagion_descriptive"][r]["lift"] for r in rels]
    x=np.arange(len(rels))
    ax[1].bar(x-0.2,any_l,0.4,label="any",color="#3b6fb6"); ax[1].bar(x+0.2,adv_l,0.4,label="material",color="#8e44ad")
    ax[1].axhline(1.0,ls="--",c="gray"); ax[1].set_xticks(x); ax[1].set_xticklabels(rels)
    ax[1].set_title("Contagion lift by relation"); ax[1].legend(fontsize=8)
    for i,(a,b) in enumerate(zip(any_l,adv_l)):
        ax[1].text(i-0.2,a+.02,f"{a}",ha="center",fontsize=8); ax[1].text(i+0.2,b+.02,f"{b}",ha="center",fontsize=8)
    plt.tight_layout(rect=[0,0,1,0.93]); plt.savefig("figs/fig2_robustness.png",dpi=150); plt.close()
    print("wrote figs/fig2_robustness.png")
