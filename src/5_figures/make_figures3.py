#!/usr/bin/env python3
"""Round-3 figures: honest marginal-value decomposition, CARD vs baselines, partner ablation."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; import numpy as np
for f in ["AppleGothic","Apple SD Gothic Neo","NanumGothic"]:
    try: matplotlib.rcParams["font.family"]=f; break
    except Exception: pass
matplotlib.rcParams["axes.unicode_minus"]=False; os.makedirs("figs",exist_ok=True)
DEC=json.load(open("data/panel_decomp.json"))
CARD={"any":json.load(open("data/card_results.json")),"material":json.load(open("data/card_results_adverse.json")),
      "severe":json.load(open("data/card_results_severe.json"))}
PR=json.load(open("data/partner_recent.json"))
cols={"any":"#3b6fb6","material":"#8e44ad","severe":"#c0392b"}; LK={"any":"전체정정","material":"중대정정","severe":"심각"}

# Fig6: marginal-value decomposition (panel 10yr, XGBoost)
steps=["feat","feat+own","feat+own+graphExp","feat+own+graphExp+nbrFeat"]
sko=["재무피처","+자기지속성","+그래프\n전염노출","+이웃피처"]
fig,ax=plt.subplots(figsize=(10,5.2))
x=np.arange(len(steps)); w=0.34
for i,lab in enumerate(["any","severe"]):
    v=[DEC[lab][s]["roc"] for s in steps]
    b=ax.bar(x+(i-0.5)*w,v,w,label=LK[lab],color=cols[lab])
    for bb,vv in zip(b,v): ax.text(bb.get_x()+bb.get_width()/2,vv+.004,f"{vv:.3f}",ha="center",fontsize=8)
# delta annotations for graph step
for i,lab in enumerate(["any","severe"]):
    d=DEC[lab]["feat+own+graphExp"]["roc"]-DEC[lab]["feat+own"]["roc"]
    ax.annotate(f"그래프 한계기여\n+{d:.3f}",xy=(2+(i-0.5)*w,DEC[lab]["feat+own+graphExp"]["roc"]),
                xytext=(2+(i-0.5)*w,0.50+i*0.03),fontsize=8,ha="center",color=cols[lab],
                arrowprops=dict(arrowstyle="->",color=cols[lab]))
ax.axhline(0.5,ls="--",c="gray"); ax.set_xticks(x); ax.set_xticklabels(sko); ax.set_ylim(0.48,0.82)
ax.set_ylabel("ROC-AUC (test 2018–2019)"); ax.legend()
ax.set_title("그래프 전염노출의 '순수 한계기여' 분해 (패널 10년, XGBoost)\n자기지속성이 최대 기여 → 그래프는 그 위에 modest하게 추가 (noisy 라벨서 더 큼)",fontsize=10.5,fontweight="bold")
plt.tight_layout(); plt.savefig("figs/fig6_decomposition.png",dpi=150); plt.close(); print("fig6")

# Fig7: CARD vs baselines across labels
mods=[("LR_feat","LR","b"),("XGB_feat","XGB\n(feat)","b"),("XGB_feat+graph","XGB\n+graph","b"),
      ("CARD-Hybrid_FULL","CARD-\nHybrid","m"),("CARD_pureNeural","CARD\n(pure GNN)","m")]
def val(res,k):
    return res["baselines"][k]["roc"] if k in res["baselines"] else res["models"][k]["roc"]
fig,ax=plt.subplots(figsize=(11,5))
x=np.arange(len(mods)); w=0.26
for i,lab in enumerate(["any","material","severe"]):
    v=[val(CARD[lab],k) for k,_,_ in mods]
    b=ax.bar(x+(i-1)*w,v,w,label=LK[lab],color=cols[lab])
    for bb,vv in zip(b,v): ax.text(bb.get_x()+bb.get_width()/2,vv+.004,f"{vv:.3f}",ha="center",fontsize=7)
ax.axhline(0.5,ls="--",c="gray"); ax.set_xticks(x); ax.set_xticklabels([m[1] for m in mods]); ax.set_ylim(0.5,0.82)
ax.set_ylabel("ROC-AUC"); ax.legend(); ax.axvspan(2.5,4.5,alpha=0.05,color="purple")
ax.set_title("CARD-Hybrid vs 베이스라인 — 라벨 3종 (test 2018–2019)\nCARD-Hybrid ≈ XGB+graph(챔피언) 동급; 순수 GNN은 약간 아래; 그래프 증강이 핵심",fontsize=10.5,fontweight="bold")
plt.tight_layout(); plt.savefig("figs/fig7_card_vs_baselines.png",dpi=150); plt.close(); print("fig7")

# Fig8: partner recent ablation
order=["base(feat+own)","+panel graph","+panel+PARTNER"]; oko=["base\n(feat+own)","+패널\n그래프","+개별\n파트너"]
fig,ax=plt.subplots(figsize=(8.5,5))
x=np.arange(len(order)); w=0.34
for i,lab in enumerate(["any","severe"]):
    v=[PR[lab][s]["roc"] for s in order]
    b=ax.bar(x+(i-0.5)*w,v,w,label=LK[lab],color=cols[lab])
    for bb,vv in zip(b,v): ax.text(bb.get_x()+bb.get_width()/2,vv+.003,f"{vv:.3f}",ha="center",fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(oko); ax.set_ylabel("ROC-AUC (2017–2019)"); ax.legend()
ax.set_title("개별 engagement-partner 채널의 증분 기여 (최근기간 2017–2019)\n파트너 단변량 lift 2.54이나, firm피처+지속성 통제 시 다변량 증분은 작음(희소·교란)",fontsize=10,fontweight="bold")
plt.tight_layout(); plt.savefig("figs/fig8_partner_recent.png",dpi=150); plt.close(); print("fig8")
