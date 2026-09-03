#!/usr/bin/env python3
"""Round-4 figures: HAT learned hierarchy weights, performance, data-augmentation effect."""
import json, glob, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; import numpy as np
for f in ["AppleGothic","Apple SD Gothic Neo","NanumGothic"]:
    try: matplotlib.rcParams["font.family"]=f; break
    except Exception: pass
matplotlib.rcParams["axes.unicode_minus"]=False; os.makedirs("figs",exist_ok=True)
H={}
for f in glob.glob("data/ext/hat_*.json"):
    d=json.load(open(f)); H[(d["mode"],d["label"])]=d

# Fig9: learned hierarchy weights (recent/severe) full vs flat
rs=H[("recent","severe")]["models"]
lv=["partner","office","auditor"]; lko=["파트너(L1)","사무소(L2)","감사인(L3)"]
fig,ax=plt.subplots(figsize=(8,4.8)); x=np.arange(3); w=0.38
full=[rs["HAT-GNN_full"]["aud_level_weights"][k] for k in lv]
flat=[rs["HAT_flat(no-hierarchy)"]["aud_level_weights"][k] for k in lv]
ax.bar(x-w/2,full,w,label="HAT-GNN (위계 prior)",color="#c0392b")
ax.bar(x+w/2,flat,w,label="Flat (위계 없음)",color="#95a5a6")
for i,(a,b) in enumerate(zip(full,flat)):
    ax.text(i-w/2,a+.02,f"{a:.2f}",ha="center",fontsize=9,fontweight="bold"); ax.text(i+w/2,b+.02,f"{b:.2f}",ha="center",fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(lko); ax.set_ylabel("학습된 레벨 가중치"); ax.set_ylim(0,1.15); ax.legend()
ax.set_title("HAT-GNN이 학습한 감사인 위계 (recent/severe)\n위계 prior → 파트너>사무소>감사인 단조 회복(전염 구배와 일치) ; flat은 균등분할",fontsize=10.5,fontweight="bold")
plt.tight_layout(); plt.savefig("figs/fig9_hierarchy_weights.png",dpi=150); plt.close(); print("fig9")

# Fig10: performance recent vs panel (severe + any)
fig,ax=plt.subplots(1,2,figsize=(13,5))
for j,lab in enumerate(["severe","label"]):
    mods=["XGB_feat","XGB_feat+graph","HAT-GNN_full"]
    if ("recent",lab) in H and "HAT_noPartner(office+aud)" in H[("recent",lab)]["models"]: mods=["XGB_feat","XGB_feat+graph","HAT_noPartner(office+aud)","HAT-GNN_full"]
    mko={"XGB_feat":"XGB\n(feat)","XGB_feat+graph":"XGB\n+graph","HAT-GNN_full":"HAT-GNN\n(full)","HAT_noPartner(office+aud)":"HAT\n(no partner)"}
    x=np.arange(len(mods)); w=0.38
    for i,(mode,col) in enumerate([("panel","#2c7fb8"),("recent","#d95f0e")]):
        d=H.get((mode,lab))
        if not d: continue
        v=[ (d["baselines"][m]["roc"] if m in d["baselines"] else d["models"].get(m,{}).get("roc",np.nan)) for m in mods]
        b=ax[j].bar(x+(i-0.5)*w,v,w,label=f"{mode} (test {d['years'][-2]}-{d['years'][-1]})",color=col)
        for bb,vv in zip(b,v):
            if not np.isnan(vv): ax[j].text(bb.get_x()+bb.get_width()/2,vv+.004,f"{vv:.3f}",ha="center",fontsize=7)
    ax[j].axhline(0.5,ls="--",c="gray"); ax[j].set_xticks(x); ax[j].set_xticklabels([mko[m] for m in mods],fontsize=8)
    ax[j].set_ylim(0.5,0.86); ax[j].set_title(("심각(severe)" if lab=="severe" else "전체정정(any)")+" 라벨"); ax[j].legend(fontsize=8)
    ax[j].set_ylabel("ROC-AUC")
fig.suptitle("HAT-GNN vs XGBoost — 장기패널(2005-19) & 최근(2017-22)\nHAT-GNN ≈ XGB+graph(동급), 파트너 추가 정확도이득 미미; 그래프 증강이 핵심",fontweight="bold",fontsize=11)
plt.tight_layout(rect=(0,0,1,0.92)); plt.savefig("figs/fig10_hat_performance.png",dpi=150); plt.close(); print("fig10")

# Fig11: data augmentation effect (severe panel: 10yr/6yr-train vs 17yr/11yr-train)
try:
    r3=json.load(open("data/card_results_severe.json"))  # round-3 CARD, 2010-19 panel (train 6yr)
    old_xgb=r3["baselines"]["XGB_feat+graph"]["roc"]; old_gnn=r3["models"]["CARD-Hybrid_FULL"]["roc"]
    new=H[("panel","severe")]; new_xgb=new["baselines"]["XGB_feat+graph"]["roc"]; new_gnn=new["models"]["HAT-GNN_full"]["roc"]
    fig,ax=plt.subplots(figsize=(7.5,4.8)); x=np.arange(2); w=0.38
    ax.bar(x-w/2,[old_xgb,old_gnn],w,label="패널 2010-19 (train 6년)",color="#bdc3c7")
    ax.bar(x+w/2,[new_xgb,new_gnn],w,label="패널 2005-19 (train 11년)",color="#16a085")
    for i,(o,n) in enumerate(zip([old_xgb,old_gnn],[new_xgb,new_gnn])):
        ax.text(i-w/2,o+.003,f"{o:.3f}",ha="center",fontsize=9); ax.text(i+w/2,n+.003,f"{n:.3f}",ha="center",fontsize=9,fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["XGB+graph","GNN(full)"]); ax.set_ylim(0.74,0.83); ax.set_ylabel("ROC-AUC (심각, test 2018-19)")
    ax.set_title("데이터 보강(시계열 확장)의 효과 — 심각 라벨\n학습기간 6년→11년 확장으로 +0.018 ROC",fontsize=10.5,fontweight="bold"); ax.legend()
    plt.tight_layout(); plt.savefig("figs/fig11_data_augmentation.png",dpi=150); plt.close(); print("fig11")
except Exception as e: print("fig11 skip",e)
