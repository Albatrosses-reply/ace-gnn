#!/usr/bin/env python3
"""Interpretability case-study figure (the 'why GNN, not RF' evidence), from case_studies_v2.json.
(a) hero ego-network: a 'silent' firm (no own problem) flagged via restating audit co-clients, edge width = learned attention;
(b) edge-removal counterfactual for the hero: removing the auditor channel collapses the risk, removing board/ownership does not;
(c) prevalence: distribution of the auditor-channel counterfactual drop across all 383 graph-dependent failures.
Run: python3 src/5_figures/make_case_fig.py"""
import json, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"]="DejaVu Sans"; matplotlib.rcParams["axes.unicode_minus"]=False
RC={"partner":"#8e44ad","office":"#e67e22","auditor":"#c0392b"}
RKO={"partner":"Engagement partner","office":"Audit office","auditor":"Audit firm"}
C=json.load(open("data/ext/case_studies_v2.json")); prev=C["_prevalence"]; case=C["case1"]
fig=plt.figure(figsize=(15,4.7)); gs=fig.add_gridspec(1,3,width_ratios=[1.05,0.8,1.15])

# ---- (a) ego-network of the hero 'silent' firm ----
axa=fig.add_subplot(gs[0,0]); axa.axis("off"); axa.set_xlim(-1.3,1.3); axa.set_ylim(-1.3,1.3)
peers=case["top_attended_auditor_peers"]; n=len(peers)
for k,pe in enumerate(peers):
    ang=2*np.pi*k/n+np.pi/2; x,y=np.cos(ang),np.sin(ang); rel=pe["relation"]; isr=pe["restating"]==1
    aw=pe["alpha"]
    axa.plot([0,x],[0,y],"-",color=RC.get(rel,"#999"),lw=1.0+4.5*aw,alpha=0.85 if isr else 0.5,zorder=1)
    axa.scatter([x],[y],s=300 if isr else 150,c=("#c0392b" if isr else "white"),
                edgecolors=RC.get(rel,"#999"),linewidths=2.2 if isr else 1.4,zorder=3)
    axa.text(x*1.32,y*1.32,f"SIC {pe['sic']}\n{'restating' if isr else 'clean'}",ha="center",va="center",fontsize=7.2,
             color=("#c0392b" if isr else "#555"))
axa.scatter([0],[0],s=620,c="#f1c40f",edgecolors="black",linewidths=1.8,zorder=4)
axa.text(0,0,"focal\n(no own\nproblem)",ha="center",va="center",fontsize=7,fontweight="bold",zorder=5)
axa.set_title(f"(a) A 'silent' firm flagged by its network\naerospace (SIC {case['focal']['sic']}), FY{case['year']}; "
              f"ACE p={case['ace_prob']}  (edge width = attention)",fontsize=9.5,fontweight="bold")

# ---- (b) counterfactual bars ----
axb=fig.add_subplot(gs[0,1]); cf=case["counterfactual"]
vals=[cf["full"],cf["remove_auditor_channel"],cf["remove_board_ownership"]]
labs=["full\nmodel","− auditor\nchannel","− board &\nownership"]; cols=["#2c3e50","#c0392b","#95a5a6"]
axb.bar(range(3),vals,color=cols,width=0.66)
for i,v in enumerate(vals): axb.text(i,v+0.02,f"{v:.2f}",ha="center",fontsize=10,fontweight="bold")
axb.axhline(0.5,ls=":",lw=1,color="#777"); axb.set_xticks(range(3)); axb.set_xticklabels(labs,fontsize=8.5)
axb.set_ylabel("ACE predicted risk"); axb.set_ylim(0,1.0)
axb.set_title("(b) Model-faithful counterfactual\n(removing the auditor tie collapses the flag)",fontsize=9.5,fontweight="bold")

# ---- (c) prevalence distribution ----
axc=fig.add_subplot(gs[0,2]); drops=np.array(prev["auditor_drops"])
axc.hist(drops,bins=28,color="#c0392b",alpha=0.82,edgecolor="white")
md=prev["auditor_counterfactual_drop"]["median"]
axc.axvline(md,ls="--",lw=1.6,color="#2c3e50"); axc.axvline(0.05,ls=":",lw=1.2,color="#777")
axc.text(md+0.01,axc.get_ylim()[1]*0.92,f"median {md}",fontsize=8,color="#2c3e50")
axc.set_xlabel("auditor-channel counterfactual drop in risk"); axc.set_ylabel("# firms")
share=int(round(prev["share_graph_dependent"]*100)); ngt=prev["auditor_counterfactual_drop"]["n_drop_gt_0.05"]
axc.set_title(f"(c) {prev['n_graph_dependent (own=0, auditor-peer>=1)']} of {prev['n_test_positives']} failures "
              f"({share}%) are 'silent'\nfor {ngt} of them the auditor channel is load-bearing (drop>0.05)",fontsize=9.0,fontweight="bold")
import matplotlib.patches as mp
axa.legend(handles=[mp.Patch(color=RC[r],label=RKO[r]) for r in ["partner","office","auditor"]]+
           [plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='#c0392b',markersize=9,label='restating peer')],
           loc="lower center",bbox_to_anchor=(0.5,-0.17),ncol=2,fontsize=7,frameon=False)
fig.suptitle("Why a graph model, not a tree: per-firm, per-peer, model-faithful explanations a global feature ranking cannot give",
             fontsize=11,fontweight="bold",y=1.04)
plt.tight_layout()
for d in ["figs","paper/Information_Sciences/figures"]:
    os.makedirs(d,exist_ok=True); plt.savefig(f"{d}/fig_interpretability.png",dpi=300,bbox_inches="tight")
plt.close(); print("wrote fig_interpretability  (focal SIC %s, p=%s -> %s after removing auditor)"%(case['focal']['sic'],case['ace_prob'],case['counterfactual']['remove_auditor_channel']))
