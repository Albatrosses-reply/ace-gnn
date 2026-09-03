#!/usr/bin/env python3
"""Network-visualization figures for the case studies (the paper's 'pretty' graph figures).
fig18: 2x2 ego-networks for cases A/B/D + office-cluster C.
Run from repo root: python3 src/5_figures/case_figs.py"""
import json, numpy as np, pandas as pd, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"]="DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"]=False
import matplotlib.patches as mpatches
from collections import defaultdict
G=torch.load("data/ext/graph.pt", weights_only=False)
gv=G["gvkeys"]; g2i={g:k for k,g in enumerate(gv)}; YEARS=G["years"]; yidx={y:i for i,y in enumerate(YEARS)}
rn=G["restated_now_severe"].numpy(); snaps=G["snapshots"]; REL=G["relations"]
RC={"partner":"#8e44ad","office":"#e67e22","auditor":"#c0392b","board":"#2980b9","ownership":"#16a085"}
RKO={"partner":"Partner","office":"Office","auditor":"Audit firm","board":"Board","ownership":"Common ownership"}
def neighbors(fi,t,rel):
    ei,_=snaps[t][rel]
    if ei.size(1)==0: return []
    e=ei.numpy(); return [int(j) for j in np.unique(e[1][e[0]==fi])]

def ego(ax,focal_gvkey,year,rels,title,maxper=8):
    t=yidx[year]; fi=g2i[focal_gvkey]
    ax.set_title(title,fontsize=9.5,fontweight="bold"); ax.axis("off"); ax.set_xlim(-1.25,1.25); ax.set_ylim(-1.25,1.25)
    rels=[r for r in rels if neighbors(fi,t,r)]
    nseg=max(len(rels),1)
    for ri,rel in enumerate(rels):
        nb=neighbors(fi,t,rel);
        restating=[j for j in nb if rn[t][j]==1]; other=[j for j in nb if rn[t][j]==0]
        nb=restating+other; nb=nb[:maxper]
        a0=2*np.pi*ri/nseg; aw=2*np.pi/nseg
        for k,j in enumerate(nb):
            ang=a0+aw*(0.15+0.7*(k/(max(len(nb)-1,1))))
            r=0.95; x,y=r*np.cos(ang),r*np.sin(ang)
            isr=rn[t][j]==1
            ax.plot([0,x],[0,y],"-",color=RC[rel],lw=1.6 if isr else 0.7,alpha=0.85 if isr else 0.35,zorder=1)
            ax.scatter([x],[y],s=190 if isr else 90,c=("#c0392b" if isr else "white"),edgecolors=RC[rel],
                       linewidths=2.0 if isr else 1.3,zorder=3)
            if isr: ax.scatter([x],[y],s=420,facecolors="none",edgecolors="#c0392b",lw=0.8,alpha=0.5,zorder=2)
    ax.scatter([0],[0],s=520,c="#f1c40f",edgecolors="black",linewidths=1.6,zorder=4,marker="o")
    ax.text(0,0,"Focal",ha="center",va="center",fontsize=7,fontweight="bold",zorder=5)

fig,axes=plt.subplots(2,2,figsize=(12,11))
C=json.load(open("data/ext/case_studies.json"))
# A
a=C["A_auditor_contagion_catch"]
ego(axes[0,0],a["focal"]["gvkey"],a["year"],["auditor","partner","office","board","ownership"],
    "(A) Auditor-channel contagion: pharma (SIC 2834), FY2021; ACE p=0.92")
# B
b=C["B_shared_partner"]
ego(axes[0,1],b["focal"]["gvkey"],b["year"],["partner","auditor","office","board","ownership"],
    "(B) Shared engagement partner: retail (SIC 5712), FY2022; ACE p=0.93")
# D
d=C["D_selective_noncontagion"]
ego(axes[1,1],d["focal"]["gvkey"],d["year"],["ownership","board","auditor","office","partner"],
    "(D) Selective non-contagion: railroad (SIC 4011), FY2022; ACE p=0.004")
# C: office cluster (special: hub-and-spoke)
axC=axes[1,0]; axC.axis("off"); axC.set_xlim(-1.25,1.25); axC.set_ylim(-1.3,1.25)
cc=C["C_office_cluster"]; t=yidx[cc["year"]]
sox=pd.read_pickle("data/ext/sox404.pkl"); comp=pd.read_pickle("data/legacy/comp_company.pkl").dropna(subset=["cik"]); comp["cik"]=comp["cik"].astype("int64")
cik2g={int(c):g for g,c in zip(comp["gvkey"],comp["cik"])}
sub=sox[(sox["fy"]==cc["year"])&(sox["auditor_fkey"]==cc["office"]["auditor_fkey"])&(sox["aud_city"].astype(str).str.upper().str.strip()==cc["office"]["city"])&sox["cik"].notna()]
members=sorted({cik2g.get(int(r.cik)) for r in sub.itertuples() if cik2g.get(int(r.cik)) in g2i})
mi=[g2i[m] for m in members if m]; restating=[j for j in mi if rn[t][j]==1]; other=[j for j in mi if rn[t][j]==0]
mi=restating+other
n=len(mi)
for k,j in enumerate(mi):
    ang=2*np.pi*k/n; r=1.0; x,y=r*np.cos(ang),r*np.sin(ang); isr=rn[t][j]==1
    axC.plot([0,x],[0,y],"-",color="#e67e22",lw=1.6 if isr else 0.5,alpha=0.85 if isr else 0.25,zorder=1)
    axC.scatter([x],[y],s=150 if isr else 45,c=("#c0392b" if isr else "white"),edgecolors="#e67e22",linewidths=1.8 if isr else 1.0,zorder=3)
axC.scatter([0],[0],s=620,c="#e67e22",edgecolors="black",linewidths=1.6,zorder=4,marker="s")
axC.text(0,0,"Audit\noffice",ha="center",va="center",fontsize=7.5,fontweight="bold",color="white",zorder=5)
axC.set_title("(C) Audit-office cluster: a Big-4 office, 2021; 5 of ~40 clients restate",fontsize=9.5,fontweight="bold")

# shared legend
handles=[mpatches.Patch(color=RC[r],label=RKO[r]) for r in REL]+[
    plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='#c0392b',markersize=10,label='Restating neighbor'),
    plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='#f1c40f',markeredgecolor='k',markersize=11,label='Focal firm')]
fig.legend(handles=handles,loc="lower center",ncol=7,fontsize=8.5,frameon=False,bbox_to_anchor=(0.5,-0.01))
fig.suptitle("Auditor-network contagion case studies: multiplex ego-networks",fontsize=12,fontweight="bold")
plt.tight_layout(rect=(0,0.03,1,0.96))
import os as _os
for _d in ["figs","paper/Information_Sciences/figures"]:
    _os.makedirs(_d,exist_ok=True); plt.savefig(f"{_d}/fig18_case_studies.png",dpi=160,bbox_inches="tight")
plt.close(); print("fig18 done")
# (fig19 standalone hero figure removed: the office cluster is panel C of fig18; identifying detail de-emphasized)
