#!/usr/bin/env python3
"""Model-FAITHFUL case studies = the 'why GNN, not RF' evidence.

For each focal firm ACE-GNN flags, we extract three things RF structurally cannot give:
  (1) per-firm CHANNEL GATE  g_i = softmax(Gate(h_i)) over {auditor, board, ownership}
      -> an instance-specific relation routing (RF importance is one GLOBAL vector for all firms);
  (2) edge-level ATTENTION over specific neighbours in the auditor channel
      -> which peer drove the flag (RF sees only an exposure count);
  (3) edge-removal COUNTERFACTUAL: prediction with the auditor channel removed vs board/ownership removed
      -> a model-faithful causal attribution of the flag to a relation type.
We contrast with a fair RandomForest whose global feature importances cannot localise to a firm or peer.
Trains one representative ACE-GNN (winning config: PLE + encoded exposures + relation-gated attention).
Run: python3 src/4_analysis/case_study_v2.py
"""
import json, os, sys
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.utils import scatter, softmax as gsoftmax
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(6)

NB=32; FEMB=12; NEB=8; H=160; TABM=6; ENC_DEPTH=2; EPMAX=200; DROP=0.15
G=torch.load("data/ext/graph.pt", weights_only=False)
gv=G["gvkeys"]; YEARS=G["years"]; yidx={y:i for i,y in enumerate(YEARS)}
X=G["X"].numpy().copy(); active=G["active"].numpy(); label=G["label_severe"].numpy()
rn=G["restated_now_severe"].numpy().astype(np.float32); snaps=G["snapshots"]
Tall,N,Fdim=X.shape
AUD=["partner","office","auditor"]; OTH=["board","ownership"]; ALL=AUD+OTH
USE=[yidx[y] for y in range(2017,2023)]; TR=USE[:3]; VA=[USE[3]]; TE=USE[4:6]
g2i={g:k for k,g in enumerate(gv)}

tm=np.zeros((Tall,N),bool)
for t in TR: tm[t]=active[t]&(label[t]>=0)
Xtr0=X[tm]; lo=np.nanpercentile(Xtr0,1,0); hi=np.nanpercentile(Xtr0,99,0); med=np.nanmedian(Xtr0,0)
Xc=np.clip(X,lo,hi); ix=np.where(np.isnan(Xc)); Xc[ix]=np.take(med,ix[2])
mu=np.nanmean(np.clip(Xtr0,lo,hi),0); sd=np.nanstd(np.clip(Xtr0,lo,hi),0); sd[sd<1e-6]=1
Xz=((Xc-mu)/sd).astype(np.float32)
knots=[]
for j in range(Fdim):
    q=np.quantile(Xz[tm][:,j],np.linspace(0,1,NB+1)); q[0]=-1e9; q[-1]=1e9
    knots.append(np.maximum.accumulate(q).astype(np.float32))
def ple(Xs):
    return np.concatenate([np.clip((Xs[:,j:j+1]-knots[j][:-1][None,:])/(knots[j][1:][None,:]-knots[j][:-1][None,:]+1e-9),0.,1.).astype(np.float32) for j in range(Fdim)],1)
PLE_t={t:torch.from_numpy(ple(Xz[t])) for t in USE}
def adj(t,c): ei,w=snaps[t][c]; return ei,w
def mean_agg(v,ei,w):
    if ei.size(1)==0: return torch.zeros_like(v)
    s,d=ei; num=scatter(v[s]*w,d,0,dim_size=N,reduce='sum'); den=scatter(w,d,0,dim_size=N,reduce='sum').clamp(min=1e-6); return num/den
rnt=[torch.from_numpy(rn[t]) for t in range(Tall)]
EXP=np.zeros((Tall,N,len(ALL)*2),np.float32)
for t in range(Tall):
    for ci,c in enumerate(ALL):
        ei,w=adj(t,c); EXP[t,:,ci]=mean_agg(rnt[t],ei,w).numpy()
        tl=max(t-1,0); eil,wl=adj(tl,c); EXP[t,:,len(ALL)+ci]=mean_agg(rnt[tl],eil,wl).numpy()
# EXPENC encode
eknots=[]
for j in range(EXP.shape[2]):
    v=EXP[:,:,j][tm]; vp=v[v>1e-9]
    if len(vp)<50: q=np.linspace(1e-9,1.0,NEB+1)
    else: q=np.quantile(vp,np.linspace(0,1,NEB+1)); q[0]=1e-9; q[-1]=max(float(q[-1]),1.0)+1e-6
    eknots.append(np.maximum.accumulate(q).astype(np.float32))
def expenc_row(E):
    parts=[]
    for j in range(E.shape[1]):
        x=E[:,j:j+1]; kb=eknots[j]; flag=(x>1e-9).astype(np.float32)
        parts.append(np.concatenate([flag,np.clip((x-kb[:-1][None,:])/(kb[1:][None,:]-kb[:-1][None,:]+1e-9),0.,1.).astype(np.float32)*flag],1))
    return np.concatenate(parts,1)
EXPE={t:torch.from_numpy(expenc_row(EXP[t])) for t in USE}
OWN=np.zeros((Tall,N,2),np.float32)
for t in range(Tall): OWN[t,:,0]=rn[t]; OWN[t,:,1]=rn[max(t-1,0)]
OWN_t=torch.from_numpy(OWN)
A=[{c:adj(t,c) for c in ALL} for t in range(Tall)]
lab_t=torch.from_numpy(label)
ytr=np.concatenate([label[t][active[t]&(label[t]>=0)] for t in TR]); spw=float((ytr==0).sum()/max((ytr==1).sum(),1))
POSW=torch.tensor([float(np.sqrt(spw))])
expdim=EXP.shape[2]*(1+NEB); aux_dim=2+expdim; K_ENS=TABM

class BELin(nn.Module):
    def __init__(s,di,do):
        super().__init__(); s.W=nn.Linear(di,do)
        s.r=nn.Parameter(torch.ones(K_ENS,di)+0.05*torch.randn(K_ENS,di)); s.sd=nn.Parameter(torch.ones(K_ENS,do)+0.05*torch.randn(K_ENS,do)); s.bm=nn.Parameter(torch.zeros(K_ENS,do))
    def forward(s,x): return s.W(x*s.r.unsqueeze(1))*s.sd.unsqueeze(1)+s.bm.unsqueeze(1)

class ACE(nn.Module):
    def __init__(s):
        super().__init__(); s.femb=nn.Linear(NB,FEMB); din=Fdim*FEMB+aux_dim
        s.inp=BELin(din,H); s.ln=nn.ModuleList([nn.LayerNorm(H) for _ in range(ENC_DEPTH)])
        s.f1=nn.ModuleList([BELin(H,2*H) for _ in range(ENC_DEPTH)]); s.f2=nn.ModuleList([BELin(2*H,H) for _ in range(ENC_DEPTH)])
        s.hln=nn.LayerNorm(H); s.head=BELin(H,1); s.lvl=nn.Parameter(torch.zeros(len(AUD)))
        s.nchan=1+len(OTH); s.asrc=nn.ModuleList([nn.Linear(H,1) for _ in range(s.nchan)]); s.adst=nn.ModuleList([nn.Linear(H,1) for _ in range(s.nchan)])
        s.lin=nn.ModuleList([nn.Linear(H,H) for _ in range(s.nchan)]); s.gate=nn.Linear(H,s.nchan)
        s.gh=nn.Sequential(nn.LayerNorm(H),nn.Linear(H,1)); s.gamma=nn.Parameter(torch.tensor(0.2)); s.drop=nn.Dropout(DROP)
    def lvlw(s): w=F.softplus(s.lvl); return w/w.max().clamp(min=1e-6)
    def feat(s,t,exp_override=None):
        p=s.femb(PLE_t[t].view(N,Fdim,NB)).reshape(N,Fdim*FEMB)
        ex=EXPE[t] if exp_override is None else exp_override
        return torch.cat([p,OWN_t[t],ex],1)
    def attn(s,k,h,ei,w,return_alpha=False):
        if ei.size(1)==0: return (torch.zeros_like(h),None) if return_alpha else torch.zeros_like(h)
        src,dst=ei; hs=s.lin[k](h)
        e=F.leaky_relu(s.asrc[k](hs)[src]+s.adst[k](hs)[dst]).squeeze(-1)+torch.log(w.clamp(min=1e-6))
        al=gsoftmax(e,dst,num_nodes=N); out=scatter(al.unsqueeze(1)*hs[src],dst,0,dim_size=N,reduce='sum')
        return (out,al) if return_alpha else out
    def forward(s,t,exp_override=None,adj_override=None):
        x=s.feat(t,exp_override).unsqueeze(0).expand(K_ENS,N,-1); h=s.inp(x)
        for l in range(ENC_DEPTH): h=h+s.f2[l](s.drop(F.gelu(s.f1[l](s.ln[l](h)))))
        z=s.head(s.hln(h)).squeeze(-1).mean(0); hm=s.hln(h).mean(0)
        Acur=A[t] if adj_override is None else adj_override
        w=s.lvlw(); audm=0
        for li,c in enumerate(AUD): ei,ew=Acur[c]; audm=audm+w[li]*s.attn(0,hm,ei,ew)
        chans=[audm]+[s.attn(1+j,hm,*Acur[c]) for j,c in enumerate(OTH)]
        g=torch.softmax(s.gate(hm),1)
        agg=sum(g[:,k:k+1]*chans[k] for k in range(s.nchan))
        z=z+s.gamma*s.gh(agg).squeeze(1)
        return z,g

def collect(m,ts):
    ys=[];ps=[]
    with torch.no_grad():
        for t in ts:
            zz,_=m(t); a=np.where(active[t]&(label[t]>=0))[0]; ps.append(torch.sigmoid(zz[a]).numpy()); ys.append(label[t][a])
    return np.concatenate(ys),np.concatenate(ps)

# ---- train one representative model ----
m=ACE(); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=3e-5)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPMAX)
tri={t:torch.from_numpy(np.where(active[t]&(label[t]>=0))[0]).long() for t in TR}
best=-1;bs=None;bad=0
for ep in range(EPMAX):
    m.train();opt.zero_grad();loss=0
    for t in TR:
        z,_=m(t); idx=tri[t]; loss=loss+F.binary_cross_entropy_with_logits(z[idx],lab_t[t][idx].float(),pos_weight=POSW)
    loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5.0); opt.step(); sched.step()
    if ep%5==0:
        m.eval(); yv,pv=collect(m,VA); vp=average_precision_score(yv,pv)
        if vp>best+1e-4: best=vp; bs={k:v.clone() for k,v in m.state_dict().items()};bad=0
        else: bad+=1
        if bad>10: break
if bs: m.load_state_dict(bs)
m.eval()
yte,pte=collect(m,TE); print(f"[case-model] val_pr={best:.4f} test_pr={average_precision_score(yte,pte):.4f} test_roc={roc_auc_score(yte,pte):.4f}",file=sys.stderr)

# ---- predictions per (node,year) on test ----
P={}; GATE={}
with torch.no_grad():
    for t in TE:
        z,g=m(t); p=torch.sigmoid(z).numpy()
        for i in np.where(active[t]&(label[t]>=0))[0]: P[(i,t)]=float(p[i]); GATE[(i,t)]=g[i].numpy().tolist()

# ---- attribution helpers ----
CH=["auditor","board","ownership"]
def neigh(i,t,rel):
    ei,_=snaps[t][rel]
    if ei.size(1)==0: return []
    e=ei.numpy(); return [int(j) for j in np.unique(e[1][e[0]==i])]
def aud_attention(i,t,topk=5):
    """top source neighbours into i across auditor relations, by learned attention alpha."""
    out=[]
    with torch.no_grad():
        x=m.feat(t).unsqueeze(0).expand(K_ENS,N,-1); h=m.inp(x)
        for l in range(ENC_DEPTH): h=h+m.f2[l](F.gelu(m.f1[l](m.ln[l](h))))
        hm=m.hln(h).mean(0)
        for c in AUD:
            ei,ew=A[t][c]
            if ei.size(1)==0: continue
            _,al=m.attn(0,hm,ei,ew,return_alpha=True)
            src,dst=ei.numpy(); al=al.numpy()
            for e_ in np.where(dst==i)[0]:
                out.append((float(al[e_]),int(src[e_]),c))
    agg={}
    for a_,j,c in out: agg[j]=max(agg.get(j,(0,c)),(a_,c))
    rows=sorted(([a_,j,c] for j,(a_,c) in agg.items()),reverse=True)[:topk]
    return rows
def counterfactual(i,t,zero_rels):
    """prediction for i with channel(s) removed: zero those exposure cols + drop incoming edges."""
    ex=EXPE[t].clone()
    # EXPENC columns: per relation in ALL, (1+NEB) cols, x2 lags. Zero the rels in zero_rels.
    blk=1+NEB; nrel=len(ALL)
    for lag in range(2):
        for ci,c in enumerate(ALL):
            if c in zero_rels:
                st=(lag*nrel+ci)*blk; ex[i,st:st+blk]=0.0
    ado={}
    for c in ALL:
        ei,w=A[t][c]
        if c in zero_rels and ei.size(1)>0:
            keep=ei[1]!=i; ado[c]=(ei[:,keep],w[keep])
        else: ado[c]=(ei,w)
    with torch.no_grad():
        z,_=m(t,exp_override=ex,adj_override=ado); return float(torch.sigmoid(z)[i])

# ---- fair RF for contrast (global importances only) ----
def pool(ts,arr):
    xs=[];ys=[]
    for t in ts: a=np.where(active[t]&(label[t]>=0))[0]; xs.append(arr[t][a]); ys.append(label[t][a])
    return np.concatenate(xs),np.concatenate(ys)
GF=np.concatenate([Xz,OWN,EXP],2).astype(np.float32)
Xtr_g,ytr2=pool(TR,GF); rf=RandomForestClassifier(n_estimators=400,min_samples_leaf=5,class_weight="balanced_subsample",n_jobs=6,random_state=0).fit(Xtr_g,ytr2)
fnames=[f"feat{j}" for j in range(Fdim)]+["own_t","own_t-1"]+[f"exp_{c}_t" for c in ALL]+[f"exp_{c}_t-1" for c in ALL]
imp=sorted(zip(rf.feature_importances_,fnames),reverse=True)[:8]

# ---- entity metadata ----
comp=pd.read_pickle("data/legacy/comp_company.pkl"); g2cik=dict(zip(comp["gvkey"],comp["cik"])); g2sic=dict(zip(comp["gvkey"],comp["sic"]))
def info(i):
    sic=g2sic.get(gv[i]); cik=g2cik.get(gv[i])
    return {"gvkey":gv[i],"cik":(int(cik) if pd.notna(cik) else -1),"sic":(int(float(sic)) if pd.notna(sic) else None)}

# ---- GRAPH-DEPENDENT cases only: firm has NO own problem this year (rn=0) but restating auditor peers.
# These are the only firms where the flag can come from the network rather than the firm's own history. ----
graph_pos=[]   # true positives, own_t==0, >=1 restating auditor peer
for (i,t),p in P.items():
    if label[t][i]==1 and rn[t][i]==0:
        aexp=len([j for j in neigh(i,t,"auditor")+neigh(i,t,"office")+neigh(i,t,"partner") if rn[t][j]==1])
        if aexp>=1: graph_pos.append((p,i,t,aexp))
graph_pos.sort(reverse=True)
# prevalence + how often the auditor channel is actually load-bearing (counterfactual drop > 0.05)
n_pos=int(sum(1 for (i,t) in P if label[t][i]==1)); n_graph=len(graph_pos)
drops=[]
for p,i,t,ae in graph_pos:
    cf=counterfactual(i,t,set(AUD)); drops.append(p-cf)
drops=np.array(drops) if drops else np.array([0.0])
PREVALENCE={"n_test_positives":n_pos,"n_graph_dependent (own=0, auditor-peer>=1)":n_graph,
            "share_graph_dependent":round(n_graph/max(n_pos,1),3),
            "auditor_counterfactual_drop":{"median":round(float(np.median(drops)),3),"p90":round(float(np.percentile(drops,90)),3),"max":round(float(drops.max()),3),
                                           "n_drop_gt_0.05":int((drops>0.05).sum()),"n_drop_gt_0.10":int((drops>0.10).sum())},
            "auditor_drops":[round(float(x),4) for x in drops]}
print("[prevalence]",json.dumps(PREVALENCE),file=sys.stderr)
# focal cases = the graph-dependent firms where the auditor counterfactual drop is LARGEST (graph genuinely drives the flag)
cand=sorted([(p-counterfactual(i,t,set(AUD)),p,i,t,ae) for p,i,t,ae in graph_pos[:40]],reverse=True)
cand=[(p,i,t,ae) for _,p,i,t,ae in cand]
CASES={"_model":{"val_pr":round(best,4),"test_pr":round(float(average_precision_score(yte,pte)),4),"test_roc":round(float(roc_auc_score(yte,pte)),4)},
       "_prevalence":PREVALENCE,
       "_rf_global_importance":[[round(float(v),4),n] for v,n in imp],
       "_note":"RF importance is ONE global ranking shared by all firms; it cannot produce the per-firm gate, per-peer attention, or edge-removal counterfactual below."}
for rank,(p,i,t,aexp) in enumerate(cand[:3]):
    g=GATE[(i,t)]
    att=aud_attention(i,t)
    cf_aud=counterfactual(i,t,set(AUD)); cf_bo=counterfactual(i,t,set(OTH))
    CASES[f"case{rank+1}"]={
        "focal":info(i),"year":int(YEARS[t]),"label":1,"ace_prob":round(p,3),
        "channel_gate":{CH[k]:round(float(g[k]),3) for k in range(3)},
        "top_attended_auditor_peers":[{"alpha":round(a_,3),"relation":c,"restating":int(rn[t][j]==1),**info(j)} for a_,j,c in att],
        "counterfactual":{"full":round(p,3),"remove_auditor_channel":round(cf_aud,3),"remove_board_ownership":round(cf_bo,3),
                          "drop_if_auditor_removed":round(p-cf_aud,3),"drop_if_BO_removed":round(p-cf_bo,3)},
        "why_gnn":"RF gives the same global ranking to every firm; ACE attributes THIS flag to the auditor channel (gate+counterfactual) and to SPECIFIC restating peers (attention)."}
os.makedirs("data/ext",exist_ok=True)
json.dump(CASES,open("data/ext/case_studies_v2.json","w"),indent=2,ensure_ascii=False)
print(json.dumps(CASES,ensure_ascii=False,indent=1)[:2600])
print("\nsaved -> data/ext/case_studies_v2.json")
