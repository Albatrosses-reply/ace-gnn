#!/usr/bin/env python3
"""Precision-recall curves for the primary recent/severe setting (revision, Reviewer 3).
ACE-GNN curve from the saved FINAL_L bagged test predictions; fair random forest
([Xz, OWN], the Table-6 configuration) and exposure-augmented random forest
([Xz, OWN, EXP]) retrained here with the benchmark.py configuration (seeded,
deterministic). Marks the top-1% .. top-10% screening operating points.
Run from repo root:  python3 src/5_figures/make_pr_curves.py
"""
import json, os, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.utils import scatter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False
np.random.seed(0); torch.manual_seed(0)

G = torch.load("data/ext/graph.pt", weights_only=False)
YEARS = G["years"]; yidx = {y: i for i, y in enumerate(YEARS)}
X = G["X"].numpy().copy(); active = G["active"].numpy(); label = G["label_severe"].numpy()
rn = G["restated_now_severe"].numpy().astype(np.float32)
Tall, N, Fdim = X.shape; snaps = G["snapshots"]
REL = ["partner", "office", "auditor", "board", "ownership"]
USE = [yidx[y] for y in range(2017, 2023)]; TR = USE[:3]; VA = [USE[3]]; TE = USE[4:6]

tm = np.zeros((Tall, N), bool)
for t in TR: tm[t] = active[t] & (label[t] >= 0)
Xtr0 = X[tm]; lo = np.nanpercentile(Xtr0, 1, 0); hi = np.nanpercentile(Xtr0, 99, 0); med = np.nanmedian(Xtr0, 0)
Xc = np.clip(X, lo, hi); ix = np.where(np.isnan(Xc)); Xc[ix] = np.take(med, ix[2])
mu = np.nanmean(np.clip(Xtr0, lo, hi), 0); sd = np.nanstd(np.clip(Xtr0, lo, hi), 0); sd[sd < 1e-6] = 1
Xz = ((Xc - mu) / sd).astype(np.float32)

def adj(t, c): ei, w = snaps[t][c]; return ei, w
def mean_agg(vec, ei, w):
    if ei.size(1) == 0: return torch.zeros_like(vec)
    src, dst = ei; v = vec if vec.dim() > 1 else vec.unsqueeze(1)
    num = scatter(v[src] * w.unsqueeze(1), dst, 0, dim_size=N, reduce='sum')
    den = scatter(w, dst, 0, dim_size=N, reduce='sum').clamp(min=1e-6)
    o = num / den.unsqueeze(1); return o if vec.dim() > 1 else o.squeeze(1)

rnt = [torch.from_numpy(rn[t]) for t in range(Tall)]
EXP = np.zeros((Tall, N, len(REL) * 2), np.float32)
for t in range(Tall):
    for ci, c in enumerate(REL):
        ei, w = adj(t, c); EXP[t, :, ci] = mean_agg(rnt[t], ei, w).numpy()
        tl = max(t - 1, 0); eil, wl = adj(tl, c); EXP[t, :, len(REL) + ci] = mean_agg(rnt[tl], eil, wl).numpy()
OWN = np.zeros((Tall, N, 2), np.float32)
for t in range(Tall): OWN[t, :, 0] = rn[t]; OWN[t, :, 1] = rn[max(t - 1, 0)]
FAIR = np.concatenate([Xz, OWN], 2).astype(np.float32)
GF = np.concatenate([Xz, OWN, EXP], 2).astype(np.float32)

def pool(ts, arr):
    xs = []; ys = []
    for t in ts:
        a = np.where(active[t] & (label[t] >= 0))[0]; xs.append(arr[t][a]); ys.append(label[t][a])
    return np.concatenate(xs), np.concatenate(ys)

Xtr_f, ytr = pool(TR, FAIR); Xte_f, yte = pool(TE, FAIR)
Xtr_g, _ = pool(TR, GF); Xte_g, _ = pool(TE, GF)

rf = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=5,
                            class_weight="balanced_subsample", n_jobs=4, random_state=0)
p_rf = rf.fit(Xtr_f, ytr).predict_proba(Xte_f)[:, 1]
rfg = RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                             class_weight="balanced_subsample", n_jobs=4, random_state=0)
p_rfg = rfg.fit(Xtr_g, ytr).predict_proba(Xte_g)[:, 1]

npz = np.load("data/ext/pure/v2/FINAL_L_recent_severe_preds.npz")
y_ace = npz["y"]; p_ace = npz["p"]
assert np.array_equal(y_ace.astype(int), yte.astype(int)), "test-set alignment mismatch"

lb = json.load(open("data/ext/pure/leaderboard_recent_severe.json"))["leaderboard_pure"]
print(f"[check] fair RF  AP={average_precision_score(yte,p_rf):.4f} (leaderboard {lb['RandomForest']['pr']})", file=sys.stderr)
print(f"[check] RF+expo  AP={average_precision_score(yte,p_rfg):.4f} (bench 0.506)", file=sys.stderr)
print(f"[check] ACE bag  AP={average_precision_score(yte,p_ace):.4f} (leaderboard {lb['ACE-GNN (ours)']['pr']})", file=sys.stderr)

def topk_points(y, p, ks):
    o = np.argsort(-p); npos = max(int(y.sum()), 1); out = []
    for k in ks:
        n = max(1, int(round(len(p) * k / 100.0)))
        sel = o[:n]; out.append((float(y[sel].sum()) / npos, float(y[sel].mean())))
    return out  # (recall, precision)

C_OURS = "#c0392b"; C_TAB = "#95a5a6"; C_TABG = "#5d6d7e"
fig, ax = plt.subplots(figsize=(7.2, 5.0))
ks = list(range(1, 11))
for name, y, p, col, ls in [
        ("ACE-GNN (bagged)", yte, p_ace, C_OURS, "-"),
        ("Random forest (fair: ratios + own history)", yte, p_rf, C_TAB, "-"),
        ("Random forest + exposure features", yte, p_rfg, C_TABG, "--")]:
    pr, rc, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    ax.plot(rc, pr, color=col, linestyle=ls, lw=1.8, label=f"{name}  (AUPRC {ap:.3f})")
    pts = topk_points(y, p, ks)
    ax.scatter([r for r, _ in pts], [q for _, q in pts], color=col, s=18, zorder=5)
    for k, (r, q) in zip(ks, pts):
        if k in (1, 5, 10):
            ax.annotate(f"{k}%", (r, q), textcoords="offset points", xytext=(4, 5),
                        fontsize=7.5, color=col)
base = yte.mean()
ax.axhline(base, color="k", lw=0.8, ls=":", alpha=0.6)
ax.text(0.995, base + 0.008, f"prevalence {base:.3f}", ha="right", fontsize=7.5, alpha=0.7)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.0)
ax.set_title("Precision–recall on recent/severe; dots mark top-1% to top-10% screening cutoffs")
ax.legend(loc="upper right", fontsize=8.5, frameon=False)
ax.grid(alpha=0.25, lw=0.5)
for d in ["figs", "paper/Information_Sciences/major/figures"]:
    os.makedirs(d, exist_ok=True)
    fig.savefig(f"{d}/fig_pr_recent_severe.png", dpi=300, bbox_inches="tight")
print("wrote fig_pr_recent_severe.png", file=sys.stderr)
