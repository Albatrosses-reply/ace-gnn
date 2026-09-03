#!/usr/bin/env python3
"""Rebuild ONLY the ownership layer of data/ext/graph.pt at a different shared-owner
threshold (OWN_MIN), swap it into a copy of the canonical graph, and save the variant.
Used for the ownership-threshold sensitivity analysis (revision, Reviewer 3 comment 2).
Replicates the ownership kNN block of build_graph_ext.py exactly (same constants,
same tie-breaking, same to_ei edge encoding).

ENV: OWN_MIN=3|4|5   OUT=data/ext/graph_own<OWN_MIN>.pt   VERIFY=1
With VERIFY=1 and OWN_MIN=4 the rebuilt layer is asserted equal to the canonical one
(exact edge/weight sets per year) and nothing is saved.
Run from repo root:  OWN_MIN=3 python3 src/2_graph/build_own_variant.py
"""
import os, sys
import numpy as np, pandas as pd, torch, scipy.sparse as sp
from collections import defaultdict

OWN_MIN = int(os.environ.get("OWN_MIN", "4"))
OWN_TOPK = 15; OWN_SKIP = 1000                      # as in build_graph_ext.py
OUT = os.environ.get("OUT", f"data/ext/graph_own{OWN_MIN}.pt")
VERIFY = int(os.environ.get("VERIFY", "0"))

G = torch.load("data/ext/graph.pt", weights_only=False)
gvkeys = G["gvkeys"]; YEARS = G["years"]; N = len(gvkeys)
gid = {g: i for i, g in enumerate(gvkeys)}

fr = pd.read_pickle("data/ext/firm_ratio.pkl"); fr = fr[fr["year"].isin(YEARS)]
assert sorted(fr["gvkey"].unique()) == list(gvkeys), "gvkey universe mismatch vs graph.pt"
cy2g = {}
for r in fr[["gvkey", "cusip", "year"]].dropna().itertuples():
    if isinstance(r.cusip, str) and len(r.cusip) == 8:
        cy2g[(r.cusip, int(r.year))] = r.gvkey

hold = pd.read_pickle("data/ext/holdings.pkl"); hold["mgrno"] = hold["mgrno"].astype(str)

def to_ei(edic):
    if not edic:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,), dtype=torch.float)
    es = list(edic.items())
    ei = torch.tensor([[a for (a, b), _ in es] + [b for (a, b), _ in es],
                       [b for (a, b), _ in es] + [a for (a, b), _ in es]], dtype=torch.long)
    w = torch.tensor([w for _, w in es] + [w for _, w in es], dtype=torch.float)
    return ei, w

ownE = {y: defaultdict(float) for y in YEARS}
for y in YEARS:
    sub = hold[hold["year"] == y].drop_duplicates(["mgrno", "cusip"]); firms = []; mgrs = []; mid = {}
    for r in sub.itertuples():
        g = cy2g.get((r.cusip, y))
        if g is None: continue
        firms.append(gid[g]); mgrs.append(mid.setdefault(r.mgrno, len(mid)))
    if not firms: continue
    M = sp.csr_matrix((np.ones(len(firms)), (firms, mgrs)), shape=(N, len(mid)))
    cs = np.asarray(M.sum(0)).ravel(); M = M[:, cs <= OWN_SKIP]; C = (M @ M.T).tocoo()
    rows = defaultdict(list)
    for a, b, c in zip(C.row, C.col, C.data):
        if a < b and c >= OWN_MIN: rows[a].append((c, b)); rows[b].append((c, a))
    for a, lst in rows.items():
        lst.sort(reverse=True)
        for c, b in lst[:OWN_TOPK]:
            e = (a, b) if a < b else (b, a); ownE[y][e] = max(ownE[y][e], float(c))

covA = 0; covB = 0
for yi, y in enumerate(YEARS):
    ei, w = to_ei(ownE[y])
    if VERIFY:
        ei0, w0 = G["snapshots"][yi]["ownership"]
        A = set(zip(ei0[0].tolist(), ei0[1].tolist(), [round(float(x), 4) for x in w0.tolist()]))
        B = set(zip(ei[0].tolist(), ei[1].tolist(), [round(float(x), 4) for x in w.tolist()]))
        assert A == B, f"mismatch in year {y}: canonical {len(A)} vs rebuilt {len(B)}, sym-diff {len(A ^ B)}"
    G["snapshots"][yi]["ownership"] = (ei, w)
    act = G["active"][yi].numpy(); has = np.zeros(N, bool)
    if ei.numel(): has[np.unique(ei[0].numpy())] = True
    covA += int(act.sum()); covB += int((has & act).sum())
    print(f"[own{OWN_MIN} {y}] edges(undirected)={len(ownE[y])} cov={(has & act).sum() / max(act.sum(), 1):.3f}",
          file=sys.stderr)

print(f"[own{OWN_MIN}] overall active firm-year coverage {covB}/{covA} = {covB / covA:.4f}", file=sys.stderr)
if VERIFY:
    print("[VERIFY OK] rebuilt ownership layer matches canonical graph.pt exactly", file=sys.stderr)
else:
    torch.save(G, OUT)
    print(f"[saved -> {OUT}]", file=sys.stderr)
