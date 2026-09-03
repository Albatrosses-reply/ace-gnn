#!/usr/bin/env python3
"""English regeneration of Fig. (hierarchy weights) for the Information Sciences submission.
Reads the learned auditor-level weights from data/ext/hat_recent_severe.json and renders an
English, ACE-GNN-labelled bar chart. Writes to both figs/ and the paper figures dir."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

d = json.load(open("data/ext/hat_recent_severe.json"))["models"]
lv = ["partner", "office", "auditor"]
labels = ["Partner (L1)", "Office (L2)", "Audit firm (L3)"]
full = [d["HAT-GNN_full"]["aud_level_weights"][k] for k in lv]
flat = [d["HAT_flat(no-hierarchy)"]["aud_level_weights"][k] for k in lv]

fig, ax = plt.subplots(figsize=(8, 4.8))
x = np.arange(3); w = 0.38
ax.bar(x - w/2, full, w, label="ACE-GNN (monotone prior)", color="#c0392b")
ax.bar(x + w/2, flat, w, label="Ablated: free weights (no monotone prior)", color="#95a5a6")
for i, (a, b) in enumerate(zip(full, flat)):
    ax.text(i - w/2, a + .02, f"{a:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.text(i + w/2, b + .02, f"{b:.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Learned auditor-level weight")
ax.set_ylim(0, 1.18); ax.legend(loc="upper right", fontsize=9)
ax.set_title("Learned auditor-hierarchy weights (recent/severe)\n"
             "ACE-GNN recovers a monotone partner > office > firm ordering; "
             "removing the prior yields near-uniform weights",
             fontsize=10.5, fontweight="bold")
plt.tight_layout()
for outdir in ["figs", "paper/Information_Sciences/figures"]:
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/fig9_hierarchy_weights.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig9 (English) regenerated; weights full=%s flat=%s" % (full, flat))
