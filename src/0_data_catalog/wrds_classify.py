#!/usr/bin/env python3
"""Step 2: classify libraries by table count (populated vs empty/sample),
and dump table lists for GNN-relevant candidate libraries. One connection."""
import os
import json, sys, time
import wrds

t0 = time.time()
db = wrds.Connection(wrds_username=os.environ["WRDS_USERNAME"])
print(f"[connected {time.time()-t0:.1f}s]", file=sys.stderr)

# 1) table+view counts per schema in ONE query (fast)
q = """
SELECT table_schema, table_type, count(*) AS n
FROM information_schema.tables
GROUP BY table_schema, table_type
ORDER BY table_schema, table_type
"""
df = db.raw_sql(q)
counts = {}
for _, r in df.iterrows():
    s = r["table_schema"]; counts.setdefault(s, {})[r["table_type"]] = int(r["n"])
with open("wrds_schema_counts.json", "w") as f:
    json.dump(counts, f, indent=2)
print(f"[schemas with tables/views: {len(counts)}]", file=sys.stderr)

# 2) For GNN-relevant FULL-access candidate libs, list tables
candidates = [
    "boardex", "boardex_na",          # board interlock / director networks
    "tfn", "tr_13f", "tr_common",     # institutional holdings (ownership graph)
    "ibes", "tr_ibes",                # analyst coverage (analyst-firm bipartite)
    "trace", "trace_enhanced",        # bond dealer trading networks
    "wrdsapps_patents",               # patent citation / inventor networks
    "sdc", "tr_sdc_ma",               # M&A acquirer-target networks
    "wrdsapps_subsidiary",            # parent-subsidiary structure
    "comp", "comp_segments_hist_daily", "compseg",  # supply-chain segments / fundamentals
    "crsp", "crsp_a_stock",           # securities / returns (node features)
    "execcomp", "comp_execucomp",     # executives
    "fjc_litigation", "fjc_linking",  # litigation networks
    "msrb", "otc",                    # municipal / OTC
    "wrdsapps_finratio",              # firm financial ratios (node features)
]
tables = {}
for lib in candidates:
    try:
        ts = db.list_tables(library=lib)
        tables[lib] = sorted(ts)
    except Exception as e:
        tables[lib] = [f"ERROR: {e}"]
with open("wrds_candidate_tables.json", "w") as f:
    json.dump(tables, f, indent=2)
for lib in candidates:
    print(f"{lib}: {len(tables[lib])} tables -> {tables[lib][:12]}")

db.close()
print(f"[done {time.time()-t0:.1f}s]", file=sys.stderr)
