#!/usr/bin/env python3
"""Step 3: extract full column structure + safe samples for GNN-relevant
key tables, in ONE persistent connection. Output -> wrds_attributes.json"""
import os
import json, sys, time, decimal, datetime
import wrds
from sqlalchemy import text

# (group, schema, table, sample?) — curated high-value tables for GNN
KEY = [
  # ---- BOARD INTERLOCK (BoardEx NA) ----
  ("board","boardex_na","na_wrds_org_composition",True),
  ("board","boardex_na","na_wrds_company_networks",True),
  ("board","boardex_na","na_wrds_individual_networks",True),
  ("board","boardex_na","na_wrds_dir_profile_emp",True),
  ("board","boardex_na","na_dir_profile_details",True),
  ("board","boardex_na","na_wrds_company_profile",True),
  ("board","boardex_na","na_company_profile_stocks",True),
  ("board","boardex_na","na_board_characteristics",True),
  # ---- INSTITUTIONAL OWNERSHIP (13F) ----
  ("ownership","tr_13f","s34",True),
  ("ownership","tr_13f","s34names",True),
  ("ownership","tr_13f","s34type3",True),
  # ---- ANALYST COVERAGE (IBES) ----
  ("analyst","ibes","id",True),
  ("analyst","ibes","detu_epsus",True),
  ("analyst","ibes","recddet",True),
  ("analyst","ibes","recdsum",True),
  # ---- PATENT NETWORKS ----
  ("patent","wrdsapps_patents","uspatents_meta",True),
  ("patent","wrdsapps_patents","uspatents_citations",True),
  ("patent","wrdsapps_patents","uspatents_gvkey_linking",True),
  # ---- SUPPLY CHAIN (Compustat segments) ----
  ("supplychain","compseg","wrds_seg_customer",True),
  ("supplychain","compseg","seg_customer",True),
  ("supplychain","compseg","names_seg",True),
  # ---- M&A NETWORKS (SDC) ----
  ("mna","tr_sdc_ma","wrds_ma_details",True),
  ("mna","tr_sdc_ma","wrds_ma_advisors",True),
  ("mna","tr_sdc_ma","wrds_ma_related",True),
  # ---- SUBSIDIARY STRUCTURE ----
  ("subsidiary","wrdsapps_subsidiary","wrds_relationships",True),
  ("subsidiary","wrdsapps_subsidiary","chars",True),
  # ---- LITIGATION NETWORKS ----
  ("litigation","fjc_litigation","civil",True),
  ("litigation","fjc_litigation","criminal",True),
  ("litigation","fjc_linking","wrds_civil_link",True),
  # ---- EXECUTIVES ----
  ("execs","execcomp","anncomp",True),
  ("execs","execcomp","coperol",True),
  ("execs","execcomp","person",True),
  # ---- NODE FEATURES / LINKING BACKBONE ----
  ("backbone","crsp","stocknames",True),
  ("backbone","crsp","ccmxpf_lnkhist",True),
  ("backbone","crsp","crsp_cik_map",True),
  ("backbone","comp","company",True),
  ("backbone","comp","funda",False),   # huge: columns only
  ("backbone","comp","security",True),
  ("backbone","wrdsapps_finratio","firm_ratio",True),
  ("backbone","wrdsapps_finratio","id",True),
  # ---- BOND DEALER NETWORK (TRACE) ----
  ("bond","trace","trace_enhanced",False),  # huge: columns only
]

def jsonable(v):
    if isinstance(v,(decimal.Decimal,)): return float(v)
    if isinstance(v,(datetime.date,datetime.datetime)): return str(v)
    if isinstance(v,(bytes,bytearray)): return "<bytes>"
    return v

t0=time.time()
db=wrds.Connection(wrds_username=os.environ["WRDS_USERNAME"])
print(f"[connected {time.time()-t0:.1f}s]",file=sys.stderr)
# persistent connection -> set a guard timeout so nothing hangs
try:
    db.connection.execute(text("SET statement_timeout='25000'"))
except Exception as e:
    print(f"[timeout set warn: {e}]",file=sys.stderr)

# 1) columns for all curated tables in ONE query (row-wise IN)
combos=",".join("('%s','%s')"%(s,t) for _,s,t,_ in KEY)
colq=f"""
SELECT table_schema, table_name, ordinal_position, column_name, data_type
FROM information_schema.columns
WHERE (table_schema, table_name) IN ({combos})
ORDER BY table_schema, table_name, ordinal_position
"""
cols=db.raw_sql(colq)
print(f"[columns rows: {len(cols)}]",file=sys.stderr)

out={}
for grp,sch,tbl,_ in KEY:
    sub=cols[(cols.table_schema==sch)&(cols.table_name==tbl)]
    out.setdefault(grp,{})[f"{sch}.{tbl}"]={
        "schema":sch,"table":tbl,"group":grp,
        "n_columns":int(len(sub)),
        "columns":[{"pos":int(r.ordinal_position),"name":r.column_name,"type":r.data_type}
                   for r in sub.itertuples()],
        "sample":None,"sample_error":None,"est_rows":None,
    }

# 2) row estimate via pg_class.reltuples (instant, best-effort)
estq=f"""
SELECT n.nspname AS s, c.relname AS t, c.reltuples::bigint AS est, c.relkind AS kind
FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
WHERE (n.nspname,c.relname) IN ({combos})
"""
try:
    est=db.raw_sql(estq)
    estmap={(r.s,r.t):(int(r.est),r.kind) for r in est.itertuples()}
    for grp in out:
        for k,v in out[grp].items():
            key=(v["schema"],v["table"])
            if key in estmap:
                v["est_rows"]=estmap[key][0]; v["relkind"]=estmap[key][1]
except Exception as e:
    print(f"[est warn: {e}]",file=sys.stderr)

# 3) safe samples (LIMIT 3) guarded by statement_timeout
for grp,sch,tbl,do_sample in KEY:
    key=f"{sch}.{tbl}"
    if not do_sample:
        out[grp][key]["sample_error"]="skipped (large table)"
        continue
    try:
        s=db.raw_sql(f"SELECT * FROM {sch}.{tbl} LIMIT 3")
        recs=[]
        for r in s.to_dict(orient="records"):
            recs.append({k:jsonable(val) for k,val in r.items()})
        out[grp][key]["sample"]=recs
    except Exception as e:
        out[grp][key]["sample_error"]=str(e)[:200]
    print(f"  sampled {key}: {'OK' if out[grp][key]['sample'] else out[grp][key]['sample_error']}",file=sys.stderr)

with open("wrds_attributes.json","w") as f:
    json.dump(out,f,indent=2,ensure_ascii=False)
db.close()
print(f"[done {time.time()-t0:.1f}s -> wrds_attributes.json]",file=sys.stderr)
