#!/usr/bin/env python3
"""Probe which cross-dataset LINKING tables are actually queryable + key id columns."""
import os
import json, sys, time
import wrds
from sqlalchemy import text

PROBE = [
  # CRSP-Compustat link candidates
  ("crsp","ccmxpf_linktable"), ("crsp","ccm_lookup"), ("crsp","ccmxpf_lnkused"),
  ("crsp","ccmxpf_lnkrng"),
  # dedicated wrds linking apps
  ("wrdsapps_link_crsp_ibes","ibcrsphist"),
  ("wrdsapps_link_crsp_bond","bondcrsp_link"),
  ("wrdsapps_link_crsp_taq","taqmclink"),
  ("wrdsapps_plink_exec_boardex","wrds_emp_boardex"),
  ("wrdsapps_link_crsp_factset","crsp_factset"),
  # subsidiary chars recheck
  ("wrdsapps_subsidiary","chars"),
]
t0=time.time(); db=wrds.Connection(wrds_username=os.environ["WRDS_USERNAME"])
try: db.connection.execute(text("SET statement_timeout='20000'"))
except Exception: pass
res={}
for sch,tbl in PROBE:
    rec={"queryable":False,"error":None,"columns":None,"sample":None,"nrows_probe":None}
    # columns
    try:
        c=db.raw_sql(f"""SELECT column_name,data_type FROM information_schema.columns
                         WHERE table_schema='{sch}' AND table_name='{tbl}' ORDER BY ordinal_position""")
        if len(c): rec["columns"]=[(r.column_name,r.data_type) for r in c.itertuples()]
    except Exception as e:
        rec["error_cols"]=str(e)[:150]
    # sample
    try:
        s=db.raw_sql(f"SELECT * FROM {sch}.{tbl} LIMIT 2")
        rec["queryable"]=True
        rec["sample"]=[{k:(str(v)[:40] if v is not None else None) for k,v in row.items()}
                       for row in s.to_dict(orient="records")]
        rec["nrows_probe"]=len(s)
    except Exception as e:
        rec["error"]=str(e)[:160]
    res[f"{sch}.{tbl}"]=rec
    print(f"{sch}.{tbl}: queryable={rec['queryable']} err={rec['error']}",file=sys.stderr)

# also: does wrds list these link libs?
for lib in ["wrdsapps_link_crsp_ibes","wrdsapps_link_crsp_bond","wrdsapps_link_crsp_taq",
            "wrdsapps_plink_exec_boardex","wrdsapps_link_crsp_factset","crsp_a_ccm"]:
    try:
        t=db.list_tables(library=lib); res.setdefault("_libtables",{})[lib]=t
    except Exception as e:
        res.setdefault("_libtables",{})[lib]=f"ERROR: {str(e)[:120]}"

# ---- AUDIT ANALYTICS (Accounting+Oversight, Audit+Compliance modules) ----
AUDIT_LIBS=["audit","audit_acct_os","audit_audit_comp","audit_common"]
audit={}
for lib in AUDIT_LIBS:
    try:
        tbls=db.list_tables(library=lib)
    except Exception as e:
        audit[lib]={"error":str(e)[:150]}; continue
    audit[lib]={"n_tables":len(tbls),"tables":sorted(tbls),"detail":{}}
    # pull columns for every audit table (fast, info_schema)
    if tbls:
        combos=",".join("('%s','%s')"%(lib,t) for t in tbls)
        try:
            c=db.raw_sql(f"""SELECT table_name,ordinal_position,column_name,data_type
                             FROM information_schema.columns
                             WHERE (table_schema,table_name) IN ({combos})
                             ORDER BY table_name,ordinal_position""")
            for r in c.itertuples():
                audit[lib]["detail"].setdefault(r.table_name,[]).append((r.column_name,r.data_type))
        except Exception as e:
            audit[lib]["col_error"]=str(e)[:150]
    print(f"AUDIT {lib}: {audit[lib].get('n_tables')} tables",file=sys.stderr)
# sample a few central audit tables (best-effort)
audit_samples={}
for sch,tbl in [("audit","auditnonreli"),("audit","auditopin"),("audit","auditsox404"),
                ("audit","feed03_director_legal"),("audit_common","company"),
                ("audit_audit_comp","auditcompfees"),("audit_acct_os","accntrestate")]:
    try:
        s=db.raw_sql(f"SELECT * FROM {sch}.{tbl} LIMIT 2")
        audit_samples[f"{sch}.{tbl}"]=[{k:(str(v)[:40] if v is not None else None)
                                        for k,v in row.items()} for row in s.to_dict(orient="records")]
    except Exception as e:
        audit_samples[f"{sch}.{tbl}"]={"error":str(e)[:140]}
res["_audit"]=audit
res["_audit_samples"]=audit_samples

with open("wrds_linking.json","w") as f: json.dump(res,f,indent=2,ensure_ascii=False)
db.close(); print(f"[done {time.time()-t0:.1f}s]",file=sys.stderr)
