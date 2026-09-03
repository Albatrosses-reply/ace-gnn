#!/usr/bin/env python3
"""Wave 1 extraction: node universe + features, labels, auditor & board edges, id map.
All filtered/server-side reduced. Saves to data/*.pkl"""
import os, sys, time
import wrds
from sqlalchemy import text
os.makedirs("data", exist_ok=True)
t0=time.time(); db=wrds.Connection(wrds_username=os.environ["WRDS_USERNAME"])
db.connection.execute(text("SET statement_timeout='300000'"))
def run(sql,label):
    s=time.time(); df=db.raw_sql(sql); print(f"[{label}] {len(df)} rows  {time.time()-s:.1f}s",file=sys.stderr); return df

FEAT=["roa","roe","npm","opmad","gpm","cfm","ptpm","gprof","bm","ptb","pe_inc","divyield",
      "de_ratio","debt_at","debt_ebitda","lt_debt","intcov_ratio","capital_ratio",
      "curr_ratio","quick_ratio","cash_ratio","ocf_lct","at_turn","inv_turn","rect_turn",
      "accrual","mktcap","ret_crsp"]

# 1) Compustat company master (US) : gvkey, cik(bigint), sic, gsector, costat, dldte
comp=run(f"""
  SELECT gvkey, NULLIF(regexp_replace(cik,'[^0-9]','','g'),'')::bigint AS cik,
         sic, gsector, costat, fic, dldte
  FROM comp.company
  WHERE fic='USA' AND cik IS NOT NULL AND cik <> ''
""","comp.company")
comp.to_pickle("data/comp_company.pkl")

# 2) firm_ratio: one row per (gvkey, year) = latest public_date in year, 2010-2019
cols=",".join(FEAT)
fr=run(f"""
  SELECT DISTINCT ON (gvkey, EXTRACT(YEAR FROM public_date))
         gvkey, permno, cusip, public_date,
         EXTRACT(YEAR FROM public_date)::int AS year,
         ffi12, ffi49, gsector AS fr_gsector, {cols}
  FROM wrdsapps_finratio.firm_ratio
  WHERE public_date BETWEEN '2010-01-01' AND '2019-12-31'
  ORDER BY gvkey, EXTRACT(YEAR FROM public_date), public_date DESC
""","firm_ratio latest-in-year")
fr.to_pickle("data/firm_ratio.pkl")

# 3) auditor-client (feed03): cik, auditor_fkey, fiscal_year 2010-2019
aud=run("""
  SELECT NULLIF(regexp_replace(company_fkey,'[^0-9]','','g'),'')::bigint AS cik,
         auditor_fkey, fiscal_year::int AS fyear, auditor_name
  FROM audit.feed03_audit_fees
  WHERE fiscal_year BETWEEN 2010 AND 2019 AND company_fkey ~ '^[0-9]+$'
""","feed03 auditor")
aud.to_pickle("data/auditor.pkl")

# 4) restatement labels (feed39): cik, file_date, announce year, severity
res=run("""
  SELECT NULLIF(regexp_replace(company_fkey,'[^0-9]','','g'),'')::bigint AS cik,
         file_date, EXTRACT(YEAR FROM file_date)::int AS ann_year,
         res_fraud, res_adverse, res_accounting
  FROM audit.feed39_financial_restatements
  WHERE file_date BETWEEN '2011-01-01' AND '2020-12-31' AND company_fkey ~ '^[0-9]+$'
""","feed39 restatement labels")
res.to_pickle("data/restate.pkl")

# 5) board interlock edges, server-side joined to cik, window 2010-2019, !=Education
bd=run("""
  WITH prof AS (
    SELECT boardid, cikcode::bigint AS cik
    FROM boardex_na.na_wrds_company_profile
    WHERE cikcode IS NOT NULL AND cikcode::text ~ '^[0-9]+$'
  )
  SELECT p1.cik AS src_cik, p2.cik AS dst_cik,
         e.overlapyearstart_int AS y0, e.overlapyearend_int AS y1,
         count(*) AS n_shared_dir
  FROM boardex_na.na_wrds_company_networks e
  JOIN prof p1 ON e.boardid=p1.boardid
  JOIN prof p2 ON e.companyid=p2.boardid
  WHERE e.associationtype <> 'Education'
    AND e.overlapyearend_int >= 2010 AND e.overlapyearstart_int <= 2019
    AND p1.cik <> p2.cik
  GROUP BY p1.cik, p2.cik, e.overlapyearstart_int, e.overlapyearend_int
""","boardex interlock edges")
bd.to_pickle("data/board_edges.pkl")

db.close(); print(f"[WAVE1 done {time.time()-t0:.1f}s]",file=sys.stderr)
print("FEAT_COLS:", ",".join(FEAT))
