#!/usr/bin/env python3
"""Extended extraction A (2004-2024): firm_ratio, auditor, restatement, board, SOX404, AAER, Form AP.
Wider window for a longer panel (2005-2019) + recent partner series (2017-2022). -> data/ext/*.pkl"""
import os, sys, time
import wrds
from sqlalchemy import text
os.makedirs("data/ext", exist_ok=True)
t0=time.time(); db=wrds.Connection(wrds_username=os.environ["WRDS_USERNAME"])
db.connection.execute(text("SET statement_timeout='400000'"))
def run(sql,label):
    s=time.time(); df=db.raw_sql(sql); print(f"[{label}] {len(df)} rows {time.time()-s:.1f}s",file=sys.stderr); return df
FEAT=["roa","roe","npm","opmad","gpm","cfm","ptpm","gprof","bm","ptb","pe_inc","divyield",
      "de_ratio","debt_at","debt_ebitda","lt_debt","intcov_ratio","capital_ratio",
      "curr_ratio","quick_ratio","cash_ratio","ocf_lct","at_turn","inv_turn","rect_turn",
      "accrual","mktcap","ret_crsp"]
cols=",".join(FEAT)
run(f"""SELECT DISTINCT ON (gvkey, EXTRACT(YEAR FROM public_date)) gvkey, permno, cusip, public_date,
        EXTRACT(YEAR FROM public_date)::int AS year, ffi12, ffi49, {cols}
        FROM wrdsapps_finratio.firm_ratio
        WHERE public_date BETWEEN '2004-01-01' AND '2023-12-31'
        ORDER BY gvkey, EXTRACT(YEAR FROM public_date), public_date DESC""","firm_ratio").to_pickle("data/ext/firm_ratio.pkl")
run("""SELECT NULLIF(regexp_replace(company_fkey,'[^0-9]','','g'),'')::bigint AS cik,
       auditor_fkey, fiscal_year::int AS fyear FROM audit.feed03_audit_fees
       WHERE fiscal_year BETWEEN 2004 AND 2022 AND company_fkey ~ '^[0-9]+$'""","feed03").to_pickle("data/ext/auditor.pkl")
run("""SELECT NULLIF(regexp_replace(company_fkey,'[^0-9]','','g'),'')::bigint AS cik,
       EXTRACT(YEAR FROM file_date)::int AS ann_year, res_fraud, res_adverse
       FROM audit.feed39_financial_restatements
       WHERE file_date BETWEEN '2005-01-01' AND '2024-12-31' AND company_fkey ~ '^[0-9]+$'""","feed39").to_pickle("data/ext/restate.pkl")
run("""WITH prof AS (SELECT boardid, cikcode::bigint AS cik FROM boardex_na.na_wrds_company_profile
                     WHERE cikcode IS NOT NULL AND cikcode::text ~ '^[0-9]+$')
       SELECT p1.cik AS src_cik, p2.cik AS dst_cik, e.overlapyearstart_int AS y0, e.overlapyearend_int AS y1,
              count(*) AS n_shared_dir
       FROM boardex_na.na_wrds_company_networks e
       JOIN prof p1 ON e.boardid=p1.boardid JOIN prof p2 ON e.companyid=p2.boardid
       WHERE e.associationtype<>'Education' AND e.overlapyearend_int>=2004 AND e.overlapyearstart_int<=2023
             AND p1.cik<>p2.cik
       GROUP BY p1.cik,p2.cik,e.overlapyearstart_int,e.overlapyearend_int""","board").to_pickle("data/ext/board_edges.pkl")
run("""SELECT NULLIF(regexp_replace(company_fkey,'[^0-9]','','g'),'')::bigint AS cik,
       ic_is_effective, fy_ic_op::int AS fy, EXTRACT(YEAR FROM file_date)::int AS file_year,
       aud_city, aud_state, auditor_fkey FROM audit.feed11_sox_404_internal_controls
       WHERE file_date BETWEEN '2004-01-01' AND '2024-12-31' AND company_fkey ~ '^[0-9]+$'""","feed11").to_pickle("data/ext/sox404.pkl")
run("""SELECT DISTINCT NULLIF(regexp_replace(r.cik::text,'[^0-9]','','g'),'')::bigint AS cik,
       EXTRACT(YEAR FROM a.first_release_date)::int AS ann_year
       FROM audit.f91_feed91_respondent r JOIN audit.feed91_aaer a ON r.aaer_release_fkey=a.first_release_fkey
       WHERE r.type='Entity' AND r.cik IS NOT NULL
             AND a.first_release_date BETWEEN '2005-01-01' AND '2024-12-31'""","aaer").to_pickle("data/ext/aaer.pkl")
run("""SELECT f.engagement_partner_id AS partner_id,
       NULLIF(regexp_replace(o.company_fkey,'[^0-9]','','g'),'')::bigint AS cik,
       EXTRACT(YEAR FROM f.fiscal_period_end_date)::int AS fpe_year
       FROM audit.f34_form_ap_filing f JOIN audit.feed34_revised_audit_opinions o ON f.audit_opinion_fkey=o.audit_op_key
       WHERE f.filing_date BETWEEN '2017-01-01' AND '2024-12-31'
             AND f.engagement_partner_id IS NOT NULL AND o.company_fkey ~ '^[0-9]+$'""","formap").to_pickle("data/ext/formap.pkl")
db.close(); print(f"[EXT-A done {time.time()-t0:.1f}s]",file=sys.stderr)
