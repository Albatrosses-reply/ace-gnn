#!/usr/bin/env python3
"""Wave 3: severe-label sources (SOX404 ICFR, AAER) + audit-office (feed11 city) +
Form AP engagement partners (2017+). -> data/sox404.pkl, aaer.pkl, formap.pkl"""
import os
import sys, time
import wrds
from sqlalchemy import text
t0=time.time(); db=wrds.Connection(wrds_username=os.environ["WRDS_USERNAME"])
db.connection.execute(text("SET statement_timeout='300000'"))
def run(sql,label):
    s=time.time(); df=db.raw_sql(sql); print(f"[{label}] {len(df)} rows {time.time()-s:.1f}s",file=sys.stderr); return df

# SOX404 internal controls (ICFR): effectiveness + audit office (aud_city)
sox=run("""
  SELECT NULLIF(regexp_replace(company_fkey,'[^0-9]','','g'),'')::bigint AS cik,
         ic_is_effective, fy_ic_op::int AS fy, EXTRACT(YEAR FROM file_date)::int AS file_year,
         aud_city, aud_state, auditor_fkey, count_weak
  FROM audit.feed11_sox_404_internal_controls
  WHERE file_date BETWEEN '2009-01-01' AND '2020-12-31' AND company_fkey ~ '^[0-9]+$'
""","feed11 SOX404")
sox.to_pickle("data/sox404.pkl")

# AAER enforcement (company respondents)
aaer=run("""
  SELECT DISTINCT NULLIF(regexp_replace(r.cik::text,'[^0-9]','','g'),'')::bigint AS cik,
         EXTRACT(YEAR FROM a.first_release_date)::int AS ann_year
  FROM audit.f91_feed91_respondent r
  JOIN audit.feed91_aaer a ON r.aaer_release_fkey=a.first_release_fkey
  WHERE r.type='Entity' AND r.cik IS NOT NULL
        AND a.first_release_date BETWEEN '2010-01-01' AND '2020-12-31'
""","feed91 AAER companies")
aaer.to_pickle("data/aaer.pkl")

# Form AP engagement partners (issuer audits, 2017+), linked to company via feed34
fap=run("""
  SELECT f.engagement_partner_id AS partner_id,
         NULLIF(regexp_replace(o.company_fkey,'[^0-9]','','g'),'')::bigint AS cik,
         EXTRACT(YEAR FROM f.fiscal_period_end_date)::int AS fpe_year,
         EXTRACT(YEAR FROM f.filing_date)::int AS file_year
  FROM audit.f34_form_ap_filing f
  JOIN audit.feed34_revised_audit_opinions o ON f.audit_opinion_fkey=o.audit_op_key
  WHERE f.filing_date BETWEEN '2017-01-01' AND '2020-12-31'
        AND f.engagement_partner_id IS NOT NULL AND o.company_fkey ~ '^[0-9]+$'
""","Form AP partners")
fap.to_pickle("data/formap.pkl")

db.close(); print(f"[WAVE3 done {time.time()-t0:.1f}s]",file=sys.stderr)
