#!/usr/bin/env python3
"""Recon: verify join keys + year coverage + positive rate. Filtered aggregates only."""
import os
import json, sys, time
import wrds
from sqlalchemy import text
t0=time.time(); db=wrds.Connection(wrds_username=os.environ["WRDS_USERNAME"])
try: db.connection.execute(text("SET statement_timeout='60000'"))
except Exception as e: print("warn",e,file=sys.stderr)
def q(sql,label):
    try:
        df=db.raw_sql(sql); print(f"\n### {label}\n{df.to_string(max_rows=30)}",file=sys.stderr); return df
    except Exception as e:
        print(f"\n### {label} ERROR: {str(e)[:200]}",file=sys.stderr); return None

# 1) feed39 restatements: id/date columns + year coverage + fraud share
q("SELECT company_fkey, file_date, res_fraud, res_accounting, res_adverse FROM audit.feed39_financial_restatements WHERE file_date>='2014-01-01' LIMIT 5","feed39 sample")
q("""SELECT EXTRACT(YEAR FROM file_date)::int AS yr, count(*) n, count(distinct company_fkey) firms,
       sum(CASE WHEN res_fraud='1' OR res_fraud=true OR res_fraud='Y' THEN 1 ELSE 0 END) fraud
     FROM audit.feed39_financial_restatements
     WHERE file_date BETWEEN '2009-01-01' AND '2020-12-31'
     GROUP BY 1 ORDER BY 1""","feed39 by year")

# 2) feed03 audit fees: auditor-client by fiscal_year
q("SELECT company_fkey, auditor_fkey, fiscal_year, auditor_name FROM audit.feed03_audit_fees WHERE fiscal_year=2015 LIMIT 5","feed03 sample")
q("""SELECT fiscal_year, count(*) n, count(distinct company_fkey) firms, count(distinct auditor_fkey) auditors
     FROM audit.feed03_audit_fees WHERE fiscal_year BETWEEN 2010 AND 2019 GROUP BY 1 ORDER BY 1""","feed03 by year")

# 3) comp.company cik format + coverage
q("SELECT gvkey, cik, sic, fic, costat FROM comp.company WHERE cik IS NOT NULL AND fic='USA' LIMIT 5","comp.company sample")
q("SELECT count(*) gvkeys, count(cik) with_cik, count(DISTINCT cik) distinct_cik FROM comp.company WHERE fic='USA'","comp.company cik coverage US")

# 4) JOIN RATE: audit company_fkey == comp.company.cik (cast int)
q("""WITH a AS (SELECT DISTINCT company_fkey::bigint AS cik FROM audit.feed03_audit_fees WHERE fiscal_year=2015 AND company_fkey ~ '^[0-9]+$'),
          c AS (SELECT DISTINCT NULLIF(regexp_replace(cik,'[^0-9]','','g'),'')::bigint AS cik FROM comp.company WHERE cik IS NOT NULL)
     SELECT (SELECT count(*) FROM a) audit_firms_2015,
            (SELECT count(*) FROM a JOIN c USING(cik)) matched_to_comp""","JOINRATE audit.company_fkey<->comp.cik (2015)")

# 5) BoardEx cikcode -> comp cik join
q("SELECT companyid, boardid, cikcode, orgtype, ticker FROM boardex_na.na_wrds_company_profile WHERE cikcode IS NOT NULL LIMIT 5","boardex profile sample")
q("""WITH b AS (SELECT DISTINCT cikcode::bigint AS cik FROM boardex_na.na_wrds_company_profile WHERE cikcode IS NOT NULL AND cikcode::text ~ '^[0-9]+$'),
          c AS (SELECT DISTINCT NULLIF(regexp_replace(cik,'[^0-9]','','g'),'')::bigint AS cik FROM comp.company WHERE cik IS NOT NULL)
     SELECT (SELECT count(*) FROM b) boardex_firms_with_cik,
            (SELECT count(*) FROM b JOIN c USING(cik)) matched_to_comp""","JOINRATE boardex.cikcode<->comp.cik")

# 6) firm_ratio coverage by year + cusip format
q("SELECT gvkey, permno, public_date, cusip, ticker, roa, de_ratio, mktcap, ffi49 FROM wrdsapps_finratio.firm_ratio WHERE public_date='2015-12-31' LIMIT 3","firm_ratio sample")
q("""SELECT EXTRACT(YEAR FROM public_date)::int yr, count(distinct gvkey) firms
     FROM wrdsapps_finratio.firm_ratio WHERE public_date BETWEEN '2010-01-01' AND '2019-12-31' AND EXTRACT(MONTH FROM public_date)=12
     GROUP BY 1 ORDER BY 1""","firm_ratio Dec firms by year")

# 7) 13F s34 cusip format + coverage one quarter
q("SELECT mgrno, cusip, rdate, fdate, shares, shrout2 FROM tr_13f.s34 WHERE rdate='2015-12-31' LIMIT 3","s34 sample")
q("""SELECT count(*) rows, count(distinct cusip) stocks, count(distinct mgrno) mgrs
     FROM tr_13f.s34 WHERE rdate='2015-12-31'""","s34 2015Q4 size")

db.close(); print(f"\n[done {time.time()-t0:.1f}s]",file=sys.stderr)
