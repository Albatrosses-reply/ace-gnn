#!/usr/bin/env python3
"""Wave 2: 13F common-ownership holdings, filtered to firm_ratio cusip universe.
psycopg2 ANY(%s) array binding. Q4 rdate each year 2010-2019. -> data/holdings.pkl"""
import os
import sys, time
import pandas as pd
import psycopg2

fr = pd.read_pickle("data/firm_ratio.pkl")
cusips = sorted({c for c in fr["cusip"].dropna().unique() if isinstance(c, str) and len(c) == 8})
print(f"[universe cusips: {len(cusips)}]", file=sys.stderr)

t0 = time.time()
conn = psycopg2.connect(host="wrds-pgdata.wharton.upenn.edu", port=9737,
                        dbname="wrds", user=os.environ["WRDS_USERNAME"])  # password from ~/.pgpass
cur = conn.cursor()
cur.execute("SET statement_timeout='300000'")
frames = []
for y in range(2010, 2020):
    d = f"{y}-12-31"; s = time.time()
    cur.execute("""SELECT mgrno, cusip, rdate, shares, shrout2
                   FROM tr_13f.s34
                   WHERE rdate = %s AND cusip = ANY(%s) AND shares > 0""", (d, cusips))
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["mgrno", "cusip", "rdate", "shares", "shrout2"])
    df["year"] = y
    frames.append(df)
    print(f"  {d}: {len(df)} holdings  {time.time()-s:.1f}s", file=sys.stderr)
hold = pd.concat(frames, ignore_index=True)
hold.to_pickle("data/holdings.pkl")
cur.close(); conn.close()
print(f"[WAVE2 done {len(hold)} rows {time.time()-t0:.1f}s]", file=sys.stderr)
