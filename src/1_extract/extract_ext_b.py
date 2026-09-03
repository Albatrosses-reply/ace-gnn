#!/usr/bin/env python3
"""Extended 13F holdings (Q4 2004-2022) filtered to ext firm_ratio cusips. -> data/ext/holdings.pkl"""
import os
import sys, time
import pandas as pd, psycopg2
fr=pd.read_pickle("data/ext/firm_ratio.pkl")
cusips=sorted({c for c in fr["cusip"].dropna().unique() if isinstance(c,str) and len(c)==8})
print(f"[ext cusips: {len(cusips)}]",file=sys.stderr)
t0=time.time(); conn=psycopg2.connect(host="wrds-pgdata.wharton.upenn.edu",port=9737,dbname="wrds",user=os.environ["WRDS_USERNAME"])
cur=conn.cursor(); cur.execute("SET statement_timeout='400000'")
frames=[]
for y in range(2004,2023):
    d=f"{y}-12-31"; s=time.time()
    cur.execute("""SELECT mgrno,cusip,rdate,shares,shrout2 FROM tr_13f.s34
                   WHERE rdate=%s AND cusip=ANY(%s) AND shares>0""",(d,cusips))
    rows=cur.fetchall(); df=pd.DataFrame(rows,columns=["mgrno","cusip","rdate","shares","shrout2"]); df["year"]=y
    frames.append(df); print(f"  {d}: {len(df)} {time.time()-s:.1f}s",file=sys.stderr)
pd.concat(frames,ignore_index=True).to_pickle("data/ext/holdings.pkl")
cur.close(); conn.close(); print(f"[EXT-B done {time.time()-t0:.1f}s]",file=sys.stderr)
