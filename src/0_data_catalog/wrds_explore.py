#!/usr/bin/env python3
"""Step 1: connect to WRDS and enumerate all accessible libraries."""
import os
import json, sys, time
import wrds

t0 = time.time()
db = wrds.Connection(wrds_username=os.environ["WRDS_USERNAME"])
print(f"[connected in {time.time()-t0:.1f}s]", file=sys.stderr)

libs = db.list_libraries()
print(f"[{len(libs)} libraries accessible]", file=sys.stderr)

with open("wrds_libraries.json", "w") as f:
    json.dump(sorted(libs), f, indent=2)

# print to stdout too
for L in sorted(libs):
    print(L)

db.close()
