#!/usr/bin/env python3
"""Split extracted metadata into per-group files for the workflow agents."""
import json, os
os.makedirs("groups", exist_ok=True)
attrs=json.load(open("wrds_attributes.json"))
link=json.load(open("wrds_linking.json"))

# groups from wrds_attributes.json
for g,payload in attrs.items():
    json.dump(payload, open(f"groups/group_{g}.json","w"), indent=2, ensure_ascii=False)

# audit slice (from linking probe)
audit={"_audit":link.get("_audit",{}),"_audit_samples":link.get("_audit_samples",{})}
json.dump(audit, open("groups/group_audit.json","w"), indent=2, ensure_ascii=False)

# linking slice (everything except audit + libtables)
linkonly={k:v for k,v in link.items() if k not in ("_audit","_audit_samples")}
json.dump(linkonly, open("groups/group_linking.json","w"), indent=2, ensure_ascii=False)

for f in sorted(os.listdir("groups")):
    print(f, os.path.getsize(os.path.join("groups",f)))
