#!/usr/bin/env python3
"""Trim audit metadata to GNN/network-relevant tables -> compact group_audit.json"""
import json
link=json.load(open("wrds_linking.json"))
A=link["_audit"]; S=link.get("_audit_samples",{})

KEEP=set("""
feed01_auditors lookup_auditors wrds_lookup_auditors feed02_auditor_changes
feed03_audit_fees feed04_audit_fees_restated feed07_current_auditor feed08_auditor_during
f06_form_ap_filing f06_form_ap_divided f06_form_ap_not_divided f34_form_ap_filing
feed39_financial_restatements f39_restatement_category f39_restatement_periods
f39_restatement_auditor_period f39_restatement_to_category feed91_aaer
f91_feed91_respondent f91_feed91_related_party f91_feed91_respondent_order
f91_feed91_release_to_issue feed11_sox_404_internal_controls feed10_sox_302_disclosure_contro
feed34_revised_audit_opinions f34_going_concern_issues
f74_feed_support74_altman_score f74_feed_support74_beneish_score
f74_feed_support74_benfords_law f74_feed_support74_going_concern f74_feed_support74_audit_fees
f14_lit_legal_case f14_lit_legal_party f14_lit_legal_parties_to_firms f14_lit_law_firm
feed13_legal_case_feed feed14_company_legal_party_feed feed67_law_firms
f67_fee_sup_lit_law_fir_net f67_fee_sup_lit_law_fir_to_net
f25_person f25_person_employer f25_person_positions f25_person_filing
feed85_cybersecurity f85_cybersec_breach_attack f85_cybersec_breach_information
f85_cyb_bre_to_inf f85_cybersec_target_relationship f85_cybersec_filing
feed18_merger_acquisition feed17_director_and_officer_chan f17_dno_change
feed25_comment_letters feed78_critical_audit_matters f78_feed_support78_cam_topic
feed89_pcaob_report feed12_company_block wrds_lookup_edgar_company_block
""".split())

out={"module_map":{},"tables":{},"samples":S,"all_table_counts":{}}
for lib in ["audit","audit_acct_os","audit_audit_comp","audit_common"]:
    info=A.get(lib,{})
    out["all_table_counts"][lib]=info.get("n_tables")
    out["module_map"][lib]=sorted([t for t in info.get("tables",[]) if t in KEEP])
    det=info.get("detail",{})
    for t in det:
        if t in KEEP:
            out["tables"][f"{lib}.{t}"]={"columns":det[t]}
json.dump(out, open("groups/group_audit.json","w"), indent=2, ensure_ascii=False)
import os; print("group_audit.json", os.path.getsize("groups/group_audit.json"),
                  "| kept tables:", len(out["tables"]))
