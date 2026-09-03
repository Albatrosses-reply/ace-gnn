# Data directory (not distributed)

Nothing under `data/` is shipped with this repository, because every file here is either a WRDS-licensed
source table or is derived from one (firm universe, financial ratios, auditor / office / partner ties from
Audit Analytics, board interlocks from BoardEx, common-ownership ties from Thomson Reuters 13F, and the
labels built from Audit Analytics restatements, SOX 404 opinions and SEC AAERs). The WRDS subscription
terms do not permit redistribution of these tables or of data derived from them, and that includes the
multiplex graph (`data/ext/graph.pt`), the feature tables, and the saved model predictions.

With a WRDS subscription that covers Compustat, Audit Analytics, BoardEx and Thomson Reuters 13F, the whole
directory is rebuilt by the extraction and graph-construction scripts described in the top-level README:

```
data/
  ext/
    firm_ratio.pkl  auditor.pkl  formap.pkl  restate.pkl  sox404.pkl  aaer.pkl   <- src/1_extract/extract_ext_a.py
    holdings.pkl  board_edges.pkl                                                <- src/1_extract/extract_ext_a.py, extract_ext_b.py
    graph.pt                                                                     <- src/2_graph/build_graph_ext.py
    graph_own3.pt  graph_own5.pt                                                 <- src/2_graph/build_own_variant.py (threshold sensitivity)
    bench_*.json  gsat_*.json  placebo_*.json  ablation_*.json  case_studies_v2.json  <- src/3_models, src/4_analysis
    pure/
      tabular_fair_*.json(.npz)  leaderboard_*.json                              <- src/4_analysis/tabular_fair.py, assemble_leaderboard.py
      v2/
        FINAL_L_*.json(.npz)  <ablation tags>_*.json  boot_*.json  sage_*.json  ltq_synthetic*.json
```

`data/legacy/comp_company.pkl` (Compustat company header, used by the case-study scripts to attach SIC codes)
is produced by `src/1_extract/extract_wave1.py` into `data/` and can be copied or symlinked into `data/legacy/`.
