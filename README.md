# ACE-GNN — Guilt by Association: Interpretable Multiplex Graph Learning of Accounting-Risk Contagion

Code for the paper *Guilt by Association: Interpretable Multiplex Graph Learning of Accounting-Risk Contagion*
(Hojun Kang and Sang-Gun Lee, *Information Sciences*, 2026). ACE-GNN predicts next-year severe accounting
failures (ICFR material weaknesses, fraudulent restatements, SEC enforcement actions) as node classification
on a temporal multiplex firm graph whose layers are a nested auditor hierarchy (engagement partner, audit
office, audit firm), board interlocks, and common institutional ownership.

**What is here:** the complete non-proprietary stack — extraction of every input from WRDS, graph
construction, ACE-GNN and every baseline, the fair-comparison protocol, ablations, robustness checks,
interpretability analyses, and the scripts that regenerate every table and figure of the paper.

**What is not here, and why:** no data. Every input comes from WRDS-licensed tables (Compustat, Audit
Analytics, BoardEx, Thomson Reuters 13F), and the subscription terms do not permit redistribution of those
tables or of data derived from them; that includes the firm graph, the feature tables and the saved model
predictions. With a WRDS subscription covering those four products the pipeline below rebuilds everything
end to end and reproduces the paper's numbers.

## Requirements

Python 3.11 or later, CPU only (nothing in the paper used a GPU). Install PyTorch first, then PyTorch
Geometric for that torch build, then the rest:

```bash
pip install torch                       # https://pytorch.org
pip install torch_geometric             # https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
pip install -r requirements.txt
```

WRDS access is needed only for step 1. Put your WRDS user name in the environment and the password in
`~/.pgpass` as WRDS documents (no script takes a password argument):

```bash
export WRDS_USERNAME=your_wrds_login
```

All commands are run from the repository root. Every script reads its options from environment variables
(`MODE=recent|panel`, `LABEL=severe|label`, and per-script knobs listed in each file's docstring).

## Reproduction pipeline

### 1. Extract the inputs from WRDS (subscription required)

```bash
python3 src/1_extract/extract_ext_a.py    # firm universe + 28 financial ratios (wrdsapps_finratio / Compustat), auditor,
                                          # office and Form AP partner ties, restatements, SOX 404, AAERs, BoardEx interlocks
python3 src/1_extract/extract_ext_b.py    # Thomson Reuters 13F institutional holdings           -> data/ext/*.pkl
python3 src/1_extract/extract_wave1.py    # Compustat company header (SIC codes, used by the case studies) -> data/comp_company.pkl
mkdir -p data/legacy && cp data/comp_company.pkl data/legacy/
```

Window 2004–2024; identifiers are linked through CIK and CUSIP (the CRSP–Compustat link table is not used).
`src/0_data_catalog/` holds the scripts used to survey the WRDS libraries before the study; they are not
needed for reproduction.

### 2. Build the temporal multiplex graph

```bash
python3 src/2_graph/build_graph_ext.py                 # -> data/ext/graph.pt  (2005–2022, five relation layers, three labels)
OWN_MIN=3 python3 src/2_graph/build_own_variant.py     # -> data/ext/graph_own3.pt  (ownership-threshold sensitivity, Sec. 9.3)
OWN_MIN=5 python3 src/2_graph/build_own_variant.py     # -> data/ext/graph_own5.pt
```

Construction constants (auditor-group cap 40, partner cap 50, top-15 holders, at least 4 shared owners)
are set at the top of `build_graph_ext.py` and match Section 3.2 of the paper.

### 3. Train ACE-GNN (Table 6, "Ours")

The validation-selected configuration of Table 5 is `configs/ace_gnn_final.env`. Three of its switches
differ from the script defaults (`EXPENC=1`, `MP=1`, `COLLECTIVE=0`), so load the file rather than relying
on defaults:

```bash
set -a; source configs/ace_gnn_final.env; set +a
MODE=recent LABEL=severe NSEED=5 python3 src/3_models/ace_v2.py   # recent/severe (primary)
MODE=panel  LABEL=severe NSEED=4 python3 src/3_models/ace_v2.py   # panel/severe
MODE=recent LABEL=label  NSEED=5 python3 src/3_models/ace_v2.py   # recent/any
```

Each run writes `data/ext/pure/v2/FINAL_L_{mode}_{label}.json` (validation and test AUPRC, top-decile
recall and ROC, per seed and seed-bagged) and `..._preds.npz` (bagged predictions on the validation and
test sets). Selection is on validation AUPRC only; the test figure is stored for transparency.

### 4. Baselines under the fair-comparison protocol (Table 6)

```bash
for M in "recent severe" "panel severe" "recent label"; do set -- $M
  MODE=$1 LABEL=$2 python3 src/4_analysis/tabular_fair.py         # 8 tabular learners on exactly the GNN node features [Xz, OWN]
  MODE=$1 LABEL=$2 python3 src/4_analysis/benchmark.py            # GCN, GraphSAGE, GAT, APPNP, RGCN on the merged multiplex
  MODE=$1 LABEL=$2 python3 src/3_models/gsat_experiment.py        # GSAT (self-interpretable GNN)
  MODE=$1 LABEL=$2 python3 src/4_analysis/assemble_leaderboard.py # -> data/ext/pure/leaderboard_{setting}.json
done
```

`benchmark.py` additionally trains the feature-asymmetric variants of Section 5.1 (tabular learners with and
without the graph-exposure columns), which are reported there and excluded from the pure leaderboard.

### 5. Paired bootstrap against the strongest fair tabular learner (Table 7)

```bash
MODE=recent LABEL=severe BASELINE=RandomForest     TAG=RF python3 src/4_analysis/paired_bootstrap.py
MODE=panel  LABEL=severe BASELINE=HistGradBoosting TAG=HG python3 src/4_analysis/paired_bootstrap.py
```

### 6. Where the graph helps (Tables 8–10)

Table 8 is a set of tagged `ace_v2.py` runs on recent/severe; `make_tables.py` reads them by tag. Start from
the final configuration and override the switches shown (all with `MODE=recent LABEL=severe`; two seeds were
used for the design-space runs, five for the final model):

```bash
set -a; source configs/ace_gnn_final.env; set +a
# component build-up (selection path)
TAG=A_base          USE_EXP=0 EXPENC=0 MP=0 NSEED=2 python3 src/3_models/ace_v2.py   # encoder only (PLE + own history)
TAG=B_exp           USE_EXP=1 EXPENC=0 MP=0 NSEED=2 python3 src/3_models/ace_v2.py   # + raw neighbour exposures
TAG=I_expenc_nocol  USE_EXP=1 EXPENC=1 MP=0 NSEED=2 python3 src/3_models/ace_v2.py   # + encoded neighbour exposures
TAG=FINAL_L                                 NSEED=5 python3 src/3_models/ace_v2.py   # + relation-gated attention = ACE-GNN (step 3)
# exploratory branch: collective inference (APPNP on the predicted logit), rejected on validation
TAG=H_expenc        MP=0 COLLECTIVE=1        NSEED=2 python3 src/3_models/ace_v2.py   # encoder + encoded exposures + collective inference
TAG=J_mp_expenc     MP=1 COLLECTIVE=1        NSEED=2 python3 src/3_models/ace_v2.py   # ACE-GNN + collective inference
# additional levers, rejected on validation
TAG=M_tree    BINS=tree                     NSEED=2 python3 src/3_models/ace_v2.py
TAG=N_mtl     MTL=1                         NSEED=2 python3 src/3_models/ace_v2.py
TAG=O_rank    RANKW=0.5                     NSEED=2 python3 src/3_models/ace_v2.py
TAG=P_lag3    LAGS=3                        NSEED=2 python3 src/3_models/ace_v2.py
TAG=Q_cap     H=256 TABM=12 ENC_DEPTH=3     NSEED=2 python3 src/3_models/ace_v2.py
TAG=R_all     BINS=tree LAGS=3 MTL=1 RANKW=0.5 NSEED=2 python3 src/3_models/ace_v2.py
TAG=S_capmtl  H=256 TABM=12 ENC_DEPTH=3 MTL=1 NSEED=3 python3 src/3_models/ace_v2.py
# Table 9: the same encodings applied to a plain GraphSAGE
CELL=ple        python3 src/4_analysis/sage_ple.py
CELL=ple_expenc python3 src/4_analysis/sage_ple.py
# Table 10: learned-threshold-quantile aggregator against fixed aggregators on synthetic neighbourhood targets,
# and the same aggregator inside ACE-GNN (Sec. 7.2; reduced-capacity runs)
python3 src/3_models/ltq_synthetic2.py
TAG=FAST_attn MPKIND=attn TABM=6 H=160 EPMAX=200 NSEED=2 python3 src/3_models/ace_v2.py
TAG=FAST_ltq  MPKIND=ltq  TABM=6 H=160 EPMAX=200 NSEED=2 python3 src/3_models/ace_v2.py
```

### 7. Interpretability, robustness, sensitivity (Tables 11–12, Figure 5, Section 9)

```bash
set -a; source configs/ace_gnn_final.env; set +a
python3 src/4_analysis/case_study_v2.py                                   # per-firm gates, attention, edge-removal counterfactuals; silent failures
TAG=PLACEBO_L PLACEBO=1 NSEED=2 python3 src/3_models/ace_v2.py            # label-permutation placebo for ACE-GNN (train+val labels permuted)
MODE=recent LABEL=severe python3 src/4_analysis/placebo.py                # placebo for the exposure pipeline with a tree learner
TAG=SENS_OWN3 GRAPH=data/ext/graph_own3.pt NSEED=5 python3 src/3_models/ace_v2.py   # ownership-threshold sensitivity (Sec. 9.3)
TAG=SENS_OWN5 GRAPH=data/ext/graph_own5.pt NSEED=5 python3 src/3_models/ace_v2.py
python3 src/4_analysis/posthoc.py                                         # per-year breakdown, calibration, DeLong test
```

### 8. Tables and figures

```bash
python3 src/5_figures/make_tables.py        # every LaTeX table of the paper from the result files above (no hand-entered numbers)
python3 src/5_figures/make_pure_figs.py     # leaderboard figures
python3 src/5_figures/make_pr_curves.py     # Appendix A precision–recall curves
python3 src/5_figures/make_case_fig.py      # Figure 5
```

## Layout

```
src/0_data_catalog/   WRDS library survey (not needed for reproduction)
src/1_extract/        WRDS extraction (subscription required)
src/2_graph/          temporal multiplex graph construction and label definitions
src/3_models/         ACE-GNN (ace_v2.py), GSAT, earlier graph-only variants, synthetic aggregator test
src/4_analysis/       fair tabular baselines, standard-GNN benchmark, leaderboard assembly, paired bootstrap,
                      ablations, placebo, case studies
src/5_figures/        table and figure generation
configs/              the validation-selected ACE-GNN configuration (Table 5)
data/README.md        what the (non-distributed) data directory contains after steps 1–2
```

## Authors and citation

Hojun Kang (Department of Business Administration, Dongduk Women's University) and Sang-Gun Lee (Business
School, Sogang University). Citation metadata is in `CITATION.cff`.

## Licence

MIT for the code in this repository. The WRDS data remain subject to the WRDS subscription terms.
