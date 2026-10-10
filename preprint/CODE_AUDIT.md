# Code and evaluation audit (2026-10-10, manuscript v7)

Scope: every module in `src/deepgraph_gbm` (1,315 lines), all 32 files in `scripts/`,
the four test modules, `configs/`, `pyproject.toml` and `docs/`. Quantities were
re-derived independently rather than read from the project's own summary files:
`scripts/29_code_audit_checks.py` recomputes 99 published values from raw data or
saved predictions using scikit-learn directly, not `training/evaluate.py`.

**Outcome: 99/99 checks agree with the manuscript.** Eleven defects were found, none
of which changes a result; five were in the Methods text, six in the repository.
All eleven are fixed. Three known limitations remain open and are disclosed.

## What was verified

| Check | Result |
|---|---|
| Test suite | 18/18 pass (needs `PYTHONPATH=src` or the editable install) |
| Niche labels re-derived from `configs/labels.yaml` + `data/raw/snuh/meta.tsv` | reproduce the stored per-spot labels exactly (100.0000%) |
| MES binary target re-derived | reproduces exactly (100.0000%) |
| Spot counts | 100,488 total, 75,697 labeled |
| Graph geometry | 24.4% of 6-NN edges join non-adjacent spots, 24.2% of all edges being second-ring (manuscript: "about 24%") |
| Parameter counts | GraphSAGE 1,751,815; GAT 985,863 — exact |
| Fold integrity | 0 of 1,053 run records has its held-out patient in train or validation |
| Nested selection | per-fold configurations genuinely differ across the 13 folds |
| LOPO tuning support | every fold's validation block holds ≥183 spots of every class |
| Table 1, Table 2, Table 3, Tables S5–S8 | every value reproduces |
| Homophily, ablations, marker removal, chosen configurations | reproduce |
| Exact-hex sensitivity, UKF E1/E2, external cohort, survival | reproduce |

## Findings fixed in the manuscript

1. **The primary benchmark does not use the slide-level risk head.**
   `scripts/15_revision_benchmark.py` builds a two-head network and optimises
   cross-entropy + 0.5 × MES; the Methods' three-term objective (with the 0.5 × MSE
   risk term) applies only to the fixed-split pipeline in `src/`. A reader would have
   assumed the three-head model throughout. *Methods now states which objective applies
   where.* Severity: moderate — a Methods/code mismatch, not an error in any result.

2. **GAT carries a regulariser the other encoders do not.** `GATConv` is constructed
   with `dropout=p`, which applies dropout to the attention coefficients, on top of the
   activation dropout every model shares. The Methods claimed "normalization, dropout
   and heads were identical to GraphSAGE". At the sampled rates (0.1–0.5) this is a mild
   extra penalty on GAT alone, in a paper whose argument rests on matched comparison.
   *Disclosed in Methods and Limitations.* Removing it would require retraining.

3. **The "normal" rule is not strict concordance.** The rule requires the anatomical
   feature and the metaprogram each to be leading-edge, not to name the same tissue:
   3,158 of the 27,553 spots matching it pair a grey-matter with a white-matter
   annotation. The Methods said "with concordant metaprogram". *Reworded, with the count.*

4. **Two homophily definitions in one sentence.** The overall figure (0.80) is edge-level;
   the per-class figures are the mean over spots of the fraction of a spot's labeled
   neighbours sharing its label. *Both definitions now stated.*

5. **The niche head is a single linear layer**, not "linear layers". *Reworded.*

## Findings fixed in the repository

6. **`docs/WORKFLOW.md` still asserted the retracted protocol-reversal result** —
   "Scoring rules, not graph structure, created the apparent GNN advantage" — which was
   withdrawn in manuscript v5 and removed in v6. The same section carried the
   pre-correction external-validation numbers (mean r = 0.30; "compositions transfer")
   where the corrected analysis finds the opposite, and the seed-paired p = 0.0017 that
   v5 replaced with a Welch p = 0.012. Since the data-availability statement points
   readers at this repository, these were the most consequential defects found.
   *Section 6 rewritten against the verified values, with the retraction stated.*

7. **The figure/table map in `docs/WORKFLOW.md` was v4-era** (7 figures, "Figure 4 =
   protocol reversal"). *Rewritten for the current 10 main and 7 supplementary figures.*

8. **Superseded scripts were not labelled.** `14_fig_protocol_reversal.py` produces the
   withdrawn figure, `13_mlp_argmax_control.py` is subsumed by the decision-rule analysis
   in `25_review_sensitivity.py`, and `run_baselines.py` by `run_baselines_multiseed.py`.
   *All three now say so in their docstrings.*

9. **Stale supplementary numbering** in the docstrings of `09_make_table1.py` and
   `12_aggregate_multisplit.py` (Table S6/S7 under the old scheme). *Corrected.*

10. **`configs/labels.yaml` described the MES target as "fraction of MES malignant
    states"** — the wording the v5 audit corrected in the manuscript but not in the
    config that defines it. *Corrected to binary.*

11. **`pyproject.toml` dependency set was wrong in both directions**: it required
    `lab>=8.10` and `jupyter>=1.1.1`, neither imported anywhere in the codebase (`lab` is
    an unrelated PyPI package), and omitted `umap-learn`, which `scripts/07_make_figures.py`
    needs for Figure 1. `requirements.txt` and `pyproject.toml` disagreed. *Aligned.*

## Open, disclosed, not fixable without retraining

- **Six-split threshold protocol.** Four of six tuning blocks hold fewer than 20
  immune-hot spots. Disclosed in Section 2.4 and Supplementary Tables S10–S11; the
  analysis is used only for the variance decomposition. Repair needs 72 retrained models.
- **Transductive gene-panel selection.** The 3,000 HVGs were chosen once on the primary
  split's eight training patients and reused in every fold, so for 8 of 13 folds the panel
  saw the held-out patient's expression (never its labels). Identical for all models.
- **Selection-seed reuse.** Configurations are chosen on seed-42 validation scores and
  then evaluated on seeds 42, 7 and 123. Test metrics are unaffected.

## Coverage gap worth noting

The test suite covers `src/`, but the primary analysis does not run through `src/`: the
LOPO and UKF benchmarks use their own `build_model`, `tune_thresholds` and `score` inside
`scripts/15_revision_benchmark.py`, which no test exercises. The mitigation used here is
independent re-derivation — `scripts/29_code_audit_checks.py` recomputes the published
values from the saved predictions with its own metric implementations, and all 99 agree.
A unit test over the script's scoring path would make this cheaper to repeat.
