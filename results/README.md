# `results/` — what is here and what is not

Everything needed to re-derive the published numbers **without retraining any model** is tracked
in this repository. The two verification scripts read only from here:

```bash
PYTHONPATH=src python scripts/28_verify_manuscript_numbers.py   # 35/35 manuscript numbers
PYTHONPATH=src python scripts/29_code_audit_checks.py           # 99/99 independent checks
```

## Layout

| Path | Contents |
|---|---|
| `tables/main/` | Tables 1–3 of the manuscript, the slide-risk head evaluation, and the two verification logs |
| `tables/supplement/` | Supplementary Tables S1–S13, including the released UKF pathologist annotations (S9) |
| `tables/archive/` | Superseded original-protocol baselines, kept for provenance only |
| `revision/runs/` | One JSON per leave-one-patient-out run (1,053: 520 tuning, 195 final, 234 ablation, 104 marker-removal) with its configuration, fold, seed, validation and test metrics |
| `revision/` | `folds.json`, `chosen_configs.json`, `revision_summary.json`, `label_homophily.json`, `logistic_convergence_check.json` |
| `revision_ukf/`, `revision_transfer_v2/` | Per-run records and summaries for the neuropathologist-cohort experiments (E1 within-cohort, E2 transfer) |
| `revision_hex/` | Exact-hexagonal-graph sensitivity runs and `hex_sensitivity.json` |
| `multisplit/` | The 72 six-split runs (6 splits × 3 seeds × 4 models) |
| `baselines_multiseed_matched/`, `rerun_20261009/` | Fixed-split matched-protocol runs behind Table 1 and Supplementary Table S5 |
| `external_v2/`, `external_v2_ts1e4/` | Greenwald cohort predictions under matched (CP100k) and mis-scaled (CP10k) normalisation |
| `models/`, `rerun_20261009/models/` | Trained weights for the fixed-split models |

## Deliberately not tracked

| Excluded | Why | How to obtain |
|---|---|---|
| `preds/*.npz` (~90 MB) | Per-spot probability arrays for every run. Needed only to re-pool the LOPO analyses from scratch; every scalar they produce is already in `runs/*.json` and the tables. | From the author on request |
| `figures/` (~30 MB) | Intermediate renders. The published versions are in `preprint/figures/`. | `scripts/17 --figures-only`, `scripts/21`, `scripts/27` |
| `**/logs/` | Worker scratch logs from the job queue. | Regenerated on any re-run |
| `../data/` (~11 GB) | Raw and preprocessed datasets, all publicly available. | `scripts/01_download_data.sh` and the sources in the README |

## One convention that will bite you

In `multisplit/*.json` the architecture is **not** in the `model` field: GraphSAGE and GAT runs
are both stored as `"model": "gnn"` and are distinguished only by the `"encoder"` field.
Grouping by `model` silently averages the two architectures. See the `multisplit_scores` function
in `scripts/25_review_sensitivity.py` for the correct split, and `docs/WORKFLOW.md` for the other
convention (`y = -1` marks unlabeled spots in the prediction arrays and must be masked out).
