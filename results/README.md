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
| `tables/main/` | Tables 1–3, Supplementary Table S7 (`Table2_test_metrics.json`), the slide-risk head evaluation, and the two verification logs |
| `tables/supplement/` | Supplementary Tables S1–S6 and S9–S13, including the released UKF pathologist annotations (S9) |
| `revision/runs/` | One JSON per leave-one-patient-out run (1,053: 520 tuning, 195 final, 234 ablation, 104 marker-removal) with its configuration, fold, seed, validation and test metrics |
| `revision/` | `folds.json`, `search_configs.json`, `chosen_configs.json`, `revision_summary.json` (Table 2), `label_homophily.json`, `logistic_convergence_check.json`, and the job lists |
| `revision_hex/` | Exact-hexagonal-graph LOPO runs and `hex_sensitivity.json` |
| `revision_ukf/` | Neuropathologist cohort, E1 (within-cohort LOPO), and `histology_summary.json` |
| `revision_transfer_v2/` | Neuropathologist cohort, E2 (SNUH → UKF transfer) at matched CP100k scaling — **the published transfer result** |
| `revision_transfer/` | The same transfer at CP10k scaling — reported only as a scaling sensitivity analysis |
| `multisplit/` | The 72 six-split runs (6 splits × 3 seeds × 4 models) and `multisplit_summary.json` (Table S8); split 0 also supplies Table 1's GAT row |
| `models/` | Published fixed-split GraphSAGE weights (`best_model.pt` = seed 7) and per-seed metrics behind Table 1 |
| `baselines_multiseed_matched/` | Fixed-split MLP and logistic runs behind Table 1 and Table S5, plus the decision-rule controls cited in section 2.3 |
| `baselines_original_protocol_rerun/` | Single-run baselines scored by argmax; source of the logistic argmax value (0.541) in section 2.3 |
| `rerun_20261009/` | A later retrain of the fixed-split models with current software: the refit logistic value (0.557) and the slide-level head evaluation |
| `external_v2/`, `external_v2_ts1e4/` | Greenwald cohort predictions at matched (CP100k) and mis-scaled (CP10k) normalisation |
| `archive/` | Superseded records kept for provenance only; nothing current reads them (see below) |

### `archive/`

| Path | Why it is archived |
|---|---|
| `archive/baselines_multiseed_unmatched/` | Early multi-seed baselines whose MLP was not trained under the matched protocol; replaced by `baselines_multiseed_matched/` |
| `archive/tables/baseline_original_protocol_metrics*.json` | Baselines of the retracted protocol-reversal analysis (one from a superseded feature set; see `preprint/ANALYSIS_AUDIT.md`, finding 1) |
| `archive/tables/TableS{1,3}_*_v1_cp10k.*` | Greenwald tables from before the normalisation fix (audit finding 2) |

## Deliberately not tracked

| Excluded | Why | How to obtain |
|---|---|---|
| `preds/*.npz` (~90 MB) | Per-spot probability arrays for every run. Needed only to re-pool the LOPO analyses from scratch; every scalar they produce is already in `runs/*.json` and the tables. | From the author on request |
| `figures/` (~30 MB) | Intermediate renders. The published versions are in `preprint/figures/`. | Steps B6, C, D3 and E4 of `docs/WORKFLOW.md` |
| `**/logs/` | Worker scratch logs from the job queue. | Regenerated on any re-run |
| `../data/` (~11 GB) | Raw and preprocessed datasets, all publicly available. | `scripts/01_download_data.sh` and the sources in the README |

## One convention that will bite you

In `multisplit/*.json` the architecture is **not** in the `model` field: GraphSAGE and GAT runs
are both stored as `"model": "gnn"` and are distinguished only by the `"encoder"` field.
Grouping by `model` silently averages the two architectures. See the `multisplit_scores` function
in `scripts/25_review_sensitivity.py` for the correct split, and `docs/WORKFLOW.md` for the other
convention (`y = -1` marks unlabeled spots in the prediction arrays and must be masked out).
