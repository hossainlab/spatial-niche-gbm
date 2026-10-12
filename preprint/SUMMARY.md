# DeepGraph-GBM manuscript: summary (v7.1; repository cleaned in v7.2, 2026-10-13)

## Deliverables (`preprint/`)

| File | Content |
|---|---|
| `deepgraph-gbm.tex` / `.pdf` | Current manuscript (v7.1; 40 pages, 53 references, 10 main and 7 supplementary figures plus the graphical abstract, 3 main and 13 supplementary tables) |
| `archive/` | Version snapshots v1–v7 with their bibliographies, and the pre-pivot manuscript (`manuscript_v1_prepivot.md/.pdf`) and graphical abstract |
| `SUBMISSION.md` | bioRxiv-first submission checklist |
| `PEER_REVIEW.md` | Independent review of v6, the v7 response, and the v7.1 code audit |
| `CODE_AUDIT.md` | Code, modeling and evaluation audit: 99/99 independent checks, 11 defects fixed |
| `ANALYSIS_AUDIT.md` | Earlier scientific audit: errors found, reruns, disclosures |
| `revision_notes.md` | What changed in each version |
| `references.bib` | Bibliography (53 entries; DOIs checked against Crossref, non-DOI entries against arXiv) |
| `figures/` | All figures, including `FigS7_robustness.png` (new in v7) |

## What changed in v7

The v6 manuscript was re-audited against the saved run records rather than its
own summaries. Every published number reproduced; three analysis choices that
were not reported turned out to move results as much as the architecture did,
and are now reported. See `PEER_REVIEW.md` for the full response. In short:

- the six-split benchmark is demoted to the variance decomposition, with its
  threshold-tuning support disclosed and a three-class sensitivity analysis;
- the primary model contrast is the pooled macro-F1 difference with a patient
  bootstrap; per-patient tests are secondary, Holm-adjusted, and reported with a
  class-support floor;
- the argmax-versus-tuned-threshold comparison is reported and shows the
  protocol is conservative for the paper's conclusion;
- survival results are per SD, with an age positive control and a detectable-effect bound;
- the slide-level risk head is evaluated, fails, and is reported as a negative result;
- the neuropathologist annotations are released as Supplementary Table S9 rather
  than cited by deposit alone.

## Compute status

All benchmark runs are complete, with no failures. v7 added no model training;
every new analysis is computed from the records below.

| Queue | Runs | Done |
|---|---|---|
| SNUH tuning | 520 | 520 |
| SNUH final | 195 | 195 |
| SNUH ablation and marker removal | 338 | 338 |
| UKF pathologist LOPO | 288 | 288 |
| SNUH → UKF transfer (v2, matched scaling) | 16 | 16 |
| Exact-hex GraphSAGE/GAT LOPO sensitivity | 78 | 78 |

## Reproduce

The complete, script-by-script guide is `docs/WORKFLOW.md`. The commands below re-run the
analyses that read only `results/` and `data/preprocessed*`.

```bash
cd deepgraph-gbm
.venv/bin/python -m pytest -q tests                        # 18 tests
.venv/bin/python scripts/17_revision_analysis.py           # SNUH LOPO, ablations, markers (~15 min)
.venv/bin/python scripts/17_revision_analysis.py --figures-only
.venv/bin/python scripts/21_histology_analysis.py          # UKF + transfer (v2)
.venv/bin/python scripts/23_hex_sensitivity.py             # hex-graph sensitivity
.venv/bin/python scripts/05_external_validation.py --greenwald-dir data/raw/greenwald \
    --ckpt results/models/best_model.pt --hvg data/preprocessed/hvg_genes.json --out results/external_v2
.venv/bin/python scripts/22_graphical_abstract.py          # graphical abstract

# v7 review-response analyses (no retraining; all read results/)
.venv/bin/python scripts/24_export_ukf_annotations.py data/raw/ukf_spata \
    results/tables/supplement/TableS9_ukf_pathologist_annotations.csv
.venv/bin/python scripts/25_review_sensitivity.py          # Tables S10-S13 + slide-risk eval
.venv/bin/python scripts/26_survival_power.py              # Table 3
.venv/bin/python scripts/27_fig_robustness.py              # Supplementary Figure S7
.venv/bin/python scripts/28_verify_manuscript_numbers.py   # checks every v7 number against its source
PYTHONPATH=src .venv/bin/python scripts/29_code_audit_checks.py  # 99 independent re-derivations

cd preprint && tectonic deepgraph-gbm.tex                  # compile (pdflatex is not installed)
```

To rerun the benchmark from scratch, use the worker queue (finished runs are skipped):
`UKF=1 nohup caffeinate -i bash scripts/run_revision_worker.sh 0 >> results/revision/logs/w0.log 2>&1 &`

## Before submission

1. Push the repository (see `SUBMISSION.md` for the remaining author-side items).
2. Decide whether to trim the abstract from 353 words if the target venue caps it at 350.
3. Optional, and the one substantive item left open: repair the six-split
   threshold protocol by retraining its 72 models with thresholds tuned on pooled
   out-of-fold predictions. v7 discloses the defect and demotes the analysis
   instead; the leave-one-patient-out benchmark already carries the conclusion.
