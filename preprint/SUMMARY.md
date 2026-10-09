# DeepGraph-GBM manuscript: summary (v6, 2026-10-09)

## Deliverables (`preprint/`)

| File | Content |
|---|---|
| `deepgraph-gbm.tex` / `.pdf` | Current manuscript (v6; 34 pages, 49 references, 16 figures, 9 tables) |
| `deepgraph-gbm.v1`–`v6.tex`, `references.v*.bib` | Version snapshots |
| `ANALYSIS_AUDIT.md` | Scientific audit: errors found, reruns, disclosures |
| `references.bib` | Bibliography (DOIs checked against Crossref, non-DOI entries checked against arXiv) |
| `figures/` | All figures, including the new `graphical_abstract.png` and `FigR2_ablation.png` / `FigR3_marker_removal.png` |
| `revision_notes.md` | What changed in each version |
| `PEER_REVIEW.md` | Internal review, plus what was fixed and what is still open |

## Compute status

All benchmark runs are complete, with no failures:

| Queue | Runs | Done |
|---|---|---|
| SNUH tuning | 520 | 520 |
| SNUH final | 195 | 195 |
| SNUH ablation and marker removal | 338 | 338 |
| UKF pathologist LOPO | 288 | 288 |
| SNUH → UKF transfer (v2, matched scaling) | 16 | 16 |
| Exact-hex GraphSAGE/GAT LOPO sensitivity | 78 | 78 |

## Reproduce

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
cd preprint && tectonic deepgraph-gbm.tex                  # compile (pdflatex is not installed)
```

To rerun the benchmark from scratch, use the worker queue (finished runs are skipped):
`UKF=1 nohup caffeinate -i bash scripts/run_revision_worker.sh 0 >> results/revision/logs/w0.log 2>&1 &`

## Before submission

1. Fill in the UKF annotated-object source (red placeholder in the data-availability statement).
2. Shorten the abstract (591 words) to the target venue's limit.
3. Commit and push the code, results and preprint (nothing is committed yet).
4. Consider the minor points in PEER_REVIEW.md (title length; fixed-split spatial-map figure to the supplement; multiple-testing correction).
