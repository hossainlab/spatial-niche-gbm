# Revision notes

## v4 (2026-10-09): `deepgraph-gbm.tex` (v3 snapshot: `deepgraph-gbm.v3.tex`, `references.v3.bib`)

**New results** (all 338 ablation and marker-removal runs complete; `scripts/17_revision_analysis.py` rerun):
- Results 2.6: new paragraphs "Graph depth and neighborhood size" and "Removing the most label-associated genes", plus new Figure 7 (`FigR2_ablation.png` and `FigR3_marker_removal.png`). The paired Wilcoxon statistics were computed from `results/revision/runs/`.
- Abstract (Results): one sentence on the ablations and marker-gene removal.
- Discussion ("Why did the graph not help?"): adds the ablation and marker-removal evidence.
- Methods (Ablations): the default model configurations used for marker removal; single seed; per-patient mean summary.
- Methods (Statistics): p values are uncorrected and descriptive.
- Limitations (iv): power of the single-seed ablations.
- UKF Results: caveat about default hyperparameters.

**Graphical abstract:** new, data-driven graphical abstract (`scripts/22_graphical_abstract.py` → `figures/graphical_abstract.png/.pdf`) placed before the abstract. The old AI-generated graphic presented the GNN as the method, which contradicts the findings, so it is no longer used (the file is kept in `preprint/`).

**References** (checked against Crossref and arXiv; logs in `sources/`):
- neftel2019: pages 835–849.e21
- ravi2022: pages 639–655.e13
- lin2024: number 1, article 31 (was 35)
- peng2023: volume 166 (was 163)
- liu2023review: 2024, volume 23, pages 106–128 (was 2023, 21, 3957–3972)
- ruizmoreno2025: 27(9):2281–2295
- ostrom2023: Suppl 4 (was "12 Suppl 2"); "{United States}" capitalization protected

**Layout:**
- Removed the `\clearpage` after the abstract, which left page 2 nearly blank.
- Figure 7 panels are placed side by side, so the figure no longer leaves a half-empty page.
- The PDF went from 35 to 33 pages.

**Still open:**
- UKF annotated-object source (red placeholder in the data-availability statement).
- The abstract is about 570 words.
- See PEER_REVIEW.md.

## v5 (2026-10-09): scientific audit (`deepgraph-gbm.v5.tex`)
See ANALYSIS_AUDIT.md. Major corrections:
- Protocol-reversal claim retracted (superseded-feature baselines).
- External and transfer normalization fixed (CP100k); the transfer ranking changed (GraphSAGE first, not significant).
- Hex-graph sensitivity analysis added.
- Welch tests replace seed-paired t-tests.
- MES-target description, multi-split overlap, HVG reuse, solver convergence and TCGA duplicate disclosed.

## v6 (2026-10-09): removed the non-public "earlier analysis" material
The earlier version was never public, so the following were removed:
- Section 2.4
- Figure 4 (protocol comparison)
- Contribution 5
- Old Table S1
- The "our earlier analysis" Discussion text

The decision-rule fact (argmax vs tuned: < 0.02 macro-F1) is kept as one sentence in section 2.3.

Renumbering:
- Supplementary tables are now S1–S8, and the repository files are renamed to match (S2→S1 … S7→S6).
- The original-protocol baselines moved to `results/tables/archive/`.
