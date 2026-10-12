# Revision notes

Version snapshots named below (`deepgraph-gbm.vN.tex`, `references.vN.bib`) live in
`preprint/archive/` since v7.2.

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

## v7 (2026-10-09): response to the independent review of v6 (`deepgraph-gbm.v6.tex`, `references.v6.bib`)

No model was retrained; every new number is computed from the existing run
records. Point-by-point response in `PEER_REVIEW.md`.

**Analyses added** (`scripts/24`–`28`):
- Supplementary Table S9: the per-spot neuropathologist annotation table (68,015 rows,
  25,374 annotated, 23 sections), released so the UKF labels need not be re-derived
  from the source SPATA objects.
- Supplementary Table S10 and Supplementary Figure S7c: labeled class support of each
  six-split threshold-tuning block (immune-hot 15, 2,997, 7, 10, 1, 3,705 for splits 0–5).
- Supplementary Table S11: the six-split contrasts rescored over the three adequately
  supported classes (logistic − GraphSAGE 0.086 → 0.070; MLP − GraphSAGE 0.040 → 0.047).
- Supplementary Table S12 and Supplementary Figure S7a: per-patient contrasts under
  class-support floors of none, 50 and 200 spots, with Holm adjustment.
- Supplementary Table S13 and Supplementary Figure S7b: pooled LOPO macro-F1 under argmax
  versus validation-tuned thresholds for every model.
- `results/tables/main/slide_risk_head_eval.csv`: the slide-level head against its own
  pseudo-risk target on held-out sections (r = 0.40, −0.80, 0.08 over seeds).
- Table 3 rebuilt on a per-SD scale with an age positive control (HR 1.55 per SD) and the
  minimum detectable effect (HR 1.28 per SD at 80% power, 132 events).

**Text**:
- Section 2.3 no longer calls the six-split analysis "the primary test"; it and the Methods
  now agree that pooled LOPO macro-F1 is primary. Section 2.4 gained the tuning-support
  disclosure and the restricted-class sensitivity, and its p values are declared descriptive.
- Section 2.5 leads with the pooled difference and its patient bootstrap; the per-patient
  tests follow with Holm adjustment and the support-floor sensitivity. The MLP-versus-
  GraphSAGE result at p = 0.048 is explicitly not claimed.
- New Section 2.5 paragraph on the decision rule (tuning helps graph models by 0.019–0.034
  and harms the per-spot models, halving the graph-free advantage).
- Section 2.9 opens with the slide-head negative result; the survival paragraph is per SD
  and bounded by the positive control and the detectable effect.
- Introduction no longer advertises slide-level risk scoring as a working capability.
- Abstract: 591 → 353 words over v6 and v7; the "up to 0.7" range is now 0.03–0.72 with its
  explanation; the survival sentence carries the detectable-effect bound.
- Limitations: items on threshold calibration, per-patient scoring, survival power and the
  slide head; renumbered (i)–(ix).
- Discussion: the decision rule and the scoring unit are added to the "evaluation choices"
  argument; the survival paragraph gives the bound.
- Data availability rewritten around Supplementary Table S9; it credits both source cohorts
  without asserting a file-level location in either deposit.
- MES correlation labeled as point-biserial; a clause distinguishes Table 1's MLP fit from
  Supplementary Table S6's independent re-run.

**Supplementary numbering**: v6 already printed Supplementary Tables S1–S8, so the new
v7 tables start at S9 (S9 UKF annotations, S10 tuning support, S11 restricted-class
contrasts, S12 support floor, S13 decision rule) and the new figure is S7. The repository
file names were set to match the printed numbers, and the `tab:s*` labels, which in v6 were
offset by one from the numbers they printed, were realigned.

**Figures**: the main-text map figure is now the leave-one-patient-out maps (tuned
thresholds, per-fold configurations), consistent with the primary analysis; the fixed-split
argmax maps moved to Supplementary Figure S1. Supplementary figure labels are semantic
(`fig:maps16a`, `fig:ukfmaps`, …) and supplementary table labels now match their printed
numbers.

**References**: added Schoenfeld (1983) for the sample-size formula; 53 entries, all cited.

**Build**: 39 pages; no undefined or multiply-defined references, no missing graphics and no
overfull boxes. One underfull (loose) line of badness 1400 remains in a single bibliography
entry; it is cosmetic and was left alone.

**Not done**: repairing the six-split threshold protocol, which needs 72 retrained models.
The defect is disclosed and the analysis demoted instead.

## v7.1 (2026-10-10): code, modeling and evaluation audit

No analysis changed; 99 independent re-derivations of published values all agree and the
18 unit tests pass (`preprint/CODE_AUDIT.md`, `scripts/29_code_audit_checks.py`).

**Manuscript**: Methods now state that the leave-one-patient-out and UKF benchmarks use the
niche and MES heads only (the three-term objective applies to the fixed-split models); that
`GATConv` applies dropout to its attention coefficients, a regularizer no other model carries,
with a new Limitations item (ix); that the "normal" rule requires both annotations to be
leading-edge but not to name the same tissue (3,158 of 27,553 matching spots pair grey with
white matter); that overall homophily is edge-level while the per-class values are a per-spot
mean; and that the niche head is a single linear layer.

**Repository**: `docs/WORKFLOW.md` Section 6 rewrote the headline results against the verified
values and states that the protocol-reversal claim was retracted in v5 — it had still been
presented as a finding, alongside pre-correction external numbers and the withdrawn seed-paired
p = 0.0017. The figure/table map was updated from its v4-era form. Scripts 13, 14 and
`run_baselines.py` are labeled superseded; stale supplementary numbering was fixed in the
docstrings of 09 and 12; `configs/labels.yaml` no longer describes the MES target as a
fraction; and `pyproject.toml` dropped two unused dependencies (`lab`, `jupyter`) and gained
`umap-learn`, which Figure 1 needs.

## v7.2 (2026-10-13): repository cleanup to the benchmark framing

No analysis, number or manuscript text changed; the manuscript is still v7.1.

**Documentation.** `docs/` chapters 00–07 were written for the original framing (a multi-task
GNN presented as the method) and carried pre-audit facts: 23 sections / ~35,000 spots instead of
28 / 100,488, an atlas citation with the wrong authors, MES r = 0.30 instead of 0.21, the
retracted protocol reversal, the withdrawn seed-paired p = 0.0017, and a continuous MES target.
All chapters were rewritten from the v7.1 manuscript and regrouped (03 models compared, 04
evaluation protocol, 05 benchmark results, 06 pathologist/external/survival). `docs/WORKFLOW.md`
was rewritten to cover every script (01–29), including the LOPO, UKF and review steps it lacked,
and to map every manuscript figure and table to its file and script. Chapter 7 now lists only
entries of `references.bib`.

**Graphical abstract.** `assets/graphical_abstract.png`, shown at the top of the README, was still
the pre-pivot graphic that presents the GNN as the method; it is now the data-driven abstract
from `scripts/22`.

**Reproducibility gap closed.** The exact-hex graphs (`data/preprocessed_hex`) had no builder in
the repository. `scripts/23a_build_hex_graphs.py` now builds them and reproduces the published
graphs bit for bit (edge sets, edge weights and coordinates).

**Provenance corrections.** Scripts 13 and `run_baselines.py`, labelled superseded in v7.1,
produce the argmax values in section 2.3 (MLP 0.557, logistic 0.541), and
`results/revision_transfer` holds the CP10k transfer sensitivity quoted in the Results; all are
kept and relabelled. Script 20 now writes E2 jobs to `results/revision_transfer_v2`, the published
run; `01_download_data.sh` now defaults to `data/raw`, where the pipeline reads; stale default
paths in 08, 09 and `run_baselines_multiseed.py` were corrected.

**Table 1 statistics record corrected.** `results/tables/main/Table1_paired_test.json` and
`Table1_model_comparison.json` still held the seed-paired t-test withdrawn in v5 (p = 0.0017)
and a note restating the retracted protocol-reversal claim, and no tracked script produced the
Welch statistics the manuscript reports. `scripts/08_paired_test.py` now computes exactly the
reported tests (MLP vs GraphSAGE Welch t = 4.50, p = 0.012, 95% CI 0.033–0.145; logistic
one-sample p = 0.053; GAT vs GraphSAGE Welch p = 0.30), and both files were regenerated. The
macro-F1 means and Supplementary Table S5 are unchanged.

**Archived** (moved, not deleted): manuscript snapshots v1–v7, their bibliographies and the
pre-pivot `manuscript.md`/`.pdf` and graphical abstracts → `preprint/archive/`;
`14_fig_protocol_reversal.py` → `scripts/archive/`; the unmatched multi-seed baselines, the
original-protocol tables and the pre-fix CP10k Greenwald tables → `results/archive/`.
`results/figures/` (intermediate renders, already in `.gitignore`) is no longer tracked.

**Title.** Changed from *Spatial Graph Structure Does Not Improve Supervised Niche Prediction in
Glioblastoma* to *Comparing Graph-Based and Graph-Free Models for Spatial Niche Prediction in Glioblastoma*: short, descriptive and free of negation or question form. It names the
contrast the paper actually tests (two GNN encoders against matched graph-free models, not a
survey of GNN architectures) and the task, leaving the finding to the abstract. Updated in the manuscript (and PDF
metadata), `CITATION.cff`, `README.md` and `SUBMISSION.md`.

**Metadata.** `pyproject.toml` and the package version now read 0.7.1, matching `CITATION.cff`, with
a description of the benchmark rather than of a GNN method; `configs/default.yaml` documents that
its `paths` block is informational. The notebooks were revised to the current results (see
`notebooks/`).

**Rename (v7.2).** The project, repository and package are renamed from DeepGraph-GBM /
`deepgraph-gbm` / `deepgraph_gbm` to `spatial-niche-gbm` / `spatial_niche_gbm`, since the work
is no longer presented as a GNN method. The model class `DeepGraphGBM` is now `MultiTaskGNN`, and
the manuscript source is `preprint/spatial-niche-gbm.tex` (repository URL in the availability
statement updated; two sentences that named "DeepGraph-GBM" as the contribution reworded).
Saved checkpoints and graphs contain only state dicts and tensors, so they load unchanged.
`uv.lock` was regenerated against the current `pyproject.toml`. The GitHub Actions CI workflow was
removed. Audit records and archived versions keep the old name as historical documents.

