# Internal peer review: DeepGraph-GBM (manuscript v6, 2026-10-09)

Reviewed: `preprint/deepgraph-gbm.tex` / `.pdf` (34 pages, 49 references, 16 figures including the graphical abstract, 9 tables). This review follows the full analysis audit in `ANALYSIS_AUDIT.md`. Every number in the text was checked against the result files, and the affected experiments were rerun.

## Summary

The paper tests whether spatial graph structure (GraphSAGE, GAT) improves supervised niche prediction in GBM Visium data over graph-free models (logistic regression, MLP, MLP + neighbor-averaged features). The evidence comes from:
- a fixed patient split and six repeated splits;
- a 13-fold leave-one-patient-out (LOPO) benchmark with equal nested tuning, ablations, marker-gene removal and an exact-hexagonal-graph sensitivity analysis;
- neuropathologist H&E labels in an independent cohort (UKF), both within that cohort and after transfer from SNUH;
- external transfer to the Greenwald cohort and a TCGA survival test.

Within each cohort, no graph model beat a matched graph-free model. After cross-cohort transfer, GraphSAGE was modestly but not significantly ahead.

## Overall assessment

The paper is a careful negative-result benchmark with conservative claims. After the audit, the claims match the evidence. Its strengths are equal-budget nested tuning, patient-level evaluation, labels independent of the input expression, and transparent disclosure of the remaining caveats. It suits a methods or benchmarking venue (e.g., *Bioinformatics*, *Genome Biology*, *NAR Genomics and Bioinformatics*, *PLoS Comput Biol*, *Briefings in Bioinformatics*).

## Major issues

1. **Data provenance (blocking).** The data-availability statement still has a red placeholder for the source of the annotated UKF SPATA objects. The pathologist-label analysis depends on them.
2. **Abstract length.** At 591 words, the abstract is well above typical limits (250–350). The Methods paragraph and the external-cohort and survival sentences can be condensed.
3. **The transfer result needs careful framing.** GraphSAGE led after SNUH→UKF transfer (0.395 vs MLP 0.380), and the result depended on input scaling. The text says so honestly. Reviewers may still ask whether graph smoothing helps under cohort shift. A brief discussion of why (noise smoothing) already exists; a dedicated transfer experiment would be future work.

## Minor issues

4. **Title** (25 words): consider shortening, e.g., "Spatial graph structure does not improve supervised niche prediction in glioblastoma Visium data: a patient-level benchmark with neuropathologist validation."
5. **Qualitative figures.** The fixed-split figures (UMAP, ROC, confusion matrix, spatial maps) use the original array-index graphs and argmax decisions; this is disclosed. Moving the fixed-split spatial-map figure to the supplement would keep the main figures consistent with LOPO.
6. **Multiple testing.** P values are uncorrected and declared descriptive. Some reviewers will still ask for Holm or BH correction within each family of tests.
7. **Multi-split overlap.** The six test sets share patients, so the df = 5 tests are optimistic. This is disclosed; LOPO is the primary analysis.
8. **HVG selection.** The gene panel was chosen without labels, partly from held-out patients' expression. This is disclosed and affects all models equally; nested HVG selection would remove the caveat completely.
9. **Bibliography.** 27 entries truncate long author lists with "others"; check the journal style.

## Verification performed

- **Results trace:** every number in the Results was recomputed from `results/` (LOPO, UKF E1 and E2, multi-split, ablations, marker removal, homophily, hex sensitivity, external cohort, survival).
- **Reruns after the audit:** external validation (matched normalization and hex geometry), SNUH→UKF transfer (16 jobs), exact-hex GraphSAGE and GAT LOPO (78 jobs), logistic convergence check, and the TCGA deduplicated survival analysis.
- **Citations:** all 49 keys resolve with no unused entries. 41 DOIs were checked against Crossref and 8 entries against arXiv (logs in `sources/`).
- **PDF:** all pages were rendered and inspected; there are no overlaps or unresolved references.
