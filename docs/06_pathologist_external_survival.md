# Chapter 6: Pathologist labels, external cohort and survival

## 1. Neuropathologist H&E labels (UKF cohort)

**Why this matters.** The SNUH labels are derived from expression, so a per-spot model might score
well just by reproducing the labelling rule. The Freiburg (UKF) cohort has an independent ground
truth: a neuropathologist's manual segmentation of the matched H&E image
([Ravi et al., *Cancer Cell* 2022](https://doi.org/10.1016/j.ccell.2022.05.009);
[Ritter et al., *Nat Cancer* 2025](https://doi.org/10.1038/s43018-024-00904-z)). It is also the
setting in which a graph neural network (NePSTA) was reported to classify histology accurately.

The cohort has 18 patients and 23 sections, with 25,374 annotated spots in five classes: cellular,
infiltrative, necrotic, MVP and normal. Graphs are built by `scripts/19_build_histology_graphs.py`
and jobs by `scripts/20_make_histology_jobs.py`; the analysis is `scripts/21_histology_analysis.py`.

### E1: within-cohort LOPO (18 folds, default hyperparameters, 3 seeds)

| Model | Pooled macro-F1 |
|---|---|
| **Logistic** | **0.419** |
| GraphSAGE | 0.346 ± 0.047 |
| MLP + neighbours (2-hop) | 0.346 ± 0.012 |
| MLP | 0.336 ± 0.025 |
| MLP + neighbours (1-hop) | 0.311 ± 0.030 |
| GAT | 0.291 ± 0.023 |

- No model differs significantly from GraphSAGE. Logistic vs GraphSAGE: +0.073, CI −0.032 to
  0.155, p = 0.13.
- The logistic advantage comes mostly from normal brain (F1 0.76 vs 0.30). GraphSAGE sends 87% of
  pathologist-annotated normal spots to infiltrative tumour, and the MLP sends 84%. This failure is
  shared by neural models with and without a graph, so it is not caused by message passing.
- Because all UKF models used default hyperparameters, part of the logistic lead may reflect its
  lower sensitivity to tuning.

### E2: transfer SNUH → UKF (train on expression-derived labels, test on pathologist labels)

| Model | Macro-F1 |
|---|---|
| **GraphSAGE** | **0.395 ± 0.030** |
| MLP | 0.380 ± 0.037 |
| MLP + neighbours (1-hop) | 0.377 ± 0.010 |
| Logistic | 0.352 |
| MLP + neighbours (2-hop) | 0.315 ± 0.073 |
| GAT | 0.315 ± 0.040 |

This is the **one setting in which a graph model leads**, and the lead is weak:
- **vs MLP:** +0.015 (CI −0.017 to 0.047; ahead for 11 of 18 patients; p = 0.25);
- **vs 1-hop MLP + neighbours:** +0.018 (p = 0.76);
- **vs logistic:** the pooled CI excludes zero, but the per-patient test does not (p = 0.30).

The result also depends on input scaling. With UKF counts at 10,000 instead of 100,000 per spot,
every model drops (0.20–0.34) and logistic regression ranks first.
- Published runs: `results/revision_transfer_v2`.
- Mis-scaled sensitivity runs: `results/revision_transfer`.

MVP is not recovered by any model (F1 ≤ 0.10). The SNUH training set's expression-defined MVP does
not match what a neuropathologist outlines.

## 2. External cohort: Greenwald et al. 2024

The seed-7 fixed-split model is applied, without retraining, to 13 Visium samples (25,488 spots)
([Greenwald et al., *Cell* 2024](https://doi.org/10.1016/j.cell.2024.03.029)). Inputs are scaled
exactly like the training matrix (log1p of counts at 100,000 per spot). Graphs use the true
hexagonal layout.

**The niche classifier does not transfer.**
- Across external spots: 77.8% predicted immune-hot, 20.0% necrotic, 1.4% normal, 0.8% immune-cold.
- 12 of 13 samples have fewer than 1% immune-cold, a class that is 67.9% of SNUH test spots.
- Predictions do not follow the annotated sampling region.
- At 10,000 counts per spot instead, the same model predicts 56% necrotic: its class balance flips
  under a monotone rescaling of the input.

**The MES head transfers partially.**
- Per-sample Pearson r with the Neftel MES meta-module
  ([Neftel et al., *Cell* 2019](https://doi.org/10.1016/j.cell.2019.06.024)) averages **0.21**
  (median 0.15, range −0.35 to 0.58).
- r is about 0.37 in bulk and contrast-enhancing samples and about 0.01 at the infiltrating edge.
- The pooled spot-level r = 0.65 is driven by between-sample differences and should not be read as
  within-sample agreement.

Results: `results/external_v2/` (matched scaling) and `results/external_v2_ts1e4/` (CP10k
sensitivity); Supplementary Tables S1 and S3.

## 3. Survival: TCGA-GBM (a bounded negative result)

No Visium GBM cohort with matched survival exists, so the biological hypothesis is tested in bulk
data. Each TCGA-GBM tumour is scored with fixed niche gene signatures (MES/myeloid, T-cell, hypoxia,
normal brain), combined with the pseudo-risk weights. This is a **signature score, not the trained
network**.

| Analysis (166 samples, 165 patients, 132 deaths) | Result |
|---|---|
| Univariate Cox HR per SD (95% CI) | 1.12 (0.92–1.36), p = 0.25 |
| Age-adjusted HR per SD | 1.03 (0.84–1.26), p = 0.81 |
| Concordance index | 0.52 |
| Log-rank p (median split) | 0.42 |
| Median OS, high vs low risk | 424 vs 444 days |
| Positive control: age, HR per SD | 1.55 (1.28–1.89), p < 0.001 |
| Smallest detectable HR per SD (80% power) | 1.28 |

Read this as: the pipeline can detect a real prognostic covariate (age), and **any effect of bulk
niche composition is smaller than about 1.28 per SD**. It is not evidence that niche composition
carries no prognostic information. Dropping the one duplicate-aliquot patient gives the same result
(HR 1.12, p = 0.25). Scripts: `06_survival_validation.py`, then `26_survival_power.py` (Table 3).

## 4. The slide-level risk head

The slide head was trained on the same pseudo-risk, but it does not recover that target on
held-out sections (r = 0.40, −0.80, 0.08 across seeds; n = 5 slides each). The paper therefore
reports no slide-level predictions, and no result depends on the head
(`results/tables/main/slide_risk_head_eval.csv`).

---

## Next

[Chapter 7: References](07_references_and_citations.md)
