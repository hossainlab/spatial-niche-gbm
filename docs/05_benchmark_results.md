# Chapter 5: Benchmark results on the SNUH cohort

All numbers in this chapter are those of manuscript v7.1 (`preprint/spatial-niche-gbm.tex`).
`scripts/28_verify_manuscript_numbers.py` and `scripts/29_code_audit_checks.py` re-derive them
from `results/`.

## 1. Fixed primary split, matched protocol (Table 1)

Test patients SNU16, SNU33, SNU34; validation-tuned thresholds; mean ± SD over seeds 42, 7, 123.

| Metric | GraphSAGE | GAT | MLP | Logistic |
|---|---|---|---|---|
| Macro-F1 | 0.489 ± 0.027 | 0.539 ± 0.063 | **0.578 ± 0.021** | 0.554 |
| Accuracy | 0.750 ± 0.050 | 0.742 ± 0.064 | **0.789 ± 0.004** | 0.748 |
| MES AUROC | 0.751 ± 0.041 | 0.668 ± 0.076 | **0.826 ± 0.016** | 0.751 |

The MLP leads GraphSAGE by 0.089 (95% CI 0.033 to 0.145; Welch p = 0.012). Logistic vs GraphSAGE
gives p = 0.053, and GAT vs GraphSAGE p = 0.30. With three seeds on one split, these p-values
reflect training-seed variability, not patient sampling.

The decision rule hardly matters here: argmax instead of tuned thresholds changes macro-F1 by less
than 0.02 (logistic 0.541 vs 0.557; MLP 0.557 ± 0.050 vs 0.571 ± 0.021).

## 2. Six patient splits: patient sampling dominates

Across 72 runs, split-mean macro-F1 ranges from 0.32 to 0.70, and all four models rise and fall
together. Across-split variance is 3.5× (GAT), 3.6× (GraphSAGE) and 8.1× (MLP) the within-split
seed variance. The gap between the best and worst split, about 0.3 macro-F1, is roughly three times
the largest gap between models. A single-split comparison would have reported a significant MLP
advantage (p = 0.012) that does not hold across splits (p = 0.14).

The six-split model ordering (logistic > MLP ≈ GAT > GraphSAGE) agrees with LOPO, but it is **not
used for ranking**. The tuning blocks hold 15, 2,997, 7, 10, 1 and 3,705 immune-hot spots, so
thresholds are effectively arbitrary in four splits (Supplementary Tables S10–S11).

## 3. Leave-one-patient-out, equal tuning budget (primary analysis)

| Model | Pooled macro-F1 | AUROC cold / hot / normal / necrotic | Δ vs GraphSAGE (95% CI) | Patients ahead, Wilcoxon p |
|---|---|---|---|---|
| **Logistic** | **0.691** | **0.82 / 0.97 / 0.90 / 0.96** | **+0.039 (0.009, 0.076)** | 12/13, 0.001 |
| MLP | 0.655 ± 0.020 | 0.80 / 0.89 / 0.86 / 0.91 | +0.003 (−0.045, 0.052) | 9/13, 0.048 |
| MLP + neighbours | 0.637 ± 0.047 | 0.74 / 0.93 / 0.79 / 0.92 | −0.015 (−0.065, 0.040) | 8/13, 0.27 |
| GraphSAGE | 0.652 ± 0.018 | 0.77 / 0.92 / 0.83 / 0.92 | — | — |
| GAT | 0.649 ± 0.018 | 0.75 / 0.90 / 0.80 / 0.92 | −0.004 (−0.032, 0.018) | 5/13, 0.34 |

What the table supports:
- **Logistic regression beats GraphSAGE.** It is the only contrast whose CI excludes zero, and it
  holds per patient (Holm-adjusted p = 0.003). It also has the highest AUROC for every class and the
  highest MES AUROC (0.90 vs 0.83).
- **The MLP, GAT and MLP + neighbours are level with GraphSAGE.** The MLP's per-patient p = 0.048
  does not survive Holm correction or a 50-spot support floor, so it is not claimed.
- **Adding neighbour-averaged expression to the MLP does not help** (−0.018 vs plain MLP, p = 0.84).
  Explicit access to the neighbourhood, with or without learned message passing, adds nothing.
- **The single graph-model win** is normal-class F1, where GraphSAGE (0.585) is marginally above
  the MLP (0.571).

**Patients matter more than models.** Per-patient macro-F1 runs from 0.03–0.07 (SNU46, almost all
one class) to 0.62–0.72 (SNU27) for every model. Patient identity contributes 5–21× more variance
than seed.

## 4. Robustness: nothing rescues the graph

| Check | Result |
|---|---|
| GraphSAGE depth 1–3 × k = 6/12/18 (9 settings) | Mean per-patient macro-F1 0.382–0.425, no trend. Every setting is below a one-layer MLP (0.460) and below logistic regression (0.448). |
| MLP + neighbours, hops × k | 0.383–0.425. Never helps. |
| Exact hexagonal graphs (78 runs) | GraphSAGE 0.651 (vs 0.652), GAT 0.611 (vs 0.649). Logistic still ahead (Wilcoxon p = 0.002, 7×10⁻⁴). |
| Remove top-100 label-associated genes | GraphSAGE −0.065, logistic −0.037, MLP −0.008. The logistic–GraphSAGE gap *widens* to 0.071. |
| Remove top-500 genes | All models drop 0.15–0.17. Logistic still highest (0.293). |
| Label homophily | 0.80: neighbours are informative in principle ([Chapter 2 §3](02_graph_construction.md)) |
| Decision rule (argmax vs tuned) | Argmax would *widen* the logistic lead to 0.071 ([Chapter 4 §1](04_evaluation_protocol.md)) |

If the graph supplied information the spot lacks, removing marker genes should have hurt the
graph-free models more. It did not.

## 5. Why the graph does not help (the paper's interpretation)

1. **A Visium spot already pools several cells**, and neighbouring spots have strongly correlated
   expression. The niche axes being predicted (hypoxia, MES, myeloid, T-cell programmes) are
   largely cell-intrinsic, so the neighbourhood mostly repeats what the spot already carries.
2. **Aggregation smooths.** Predicted maps from GraphSAGE are more contiguous than the labels. This
   can produce plausible-looking regions while erasing small or scattered ones. Immune-hot and
   necrotic, the classes the graph models recover worst, are the least homophilous.
3. **With 13–18 patients**, the extra parameters of a graph encoder are hard to fit in a way that
   generalises across patients. A strongly regularised linear model is robust to patient shift.

The paper does **not** claim that spatial context is biologically unimportant. The claim is
narrower: at Visium resolution and this cohort size, standard graph encoders did not extract
predictive signal beyond per-spot expression.

## 6. Spatial maps

LOPO maps (Figure 7: SNU16A, SNU34A) and fixed-split maps (Supplementary Figures S1–S5) show two
failure modes:
- **Over-smoothing:** GraphSAGE produces contiguous blocks that can cover the wrong class, while
  logistic regression's errors are scattered.
- **Majority-class over-assignment** in the fixed-split models: SNU16A is predicted almost entirely
  immune-cold, and its large necrotic region is missed.

The fixed-split maps use argmax and are qualitative.

## A retracted claim

Earlier versions reported a **"protocol reversal"**: that the GNN looked best only because it alone
was scored with tuned thresholds, and that scoring all models identically reversed the ranking.
Its baselines came from a superseded feature set.
The claim was withdrawn in v5, and the section, figure and table were removed in v6
([`preprint/ANALYSIS_AUDIT.md`](../preprint/ANALYSIS_AUDIT.md), finding 1). The figure script is
kept in `scripts/archive/` for provenance only.

---

## Next

[Chapter 6: Pathologist labels, external cohort and survival](06_pathologist_external_survival.md)
