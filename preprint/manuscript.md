# DeepGraph-GBM: Mapping Immunosuppressive Spatial Niches in Glioblastoma

**Md. Jubayer Hossain**¹,²,*

¹ Center for Health Innovation, Research, Action, and Learning - Bangladesh  
² DeepBio  
\* Corresponding author: hossainmj.me@gmail.com

*bioRxiv preprint*

---

## Abstract

**Background.** Glioblastoma (GBM) is spatially heterogeneous: within a single tumor, distinct niches—immune-cold regions of mesenchymal (MES) cancer cells and suppressive myeloid cells, rare immune-hot regions, normal peritumoral brain, and necrotic core—carry different biology and prognosis. Identifying these niches currently requires manual pathologist annotation or slow computational deconvolution.

**Results.** We present DeepGraph-GBM, an open, reproducible multi-task framework that predicts spatial niche identity directly from a 10x Visium slide, together with a rigorous patient-level benchmark. Treating each slide as a graph (spots as nodes, physical neighbors as edges), a GraphSAGE encoder learns spot embeddings that incorporate neighborhood context, feeding three task heads: a four-class niche classifier, a MES-probability regressor, and a patient-level survival-risk score. Trained on 100,488 spots from 28 sections of 13 IDH-wildtype GBM patients in the SNUH spatial atlas, the GNN achieves a held-out macro-F1 of 0.489 ± 0.027 (mean ± SD across three seeds) on unseen patients, with a representative model reaching per-class one-vs-rest AUROC of 0.82 (immune-cold), 0.98 (immune-hot), 0.73 (normal), and 0.96 (necrotic), and a MES-probability AUROC of 0.75. **A key finding is a rigorous negative benchmark:** when all models are trained under a matched optimization protocol and scored identically (validation-tuned thresholds, three seeds), the no-graph multilayer perceptron (macro-F1 0.578 ± 0.021) significantly *outperforms* the GNN (paired t-test *p* = 0.0017), and logistic regression (0.554) is comparable (*p* = 0.053)—indicating that, for this task, neighborhood graph structure does not improve niche prediction over per-spot expression alone. Repeating the full benchmark across six patient-level splits (72 trained models) confirms and sharpens this conclusion: logistic regression outperforms the GNN on all six splits (paired *p* = 0.0004), patient-split identity accounts for 3.6–8.1× more performance variance than random seed, and an attention-based (GAT) encoder significantly improves over GraphSAGE (*p* = 0.007) yet remains comparable to the MLP and below logistic regression. On an independent 13-sample cohort (Greenwald et al.), predicted niche compositions transfer across cohorts at the level of global niche composition, though they do not recapitulate annotated sampling regions and MES agreement is variable. **We also report an honest negative result for survival:** the niche-composition risk score does not significantly stratify TCGA-GBM overall survival (Cox HR 1.12, 95% CI 0.92–1.37, *p* = 0.25; C-index 0.52; log-rank *p* = 0.42), and should be treated as discovery-level.

**Conclusions.** DeepGraph-GBM provides an interpretable and fully reproducible framework for mapping immunosuppressive niches in GBM, and a rigorous benchmark showing that—contrary to the prevailing assumption—graph structure does not improve spatial niche prediction over a per-spot multilayer perceptron for this task. The non-significant survival association further underscores that spatial niche composition alone is insufficient for outcome prediction in this setting. All code, data, trained weights, and figures are openly released.

**Keywords:** glioblastoma; spatial transcriptomics; graph neural network; Visium; tumor microenvironment; mesenchymal; immunosuppression

---

## 1. Introduction

Glioblastoma (GBM) is the most aggressive primary brain tumor in adults, with a median overall survival of roughly 12–15 months despite maximal therapy [1]. A central driver of this poor prognosis is intratumoral heterogeneity: malignant cells occupy distinct, plastic cellular states—including a mesenchymal (MES) state associated with therapy resistance—that are shaped by, and in turn shape, the local tumor microenvironment [2, 3]. GBM is also profoundly immunosuppressive. Its microenvironment is dominated by tumor-associated macrophages and microglia, which can constitute up to half of the live cells in the tumor, alongside myeloid-derived suppressor cells, with relative exclusion of effector T cells [4–6].

Spatially resolved transcriptomics has made it possible to map this heterogeneity directly in tissue. Studies using 10x Visium and related platforms have shown that GBM is not randomly organized but forms stereotyped spatial niches: hypoxic/necrotic cores, perinecrotic zones of immunosuppression, infiltrating edges, and regions enriched for particular cancer-cell states [3, 7, 8]. A recent spatially resolved atlas of human GBM (the SNUH atlas) integrated Visium with matched single-cell and protein measurements to define the cellular and molecular patterns of these anatomical niches at high resolution [9]. A consistent picture has emerged: an *immune-cold* niche of MES cancer cells intermixed with suppressive myeloid cells and few T cells is associated with the worst prognosis, whereas *immune-hot* regions with T-cell infiltration are rare and relatively favorable [3, 5].

Despite this progress, two practical barriers limit the use of spatial niche information. First, assigning each spot to a niche still depends on manual pathologist annotation or on computational deconvolution against single-cell references, both of which are slow, subjective, or dependent on external references. Second, most spatial deep-learning methods for spatial transcriptomics focus on unsupervised spatial-domain clustering or cell-type deconvolution rather than directly predicting biologically meaningful, prognostically relevant niche labels [10–12].

A Visium slide is, however, intrinsically a graph: each spot is a node, and physically adjacent spots are connected. This structure is biologically meaningful—a spot's identity depends on its neighbors, because cancer cells recruit and reprogram myeloid cells from adjacent tissue. Graph neural networks (GNNs) such as GraphSAGE [13] and graph attention networks [14] are designed to learn exactly this kind of neighborhood-dependent representation, and are therefore a natural fit for spatial niche prediction.

Here we introduce **DeepGraph-GBM**, an open, reproducible multi-task GNN that predicts spatial niche identity, MES cancer-cell probability, and a patient-level survival-risk score directly from a Visium graph. We make three contributions:

1. **A method and open resource.** A complete, tested, MIT-licensed pipeline—from raw Visium data to trained model to publication-quality figures—released with all data, weights, and a reproduction guide.
2. **A rigorous benchmark with a negative result.** Patient-level held-out evaluation across three random seeds and six repeated patient splits, with matched training protocols and identical scoring for all models, showing that graph structure does not improve niche prediction for this task: logistic regression outperforms the GraphSAGE GNN on all six splits (paired *p* = 0.0004), and an attention-based GAT encoder—while significantly better than GraphSAGE (*p* = 0.007)—remains comparable to a no-graph multilayer perceptron.
3. **An honest external and clinical validation.** Predictions on an independent 13-sample cohort that transfer at the level of global niche composition (without recapitulating annotated regions), and a transparently reported *negative* survival analysis on TCGA-GBM.

---

## 2. Results

### 2.1 DeepGraph-GBM learns spatially structured niche embeddings

DeepGraph-GBM takes a Visium slide as input and represents it as a *k*-nearest-neighbor graph (*k* = 6) in physical space, with each spot described by the expression of 3,000 highly variable genes (HVGs). A three-layer GraphSAGE encoder (3000→256→256→128) produces spot embeddings that integrate neighborhood context; these feed three task heads: a four-class niche classifier (immune-cold, immune-hot, normal, necrotic), a MES-probability regressor, and an attention-pooled patient-level survival-risk head (see Methods).

We trained the model on 100,488 spots from 28 sections across 13 IDH-wildtype GBM patients in the SNUH spatial atlas [9], using silver-standard niche labels derived from the atlas's own anatomical-feature and metaprogram annotations (Methods). To visualize what the encoder learns, we embedded held-out test spots and projected them with UMAP [15] (Figure 1). The learned embeddings separate the four niches into distinct regions of the latent space, with necrotic and immune-cold spots forming particularly well-defined clusters, indicating that the encoder captures biologically meaningful spatial structure rather than memorizing per-spot expression.

> **Figure 1. Learned spot embeddings separate spatial niches.** UMAP projection of GraphSAGE spot embeddings for held-out test patients (SNU16, SNU33, SNU34), colored by silver-standard niche label (immune-cold, immune-hot, normal, necrotic).
> *(figure: `figures/main/Figure1_umap_niche.png`)*

### 2.2 A no-graph baseline outperforms the GNN in niche classification

We evaluated niche classification on held-out test patients (SNU16, SNU33, SNU34), using per-class decision thresholds tuned on separate validation patients and applied to test for **all** models (Methods). Across three random seeds the GNN achieved a macro-F1 of 0.489 ± 0.027 and an accuracy of 0.750 ± 0.050 (mean ± SD; Table 1). As a representative model we report seed 7, which reached a macro-F1 of 0.504, weighted-F1 of 0.698, and per-class one-vs-rest AUROC of 0.821 for immune-cold, 0.979 for immune-hot, 0.729 for normal, and 0.956 for necrotic (Figure 2).

To test whether neighborhood context contributes predictive signal, we compared the GNN against two baselines that use only per-spot expression—a multilayer perceptron (MLP) and multinomial logistic regression—trained under a **matched optimization protocol** (same optimizer, schedule, class weights, and validation-based early stopping) and scored across the **same three seeds under the identical tuned-threshold protocol** (Table 1, Table S6). Contrary to our expectation, the no-graph MLP (macro-F1 0.578 ± 0.021) **significantly outperformed** the GNN (0.489 ± 0.027; paired t-test on per-seed macro-F1, mean difference −0.089, 95% CI −0.105 to −0.073, t = −24.07, df = 2, *p* = 0.0017), and logistic regression (0.554) was comparable to the GNN (*p* = 0.053). The Wilcoxon signed-rank test is uninformative at n = 3 (minimum achievable two-sided *p* = 0.25). Two caveats bound this result. First, with only three seeds the paired test is low-powered, although the effect is large and consistent in direction across all three seeds (the MLP exceeds the GNN on every seed). Second, all seeds share a single fixed patient-level split, so the test quantifies training-seed variability rather than patient-sampling variability; we therefore repeat the entire comparison across six patient-level splits in the next section. Within these bounds, the data indicate that neighborhood graph structure does not improve niche prediction over per-spot expression for this task. An earlier version of this analysis scored the baselines with default argmax while the GNN used tuned thresholds, which spuriously favored the GNN; scoring all models identically reverses that result. Figure 4 dissects this reversal. Under the original mismatched scoring the GNN appeared best (0.489 versus 0.449 MLP and 0.350 logistic; Figure 4A); under the matched protocol the ranking inverts (Figure 4B). Two controls isolate why (Figure 4C). For logistic regression—whose training is deterministic and identical across protocols—switching from argmax to validation-tuned thresholds alone moves macro-F1 from 0.350 to 0.554, a pure scoring-rule effect of +0.20. A matched-protocol MLP scored by argmax (0.557) versus tuned thresholds (0.571) shows a much smaller scoring effect (+0.014); the remainder of the original MLP gap (0.449 under the earlier 120-epoch training) is attributable to the training-protocol mismatch. Scoring rules, not graph structure, created the apparent GNN advantage.

The row-normalized confusion matrix (Figure 3) shows that immune-cold and necrotic spots are classified most reliably, whereas the rare immune-hot class (~9% of spots) and the normal class are the main sources of error, with immune-cold over-predicted on test—a known consequence of class imbalance that we address in part with class-weighted loss and per-class thresholds (Methods; see Limitations).

> **Figure 2. Per-class niche classification performance.** One-vs-rest receiver operating characteristic curves for the four niche classes on held-out test patients, with area under the curve (AUC) indicated in the legend.
> *(figure: `figures/main/Figure2_roc_curves.png`)*

> **Figure 3. Niche classification confusion matrix.** Row-normalized confusion matrix on held-out test patients. Rows are true labels; columns are predicted labels; color indicates the proportion of each true class assigned to each predicted class.
> *(figure: `figures/main/Figure3_confusion_matrix.png`)*

**Table 1. Model comparison on held-out test patients (fixed split).** All models are mean ± standard deviation across three seeds (42, 7, 123), trained under a matched optimization protocol and scored with per-class thresholds tuned on validation patients and applied to test. Paired t-test on per-seed macro-F1: GNN vs MLP *p* = 0.0017 (MLP significantly higher); GNN vs logistic *p* = 0.053; GNN vs GAT *p* = 0.41 (underpowered at n = 3; see the six-split comparison in Figure 5, where GAT is significantly better). All seeds share a single fixed patient-level split, so *p*-values reflect seed-to-seed training variability (n = 3, df = 2; low-powered) rather than patient-sampling variability. Best value per row in bold.

| Metric | GNN (3 seeds) | GAT (3 seeds) | MLP (3 seeds) | Logistic (3 seeds) |
|---|---|---|---|---|
| Macro-F1 | 0.489 ± 0.027 | 0.539 ± 0.063 | **0.578 ± 0.021** | 0.554 ± 0.000 |
| Weighted-F1 | 0.707 ± 0.071 | 0.738 ± 0.051 | **0.764 ± 0.008** | 0.724 ± 0.000 |
| Accuracy | 0.750 ± 0.050 | 0.742 ± 0.064 | **0.789 ± 0.004** | 0.748 ± 0.000 |
| AUROC immune-cold | 0.782 ± 0.108 | 0.805 ± 0.069 | **0.838 ± 0.012** | 0.760 ± 0.000 |
| AUROC immune-hot | 0.947 ± 0.051 | 0.953 ± 0.018 | 0.987 ± 0.002 | **0.991 ± 0.000** |
| AUROC normal | 0.739 ± 0.051 | 0.760 ± 0.074 | **0.829 ± 0.017** | 0.687 ± 0.000 |
| AUROC necrotic | **0.951 ± 0.007** | 0.938 ± 0.008 | 0.943 ± 0.007 | 0.925 ± 0.000 |
| MES AUROC | 0.751 ± 0.041 | 0.668 ± 0.076 | **0.826 ± 0.016** | 0.751 ± 0.000 |
| MES Pearson *r* | 0.359 ± 0.016 | 0.273 ± 0.068 | **0.459 ± 0.021** | 0.410 ± 0.000 |

> **Figure 4. Identical scoring reverses the model ranking.** (A) Original mismatched evaluation: the GNN scored with validation-tuned per-class thresholds while the baselines were scored with default argmax. (B) Matched protocol: all models trained under the matched optimization protocol and scored with tuned thresholds (mean ± SD over three seeds). (C) Scoring-rule isolation: the same deterministic logistic model scored by argmax versus tuned thresholds (a pure scoring effect), and the matched-protocol MLP under both scoring rules (mean over three seeds).
> *(figure: `figures/main/Figure_protocol_reversal.png`)*

### 2.3 Robustness across patient-level splits and an attention-based encoder

The fixed-split comparison above quantifies seed-level variability only. To test whether the conclusion survives patient resampling—and whether a stronger graph encoder changes it—we repeated the full benchmark across six patient-level splits (8 train / 2 validation / 3 test patients each, with six distinct test sets; split 0 is the original split) and three seeds, for four models: the GraphSAGE GNN, the matched MLP, logistic regression, and a four-head graph-attention (GAT) encoder with otherwise identical architecture and training (Methods; 72 trained models in total).

Two findings emerge (Figure 5, Table S7). First, **patient-split identity is the dominant source of performance variance.** Split-mean macro-F1 spans 0.32–0.63 for GraphSAGE, 0.37–0.69 for the MLP, 0.43–0.68 for logistic regression, and 0.36–0.70 for GAT, and all four models rise and fall together across splits. The variance across split means is 3.6–8.1× the within-split seed variance (standard deviation across split means 0.09–0.12, versus 0.04–0.06 within split; logistic regression is deterministic given the data). Single-split benchmark numbers in this task should therefore be interpreted with caution regardless of model class.

Second, **the negative result holds, with one refinement.** Averaged over splits, logistic regression (split-mean macro-F1 0.572 ± 0.090) outperforms the GraphSAGE GNN (0.487 ± 0.109) on all six splits (paired t-test on split means, mean difference −0.086, 95% CI −0.112 to −0.059, t = −8.27, df = 5, *p* = 0.0004; Wilcoxon *p* = 0.031). The MLP (0.526 ± 0.108) exceeds the GNN on five of six splits, but at six splits this does not reach significance (mean difference −0.040, *p* = 0.137)—the fixed-split *p* = 0.0017 above therefore overstated confidence in the MLP comparison specifically. The GAT encoder (0.524 ± 0.116) significantly improves over GraphSAGE on all six splits (mean difference +0.037, 95% CI +0.015 to +0.059, *p* = 0.007), indicating that learned attention over neighbors is worth roughly four macro-F1 points over mean aggregation; however, GAT remains statistically indistinguishable from the no-graph MLP (*p* = 0.89) and below logistic regression (*p* = 0.034). Better use of graph structure thus closes much of the gap but does not overturn the central finding: for this task, per-spot expression with a simple classifier is at least as predictive as either graph encoder.

> **Figure 5. Patient-split identity dominates benchmark variance.** (A) Test macro-F1 across six patient-level splits for the GraphSAGE GNN, matched MLP, logistic regression, and GAT encoder (lines: split means over three seeds; points: individual seeds). All models rise and fall together across splits. (B) Per-split paired macro-F1 difference of each model relative to the GraphSAGE GNN (mean ± SD over seeds). Logistic regression exceeds the GNN on all six splits, GAT exceeds it on all six, and the MLP exceeds it on five of six.
> *(figure: `figures/main/Figure_split_robustness.png`)*

### 2.4 Predicted niche and MES maps are spatially coherent

Beyond per-spot metrics, the practical value of DeepGraph-GBM lies in the spatial maps it produces. Figure 6 shows a representative held-out test section (SNU16A): the silver-standard niche map (a), the model's predicted niche map (b), and the predicted MES-probability map (c). The predicted map recapitulates the large-scale organization of the true map—necrotic regions, immune-cold tumor, and normal edge—with errors concentrated at niche boundaries, as expected for a spot-level classifier. The MES-probability map highlights MES-enriched tumor regions consistent with the immune-cold niche, in line with the established association between the MES state and immunosuppression [2, 3]. Equivalent maps for the remaining test sections (SNU16B, SNU33A, SNU33B, SNU34A) are provided in Figures S1–S4.

> **Figure 6. Spatial niche and MES maps for a representative held-out section (SNU16A).** (a) Silver-standard niche labels; (b) model-predicted niche labels; (c) predicted MES cancer-cell probability per spot.
> *(figures: `figures/main/Figure4a_SNU16A_niche_true.png`, `Figure4b_SNU16A_niche_pred.png`, `Figure4c_SNU16A_mes.png`)*

### 2.5 External validation on an independent cohort

We applied the trained model, without retraining, to 13 region-labeled GBM Visium samples from Greenwald et al. [3] (Table S2). Predicted niche compositions were dominated by necrotic and immune-hot fractions across most samples and did **not** track the annotated sampling region: the single necrotic-region sample had a lower predicted necrotic fraction (0.43) than the bulk (mean 0.61) and contrast-enhancing (mean 0.60) samples, and infiltrating samples spanned 0.08–0.79. Predicted MES probability correlated only variably with the external Neftel MES meta-module reference [2] (mean Pearson *r* = 0.30, range −0.34 to 0.65; 3 of 13 samples with *r* ≤ 0.05). The model thus transfers across cohorts at the level of global niche composition, but it does not recapitulate annotated regions and its MES agreement is sample-dependent—likely reflecting differences in sample processing, region-label granularity, and domain shift between the two cohorts.

### 2.6 Survival-risk validation on TCGA-GBM: an honest negative result

Because no Visium GBM cohort has matched overall-survival data, the survival-risk head is trained on a niche-composition pseudo-risk and *validated* on TCGA-GBM bulk expression and clinical outcome (Methods). On the expression-matched TCGA cohort (*n* = 166; 132 deaths, 34 censored), the predicted risk score did **not** significantly stratify overall survival (Figure 7, Table 2): univariate Cox hazard ratio 1.12 (95% CI 0.92–1.37, *p* = 0.25), concordance index 0.52, age-adjusted HR 1.03 (*p* = 0.81), and Kaplan–Meier log-rank *p* = 0.42 (median OS 424 vs. 444 days for high vs. low predicted risk). We therefore treat the survival head as discovery-level and non-significant, and report it transparently rather than overclaiming a prognostic biomarker.

> **Figure 7. Survival stratification by predicted risk is non-significant.** Kaplan–Meier overall-survival curves for TCGA-GBM patients split at the median predicted risk score (high vs. low). The log-rank *p*-value is annotated in-plot; the separation is not statistically significant.
> *(figure: `figures/main/Figure5_km_survival.png`)*

**Table 2. TCGA-GBM survival validation of the niche-composition risk score** (*n* = 166; 132 events, 34 censored). The association is non-significant across all tests.

| Analysis | Result |
|---|---|
| Univariate Cox HR (95% CI) | 1.12 (0.92–1.37), *p* = 0.25 |
| Age-adjusted Cox HR | 1.03, *p* = 0.81 |
| Concordance index | 0.52 |
| KM log-rank *p* | 0.42 |
| Median OS, high vs. low risk | 424 vs. 444 days |

---

## 3. Discussion

We present DeepGraph-GBM, an open and reproducible graph neural network framework that predicts immunosuppressive spatial niches in glioblastoma directly from Visium data. Three findings stand out.

First, **graph structure does not improve spatial niche prediction for this task.** When all models are trained under a matched optimization protocol and scored identically across three seeds, a no-graph MLP significantly outperformed the GNN (paired *p* = 0.0017), and logistic regression was comparable (*p* = 0.053). This is a rigorous negative result: although GBM forms stereotyped, spatially organized niches [3, 7], a spot's own transcriptome already captures most of the niche-discriminative signal, and aggregating neighbor expression (GraphSAGE mean aggregation) did not add predictive value—possibly because neighbor averaging dilutes rare-but-critical signals (e.g., immune-hot spots) or because the dominant niche axes are cell-intrinsic. This cautions against the default assumption that graph context always helps in Visium niche prediction and motivates careful per-task benchmarking of whether spatial aggregation is beneficial. Repeating the benchmark across six patient-level splits confirmed the conclusion at the patient-sampling level: logistic regression outperformed the GNN on all six splits (paired *p* = 0.0004), and split identity accounted for 3.6–8.1× more performance variance than random seed—patient resampling, not seed resampling, is the uncertainty that matters. A four-head graph-attention encoder significantly improved over GraphSAGE (all six splits, *p* = 0.007), showing that learned neighbor weighting does add signal over mean aggregation, but it remained comparable to the MLP (*p* = 0.89) and below logistic regression (*p* = 0.034), so the central negative result is unchanged.

Second, **the model transfers across cohorts, with important caveats.** Without retraining, predicted niche compositions on an independent 13-sample cohort were dominated by necrotic and immune-hot fractions but did not track the annotated sampling region, and predicted MES probability correlated only variably with an external MES reference [2, 3] (mean *r* = 0.30, range −0.34 to 0.65). This indicates that the learned representations capture some transferable spatial biology at the level of global composition, but that cohort and domain shift limit region-level and fine-grained agreement.

Third, and importantly, **spatial niche composition alone did not predict survival.** The survival-risk score did not significantly stratify TCGA-GBM overall survival. We report this negative result deliberately. It has several plausible explanations: the survival head is trained on a niche-composition pseudo-risk rather than true outcomes (no Visium GBM cohort has matched survival); TCGA bulk expression is a spatially averaged proxy for niche composition; and the expression-matched cohort is modest in size with short median follow-up for censored patients, limiting power. More broadly, this result cautions against assuming that a spatially informed niche score is automatically prognostic, and it motivates future work using cohorts with true spatial–outcome pairing.

### 3.1 Limitations

Several limitations qualify these findings. (i) **Silver-standard labels.** Niche labels are derived from the SNUH atlas's own anatomical-feature and metaprogram annotations by auditable rules, not from independent pathologist ground truth; performance is therefore measured against a computational reference. (ii) **Class imbalance.** Immune-hot spots are rare (~9%), and the model over-predicts immune-cold on test; per-class thresholds tuned on validation patients mitigate but do not eliminate this. (iii) **Sensitivity to feature selection.** HVG selection materially affects external performance; we use scanpy's `seurat_v3` flavor on reconstructed counts because a naive raw-dispersion approach selected only housekeeping genes and degraded cross-cohort transfer. (iv) **Survival head.** As above, the survival analysis is a validation on a bulk-expression proxy and is non-significant. (v) **Single training cohort.** Although externally validated, the model is trained on a single (albeit large) atlas and would benefit from multi-atlas training. (vi) **External validation is limited.** Predictions on the independent Greenwald cohort did not track annotated sampling regions and MES agreement was variable (mean *r* = 0.30, range −0.34 to 0.65), indicating cohort/domain shift that the current model does not fully overcome. (vii) **Finite split resampling.** The repeated-split benchmark uses six patient-level splits drawn from 13 patients; training sets overlap across splits, so the splits are not fully independent, and at n = 6 splits the Wilcoxon test cannot resolve below *p* = 0.031. The MLP-versus-GNN comparison in particular is consistent in direction (five of six splits) but not significant at this sample size (*p* = 0.137); larger patient cohorts or fully nested cross-validation would tighten these estimates.

### 3.2 Conclusions and future work

DeepGraph-GBM provides an interpretable and fully reproducible framework for mapping immunosuppressive niches in GBM, together with a rigorous benchmark demonstrating that graph structure does not improve spatial niche prediction over a per-spot multilayer perceptron for this task. The honest, non-significant survival result defines a clear boundary for the approach and a concrete direction for the field: pairing spatial transcriptomics with true clinical outcomes. Future work will incorporate multi-atlas training, integration with single-cell deconvolution, and—critically—training and validation on spatial cohorts with matched survival.

---

## 4. Methods

### 4.1 Data

**Training data.** We used the spatially resolved human GBM atlas from Park, Shah, and colleagues (SNUH atlas; Nature Communications 2026) [9], accessed via the UCSC Cell Browser. The cohort comprises 100,488 spots across 28 Visium sections from 13 IDH-wildtype GBM patients, with per-spot anatomical-feature and metaprogram annotations. Both FFPE and fresh-frozen sections are included.

**External validation.** We used 13 region-labeled GBM Visium samples from Greenwald et al. [3] (Zenodo record 12624108, CC-BY).

**Survival validation.** We used TCGA-GBM bulk expression and clinical data from the NCI Genomic Data Commons [1]. Overall-survival times combine `days_to_death` (events) with `days_to_last_follow_up` (censored), the latter read from the GDC `diagnoses` record; the expression-matched cohort comprised 166 patients (132 deaths, 34 censored).

### 4.2 Graph construction and labels

Each section was represented as a *k*-nearest-neighbor graph (*k* = 6) in physical spot coordinates. Node features were the 3,000 most highly variable genes, selected with scanpy's `seurat_v3` flavor [16] on counts reconstructed from the log1p-normalized matrix via `expm1`; this choice was critical, as raw-dispersion selection on the normalized matrix selected only housekeeping genes and degraded external performance. Spots with fewer than 200 detected genes were excluded.

Silver-standard niche labels were derived from the atlas annotations by auditable priority-ordered rules: necrotic = pseudopalisading/perinecrotic anatomical features; immune-hot = immune metaprogram outside necrotic features; normal = concordant leading-edge grey/white-matter; immune-cold = malignant (cellular-tumor/infiltrating/hypoxic) metaprograms in tumor regions. Vascular and discordant spots were excluded from classifier training. The MES target was the fraction of MES malignant states (MES, MES.Hyp, MES.Ast) per spot.

### 4.3 Model architecture

The encoder is a three-layer GraphSAGE network [13] (3000→256→256→128) with batch normalization, ReLU, dropout 0.3, and mean aggregation, implemented in PyTorch Geometric [17] (1,751,815 parameters). As a comparator encoder we implemented a three-layer graph attention network (GAT) [14] with four attention heads per layer (hidden layers concatenate per-head outputs of width ⌊*d*/4⌋; the final layer averages heads) and otherwise identical dimensions, normalization, dropout, and task heads (985,863 parameters; the lower count reflects per-head partitioning of the hidden width). Three heads operate on the 128-dimensional spot embeddings: (1) a four-class niche classifier; (2) a MES-probability regressor (sigmoid); and (3) a survival-risk head using attention pooling over spots followed by a multilayer perceptron producing a scalar per slide.

### 4.4 Training and evaluation

Patients were split at the patient level (seed 42): train = SNU17/18/21/23/25/27/43/46, validation = SNU24/51, test = SNU16/33/34. The multi-task loss was *L* = 1.0·CE(niche) + 0.5·BCE(MES) + 0.5·MSE(pseudo-risk), with class-weighted cross-entropy and additional boosting of the rare immune-hot (×3) and necrotic (×1.5) classes. Models were trained with Adam (learning rate 10⁻³, weight decay 10⁻⁴) for up to 200 epochs with early stopping (patience 40) on full-graph batches (one section per step). Per-class decision thresholds were tuned on validation patients and applied to test. We report a three-seed summary (seeds 42, 7, 123) with seed 7 as a representative model. Baselines were a no-graph MLP and multinomial logistic regression trained on the same features and splits, and—critically—**trained and scored under a matched protocol**: the MLP used the same optimizer, cosine learning-rate schedule, boosted class weights, and early stopping on validation macro-F1 with best-checkpoint selection as the GNN, so that the GNN-versus-MLP contrast isolates neighborhood aggregation rather than training setup; and each model was evaluated across the same three seeds, with per-class thresholds tuned on validation patients and applied to test, exactly as for the GNN. The GNN-versus-baseline comparison used a paired t-test (and Wilcoxon signed-rank test) on per-seed macro-F1 (df = 2); two pre-specified comparisons (GNN vs MLP, GNN vs logistic) with Bonferroni α = 0.025 as a sensitivity threshold. As a scoring-rule control, the deterministic logistic model from the original argmax-scored analysis was rescored with validation-tuned thresholds (isolating the pure scoring effect, since its training is unchanged), and the matched-protocol MLP was scored under both argmax and tuned thresholds across the three seeds. All three seeds share a **single fixed patient-level split**; the paired test therefore quantifies training-seed variability only and does not capture patient-sampling (split) variability.

**Repeated-split benchmark.** To quantify patient-sampling variability, we generated five additional patient-level splits (generator seeds 1001–1005) with the same 8/2/3 train/validation/test patient counts, constrained so that all six splits (including the original) have distinct test sets. All four models (GraphSAGE, GAT, MLP, logistic regression) were trained under the matched protocol above for the same three seeds on every split (72 trained models). Model comparisons used paired t-tests and Wilcoxon signed-rank tests on split-mean macro-F1 (n = 6 splits, df = 5; note the Wilcoxon minimum two-sided *p* at n = 6 is 0.031). Performance variance was decomposed into across-split (patient-sampling) and within-split (seed) components via the variance of split means versus the mean within-split seed variance.

### 4.5 External and survival validation

For external validation, the trained model was applied to the Greenwald cohort without retraining; predicted MES probability was correlated with the Neftel MES meta-module [2]. For survival validation, per-patient niche-composition scores were computed from TCGA bulk expression and related to overall survival using Cox proportional-hazards regression (univariate and age-adjusted), the concordance index, and Kaplan–Meier analysis with the log-rank test, implemented in lifelines [18]; patients were split at the median risk score for the Kaplan–Meier curves.

### 4.6 Reproducibility

All analyses are implemented as a numbered, tested pipeline (18 unit tests) in the released repository. Figures were generated programmatically at 300 dpi with an Okabe–Ito colorblind-safe palette and no embedded titles. Software versions: PyTorch 2.7 (2.13 for the repeated-split and GAT benchmark), PyTorch Geometric 2.8, scanpy 1.11, lifelines 0.30, UMAP [15].

---

## Data and code availability

All code (MIT license), the complete raw and preprocessed data, trained model weights, and every figure and table are released as a single self-contained bundle with a step-by-step reproduction guide. Project repository: https://github.com/jubayer-hossain/deepgraph-gbm. Training data: SNUH GBM atlas [9] (UCSC Cell Browser, https://cells.ucsc.edu/?ds=multiomic-gbm). External cohort: Greenwald et al. [3] (Zenodo 12624108). Survival data: TCGA-GBM via the NCI GDC [1].

## Competing interests

The author declares no competing interests.

## Acknowledgments

We thank the investigators who generated and openly shared the SNUH GBM atlas, the Greenwald et al. cohort, and the TCGA program.

---

## References

1. McLendon R, Friedman A, Bigner D, et al. Comprehensive genomic characterization defines human glioblastoma genes and core pathways. *Nature* 455(7216):1061–1068, 2008. doi:10.1038/nature07385.
2. Neftel C, Laffy J, Filbin MG, et al. An integrative model of cellular states, plasticity, and genetics for glioblastoma. *Cell* 178(4):835–849, 2019. doi:10.1016/j.cell.2019.06.024.
3. Greenwald AC, Darnell NG, Hoefflin R, et al. Integrative spatial analysis reveals a multi-layered organization of glioblastoma. *Cell* 187(10):2485–2501.e26, 2024. doi:10.1016/j.cell.2024.03.029.
4. Khan F, Pang L, Dunterman M, Lesniak MS, Heimberger AB, Chen P. Macrophages and microglia in glioblastoma: heterogeneity, plasticity, and therapy. *J Clin Invest* 133(1):e163446, 2023. doi:10.1172/JCI163446.
5. Lin H, Liu C, Hu A, Zhang D, Yang H, Mao Y. Understanding the immunosuppressive microenvironment of glioma: mechanistic insights and clinical perspectives. *J Hematol Oncol* 17:35, 2024. doi:10.1186/s13045-024-01544-7.
6. DeCordova S, Shastri A, Tsolaki AG, et al. Molecular heterogeneity and immunosuppressive microenvironment in glioblastoma. *Front Immunol* 11:1402, 2020. doi:10.3389/fimmu.2020.01402.
7. Ravi VM, Will P, Kueckelhaus J, et al. Spatially resolved multi-omics deciphers bidirectional tumor-host interdependence in glioblastoma. *Cancer Cell* 40(6):639–655, 2022. doi:10.1016/j.ccell.2022.05.009.
8. Ståhl PL, Salmén F, Vickovic S, et al. Visualization and analysis of gene expression in tissue sections by spatial transcriptomics. *Science* 353(6294):78–82, 2016. doi:10.1126/science.aaf2403.
9. Park HJ, Sonpatki P, Schroeder BA, et al. A spatially resolved human glioblastoma atlas reveals distinct cellular and molecular patterns of anatomical niches. *Nat Commun* 17(1):2951, 2026. doi:10.1038/s41467-026-69716-2.
10. Xu H, Wang S, Fang M, et al. SPACEL: deep learning-based characterization of spatial transcriptome architectures. *Nat Commun* 14:7603, 2023. doi:10.1038/s41467-023-43220-3.
11. Peng L, He X, Peng X, et al. STGNNks: Identifying cell types in spatial transcriptomics data based on graph neural network, denoising auto-encoder, and k-sums clustering. *Comput Biol Med* 163:107440, 2023. doi:10.1016/j.compbiomed.2023.107440.
12. Liu T, Fang ZY, Zhang Z, Li M, Yin MZ. A comprehensive overview of graph neural network-based approaches to clustering for spatial transcriptomics. *Comput Struct Biotechnol J* 21:3957–3972, 2023. doi:10.1016/j.csbj.2023.11.055.
13. Hamilton WL, Ying Z, Leskovec J. Inductive representation learning on large graphs. *NeurIPS* 30, 2017. arXiv:1706.02216.
14. Veličković P, Cucurull G, Casanova A, Romero A, Liò P, Bengio Y. Graph attention networks. *ICLR*, 2018. arXiv:1710.10903.
15. McInnes L, Healy J, Melville J. UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426, 2018.
16. Wolf FA, Angerer P, Theis FJ. SCANPY: large-scale single-cell gene expression data analysis. *Genome Biol* 19(1):15, 2018. doi:10.1186/s13059-017-1382-0.
17. Fey M, Lenssen JE. Fast Graph Representation Learning with PyTorch Geometric. *ICLR Workshop on Representation Learning on Graphs and Manifolds*, 2019. arXiv:1903.02428.
18. Davidson-Pilon C. lifelines: survival analysis in Python. *J Open Source Softw* 4(40):1317, 2019. doi:10.21105/joss.01317.

---

## Supplementary material

- **Figures S1–S4** (a/b/c per section): true niche map, predicted niche map, and MES-probability map for held-out test sections SNU16B, SNU33A, SNU33B, and SNU34A (`figures/supplement/`).
- **Table S1.** Baseline (logistic, MLP) metrics. **Table S2.** Per-sample external-validation summary (13 Greenwald samples). **Table S3.** Per-section metadata. **Table S4.** Per-spot external predictions with Neftel MES reference. **Table S5.** Per-patient TCGA niche scores and survival covariates. **Table S6.** Per-seed matched-protocol metrics for all models on the fixed split. **Table S7.** Per-split, per-seed metrics for all four models across the six patient-level splits (72 runs) (`tables/supplement/`).
