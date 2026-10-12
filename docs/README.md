# spatial-niche-gbm documentation

spatial-niche-gbm is a **controlled comparison of graph-based and graph-free models**. It asks whether
spatial graph neural networks improve supervised niche prediction in glioblastoma Visium data over
matched graph-free models. **They do
not.** These chapters explain the biology, the design and the results for readers without a
background in genomics or deep learning.

If you only read one page, read [Chapter 0](00_overview_and_roadmap.md). Its §5 explains how the
project changed from "a GNN method" to "a comparison of that method with graph-free models", which is the usual source of
confusion when reading older material.

| Chapter | Document | What you will learn |
|---|---|---|
| **00** | [Overview](00_overview_and_roadmap.md) | The question, why it matters, the answer, and the project's history |
| **01** | [Biology, data and labels](01_biology_and_spatial_data.md) | GBM niches, Visium, the four cohorts, how niche labels are derived, HVG features |
| **02** | [Graphs, homophily and splits](02_graph_construction.md) | How sections become graphs, the array-index caveat, label homophily, patient-level splits |
| **03** | [Models compared](03_models_compared.md) | Logistic regression, MLP, MLP + neighbours, GraphSAGE, GAT; heads; training; tuning grid |
| **04** | [Evaluation protocol](04_evaluation_protocol.md) | One decision rule for all models, LOPO with nested tuning, metrics, statistics, ablations |
| **05** | [Benchmark results](05_benchmark_results.md) | Fixed split, six-split variance, LOPO (primary), robustness checks, interpretation |
| **06** | [Pathologist labels, external cohort, survival](06_pathologist_external_survival.md) | UKF within-cohort and transfer, Greenwald, TCGA, the slide-level head |
| **07** | [References](07_references_and_citations.md) | Verified references from the manuscript bibliography |
| — | [WORKFLOW.md](WORKFLOW.md) | Script-by-script reproduction, result-file layout, and the conventions that silently break re-analysis |

Every number in these chapters comes from manuscript v7.1 (`preprint/spatial-niche-gbm.tex`).
`scripts/28_verify_manuscript_numbers.py` and `scripts/29_code_audit_checks.py` re-derive them
from `results/`.

## Glossary

- **Visium spot:** a ~55 µm capture area on a 10x Visium slide that measures RNA from roughly 1–10
  cells at a known position.
- **Niche:** a recurring tissue region with distinct biology. Here: immune-cold, immune-hot,
  normal or necrotic (SNUH); cellular, infiltrative, necrotic, MVP or normal (UKF).
- **Silver-standard label:** a label derived by rules from other annotations (here, expression-based
  ones) rather than from an independent expert.
- **Graph neural network (GNN):** a model that updates each node (spot) from its neighbours.
  GraphSAGE averages neighbours; GAT weights them with learned attention.
- **MLP:** a neural network applied to each spot independently, with no graph.
- **Homophily:** the fraction of graph edges joining two nodes with the same label.
- **Leave-one-patient-out (LOPO):** train on all patients but one, test on that one, repeat for each.
- **Nested tuning:** choosing hyperparameters on validation patients inside each fold, never on the
  test patient.
- **Macro-F1:** the mean of per-class F1 scores, so rare classes count as much as common ones.
- **Patient bootstrap:** a confidence interval obtained by resampling patients, not spots, with
  replacement.
- **Data leakage:** any path by which test information reaches training or tuning.
