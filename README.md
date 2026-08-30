# DeepGraph-GBM

**Graph Neural Network for predicting immunosuppressive spatial niches in glioblastoma from 10x Visium spatial transcriptomics.**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.7](https://img.shields.io/badge/PyTorch-2.7-ee4c2c.svg)](https://pytorch.org/)
[![PyG 2.8](https://img.shields.io/badge/PyG-2.8-3c8dbc.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Graphical Abstract

![DeepGraph-GBM Graphical Abstract](assets/graphical_abstract.png)

*Figure: Overview of DeepGraph-GBM spatial graph construction from 10x Visium hexagonal coordinates, multi-task graph encoder architecture (GraphSAGE / GAT), spatial cellular niche classification, mesenchymal (MES) state regression, and cross-cohort external validation.*

---

## The Problem

Glioblastoma (GBM) is marked by extreme spatial heterogeneity. Within a single tumor, distinct cellular microenvironments govern tumor progression, treatment resistance, and immunosuppression:

| Niche | Cellular & Anatomical Composition | Clinical Prognosis |
|---|---|---|
| **Immune Cold** | Mesenchymal tumor cells, immunosuppressive myeloid cells, absent T cells | Poorest outcomes |
| **Immune Hot** | Infiltrating T lymphocytes, activated microglia/macrophages | Relatively improved (rare in GBM, ~9% of spots) |
| **Normal Brain** | Peritumoral non-neoplastic cortex and white matter | Baseline reference |
| **Necrotic Core** | Ischemic necrosis and perinecrotic pseudopalisading zones | Poor (hypoxic transition zone) |

DeepGraph-GBM models 10x Visium tissue slides as **spatial graphs** (spots as nodes with 3,000 highly variable gene expressions, physical neighbors as edges weighted by Euclidean distance decay) to predict spatial niches, continuous mesenchymal tumor states, and slide-level hazard scores.

---

## Interactive Exploration Notebooks

A sequential suite of 6 documented Jupyter notebooks is provided in [`notebooks/`](notebooks/):

| Notebook | Focus Area | Highlights & Outputs |
|---|---|---|
| [**`01_data_exploration.ipynb`**](notebooks/01_data_exploration.ipynb) | Dataset & Label Rules | SNUH atlas metadata (23 sections, 13 patients, 34,988 spots), niche priority rules, 3,000 HVG marker audit (*EGFR*, *VEGFA*, *CHI3L1*), MES state distribution. |
| [**`02_spatial_graphs_and_splits.ipynb`**](notebooks/02_spatial_graphs_and_splits.ipynb) | Spatial Graphs & Splits | $k=6$ kNN graph construction, Gaussian edge weights, PyG `Data` structures, patient-level train/val/test splits, 6-split benchmark design. |
| [**`03_model_architecture_and_training.ipynb`**](notebooks/03_model_architecture_and_training.ipynb) | Model & Loss Dynamics | GraphSAGE vs GAT encoder comparisons (~870k params), real section forward pass, multi-task loss decomposition with rare-class boosting, loss convergence curves. |
| [**`04_model_evaluation_and_inference.ipynb`**](notebooks/04_model_evaluation_and_inference.ipynb) | Evaluation & Spatial Maps | Checkpoint loading (`best_model.pt`), validation threshold tuning ($\tau_c$), test metrics, confusion matrix, true vs predicted spatial niche & MES maps. |
| [**`05_external_and_survival_validation.ipynb`**](notebooks/05_external_and_survival_validation.ipynb) | External & Survival Analysis | 13 Greenwald Visium samples (*Cell* 2024), Neftel MES correlation ($r = 0.30$), TCGA-GBM ($n=166$) Kaplan-Meier survival curves & Cox regression ($p = 0.25$). |
| [**`06_benchmarking_and_protocol_reversal.ipynb`**](notebooks/06_benchmarking_and_protocol_reversal.ipynb) | Benchmarks & Protocol Reversal | Table 1 matched baseline comparisons, paired t-tests, Protocol Reversal phenomenon (Fig. 4), 72-run multi-split variance decomposition (Fig. 5). |

---

## Model Architecture

```
Input: 10x Visium slide represented as a spatial graph
  Nodes: Spatial spots, features = 3,000 Seurat-v3 highly variable genes (log-normalized)
  Edges: Symmetric k-nearest neighbors (k = 6), Gaussian distance-weighted
        │
        ▼
Shared Graph Encoder (GraphSAGE / GAT: 3000 → 256 → 256 → 128)
  Produces 128-dimensional neighborhood-aware spot representations (z_i)
        │
        ├─► Head 1: Spatial Niche Classifier (4 classes per spot, Cross-Entropy)
        ├─► Head 2: Mesenchymal State Regressor (0–1 continuous score, BCE)
        └─► Head 3: Slide-Level Survival Risk (Global mean pooling, MSE)
```

**Multi-Task Objective**:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{niche}} \mathcal{L}_{\text{CE}}(\hat{y}_{\text{niche}}, y) + \lambda_{\text{mes}} \mathcal{L}_{\text{BCE}}(\hat{y}_{\text{mes}}, y_{\text{mes}}) + \lambda_{\text{surv}} \mathcal{L}_{\text{MSE}}(\hat{r}, r)$$

To mitigate severe class imbalance (`immune_hot` is ~9% of spots), loss weights are boosted for minority classes (`immune_hot` $\times 3.0$, `necrotic` $\times 1.5$).

---

## Datasets

| Dataset | Role | Content |
|---|---|---|
| **SNUH 2026 Atlas** ([Nat Commun 2026](https://www.nature.com/articles/s41467-026-69716-2), PRJNA1337938) | **Train / Val / Test** | 23 sections, 13 IDH-wildtype GBM patients, 34,988 spots with anatomical features and single-cell metaprogram annotations. |
| **Greenwald et al.** ([Cell 2024](https://www.cell.com/cell/fulltext/S0092-8674(24)00320-9), [Zenodo](https://zenodo.org/records/12624108)) | **External Validation** | 13 glioblastoma Visium sections, diverse tumor regions, Neftel single-cell meta-modules. |
| **TCGA-GBM** (NCI GDC) | **Clinical Survival** | Bulk RNA-seq + overall survival follow-up ($n=166$ patients, 132 events). |

**Niche Label Definition**: Derived by audited rules in [`configs/labels.yaml`](configs/labels.yaml):
- `necrotic`: Perinecrotic and pseudopalisading zones.
- `normal`: Non-neoplastic cortical grey matter and white matter.
- `immune_hot`: Microglia / macrophage / myeloid infiltration signatures.
- `immune_cold`: Dense malignant cellular tumor core.
- Ambiguous / discordant boundary spots are marked `exclude` and masked out during loss calculation.

---

## Quickstart

### 1. Environment Installation
```bash
# Clone the repository
git clone https://github.com/hossainlab/deepgraph-gbm.git
cd deepgraph-gbm

# Install dependencies using pip or uv
pip install -e .
```

### 2. End-to-End Pipeline
```bash
# Download datasets (~4.0 GB)
bash scripts/01_download_data.sh data

# Build spatial kNN graphs and preprocessed bundles
python scripts/02_build_graphs.py --data-dir data/raw --out-dir data/preprocessed

# Train the multi-task model
python scripts/03_train.py --config configs/default.yaml

# Evaluate on held-out test patients with validation threshold tuning
python scripts/04_evaluate.py --data-dir data/preprocessed --ckpt results/models/best_model.pt

# Run external cohort validation (Greenwald et al.)
python scripts/05_external_validation.py --greenwald-dir data/raw/greenwald \
    --ckpt results/models/best_model.pt --hvg data/preprocessed/hvg_genes.json

# Run TCGA clinical overall survival validation
python scripts/06_survival_validation.py --tcga-dir data/raw/tcga

# Generate publication figures and summary tables
python scripts/07_make_figures.py
```

---

## Key Benchmarking Insights

### 1. Matched-Protocol Baseline Comparison (Table 1)
When non-graph baselines (MLP, Logistic Regression) are trained under identical conditions (matched early stopping) and scored with identical validation-tuned decision thresholds, single-spot gene expression contains sufficient signal to achieve competitive or superior performance:

| Architecture | Macro-F1 (Mean ± Std) | Weighted-F1 | Accuracy | MES AUROC |
|---|---|---|---|---|
| **MLP (No-Graph Baseline)** | **0.578 ± 0.021** | **0.793 ± 0.003** | **0.789 ± 0.004** | **0.826 ± 0.016** |
| **Logistic Regression** | 0.554 | 0.761 | 0.748 | 0.751 |
| **GAT (4-Head Attention)** | 0.539 ± 0.038 | 0.768 ± 0.020 | 0.762 ± 0.022 | 0.760 ± 0.027 |
| **GraphSAGE GNN** | 0.489 ± 0.027 | 0.746 ± 0.025 | 0.750 ± 0.050 | 0.751 ± 0.041 |

### 2. The Protocol Reversal Phenomenon
Under mismatched scoring protocols (GNN evaluated with tuned thresholds while baselines used default `argmax`), the GNN appeared superior ($0.489$ vs $0.449$). Scoring all models with identical validation-tuned thresholds **inverts the ranking** ($0.578$ for MLP vs $0.489$ for GNN).

### 3. Patient Variance Dominates Seed Variance (72 Runs)
Across 6 patient-level splits and 3 random seeds (72 trained models), across-split variance is **3.6–8.1× larger** than within-split seed variance. Robust conclusions in spatial biology necessitate repeated patient-level split resampling rather than single-split evaluations.

---

## Repository Layout

```
deepgraph-gbm/
├── assets/                  # Graphical abstract and documentation images
├── configs/                 # Hyperparameters, label rules, and patient splits
│   ├── default.yaml
│   ├── labels.yaml
│   ├── splits.json
│   └── splits_multisplit.json
├── data/                    # Raw and preprocessed dataset directories
├── notebooks/               # Sequential interactive exploration notebooks (01 to 06)
├── results/                 # Metrics tables, models, figures, and benchmark logs
├── scripts/                 # Execution pipeline scripts (01_download to 14_fig_protocol_reversal)
├── src/deepgraph_gbm/       # Core Python library
│   ├── data/                # Dataset loaders, kNN graph builders, label rules
│   ├── models/              # GraphSAGE, GAT, multi-task architecture, baselines
│   ├── training/            # Training loop, multi-task loss, evaluation metrics
│   └── interpret/           # Spatial map rendering and neighbor importance
└── tests/                   # Pytest test suite (labels, graphs, model shapes, splits)
```

---

## Limitations

- **Silver-Standard Labels**: Niche annotations are derived from algorithmic rule hierarchies on single-cell metaprograms and anatomical features rather than manual pathologist annotations.
- **Discovery-Level Survival Validation**: Spatial Visium datasets currently lack long-term clinical survival follow-up; the survival head is trained on a niche-composition pseudo-risk proxy and validated on bulk TCGA-GBM.
- **Hardware Efficiency**: Optimized for standard CPU and single GPU environments using PyTorch and PyG.

---

## Citation

If you use DeepGraph-GBM or its benchmarks in your research, please cite:

```bibtex
@article{park2026deepgraph,
  title={DeepGraph-GBM: A Spatially Resolved Multi-Task Graph Neural Network for Glioblastoma Microenvironment Modeling},
  author={Park, S. and Shah, N. and Collaborators},
  journal={Nature Communications},
  year={2026}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
