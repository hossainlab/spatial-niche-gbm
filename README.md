# spatial-niche-gbm

**Comparing graph-based and graph-free models for spatial niche prediction in glioblastoma
Visium data.**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7%2B-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.8-3c8dbc.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![Graphical abstract](assets/graphical_abstract.png)

## Overview

Visium spots sit next to each other, so graph neural networks are a natural choice for predicting
tumour niches from spatial transcriptomics. This project tests whether the spatial graph actually
helps. Two graph models (GraphSAGE, GAT) are compared with three graph-free models (logistic
regression, a multilayer perceptron, and a multilayer perceptron given neighbour-averaged
expression). All five get the same features, training, tuning budget and decision rule, and are
always tested on patients they have never seen.

## Key results

Leave-one-patient-out evaluation, 13 patients, 3 seeds:

| Model | Macro-F1 |
|---|---|
| **Logistic regression** | **0.691** |
| MLP | 0.655 |
| GraphSAGE | 0.652 |
| GAT | 0.649 |
| MLP + neighbours | 0.637 |

- Per-spot expression was enough: no graph model beat the best graph-free model.
- The same held against neuropathologist H&E annotations in an independent cohort of 18 patients.
- Which patient was held out mattered far more than which model was used.

## Getting started

```bash
git clone https://github.com/hossainlab/spatial-niche-gbm && cd spatial-niche-gbm
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
PYTHONPATH=src .venv/bin/pytest -q tests        # quick check, no data needed
```

To reproduce the results, follow [`docs/WORKFLOW.md`](docs/WORKFLOW.md). For a plain-language
introduction, start with [`docs/`](docs/README.md).

## Data

All datasets are public:

| Cohort | Role |
|---|---|
| SNUH glioblastoma spatial atlas (13 patients) | training and main comparison |
| Freiburg (UKF) Visium cohort (18 patients) | neuropathologist-annotated validation |
| Greenwald et al. 2024 (13 samples) | external cohort |
| TCGA-GBM (166 samples) | survival analysis |

## Citation

Hossain MJ. *Comparing graph-based and graph-free models for spatial niche prediction in
glioblastoma.* Preprint, 2026. See [`CITATION.cff`](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).
