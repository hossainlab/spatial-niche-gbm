# DeepGraph-GBM Documentation

Welcome to the complete, beginner-friendly documentation for **DeepGraph-GBM**!

This documentation is specifically written for readers who may not have a background in **biology**, **genomics**, or **deep learning**. It breaks down every concept—from cancer cells and DNA sequencing to graph algorithms and neural networks—using clear language, intuitive real-world analogies, visual diagrams, and rigorous scientific evidence.

---

## Documentation Navigation Guide

Follow the documentation sequentially from start to finish, or jump directly to the topic you are interested in:

| Chapter | Document | What You Will Learn |
|---|---|---|
| **00** | [**Overview & Project Roadmap**](00_overview_and_roadmap.md) | What glioblastoma is, why it is so aggressive, what spatial transcriptomics does, and how this AI system helps understand brain tumors. |
| **01** | [**Biology & Data Explained Simply**](01_biology_and_spatial_data.md) | The 4 cellular microenvironments ("niches"), gene expression as a cell's recipe book, highly variable genes, and the SNUH glioblastoma dataset. |
| **02** | [**Graph Construction & Patient Splits**](02_graph_construction.md) | How tissue slides turn into spatial graphs (honeycomb lattice), distance-weighted edges, and why patient-level data splitting prevents "cheating" (leakage). |
| **03** | [**Model Architecture & Deep Learning**](03_model_architecture_and_training.md) | How Graph Neural Networks work ("talking to neighbors"), GraphSAGE vs GAT, the multi-task neural network brain, and training dynamics. |
| **04** | [**Model Evaluation & Spatial Predictions**](04_evaluation_and_spatial_maps.md) | How we measure AI accuracy, why rare classes need threshold tuning, and how the AI paints spatial maps of brain tumor tissue. |
| **05** | [**External Cohorts & Survival Analysis**](05_external_validation_and_survival.md) | Testing the AI on data from different hospitals (Greenwald cohort), testing clinical patient survival (TCGA), and why honest negative findings matter. |
| **06** | [**Benchmarking & Protocol Reversal**](06_benchmarks_and_protocol_reversal.md) | Comparing Graph AI against simpler machine learning baselines, uncovering the "Protocol Reversal" effect, and analyzing 72 experimental runs. |
| **07** | [**Scientific Evidence & Citations**](07_references_and_citations.md) | The complete, peer-reviewed bibliography with DOIs, PMIDs, and data accession links supporting all claims in this project. |

---

## Scientific Grounding & Evidence

All computational choices and biological annotations in this project are anchored in published peer-reviewed studies:
- **Spatial Glioblastoma Atlas**: Shah, Park, et al., *Nature Communications* (2026), DOI: [10.1038/s41467-026-69716-2](https://doi.org/10.1038/s41467-026-69716-2).
- **External Spatial Validation**: Greenwald et al., *Cell* (2024), DOI: [10.1016/j.cell.2024.03.029](https://doi.org/10.1016/j.cell.2024.03.029).
- **Single-Cell Cellular States**: Neftel et al., *Cell* (2019), DOI: [10.1016/j.cell.2019.06.024](https://doi.org/10.1016/j.cell.2019.06.024).
- **Clinical Genomic Cohort**: TCGA Network (Brennan et al.), *Cell* (2013), DOI: [10.1016/j.cell.2013.09.034](https://doi.org/10.1016/j.cell.2013.09.034).
- **Graph Neural Network Foundations**: Hamilton et al., *NeurIPS* (2017), [arXiv:1706.02216](https://arxiv.org/abs/1706.02216); Veličković et al., *ICLR* (2018), [arXiv:1710.10903](https://arxiv.org/abs/1710.10903).

See [**Chapter 7: Scientific Evidence, References & Citations**](07_references_and_citations.md) for full details.

---

## Quick Plain-English Glossary

If you encounter technical terms anywhere in the repository, here is a quick cheat-sheet:

- **Glioblastoma (GBM)**: The most common and aggressive type of primary brain cancer in adults.
- **Microenvironment / Niche**: The local neighborhood of cells surrounding a tumor cell (including blood vessels, immune cells, and dead tissue).
- **10x Visium**: A laboratory technology that places a thin slice of tissue on a glass slide with thousands of microscopic spots, measuring thousands of active genes at each physical spot.
- **Spot**: A microscopic circle (~55 µm in diameter) on the Visium slide containing roughly 1 to 10 cells.
- **Gene Expression**: The count of how active different genes are inside the cells of a spot.
- **Graph**: A mathematical structure composed of **nodes** (points, here representing spots) connected by **edges** (lines, representing physical adjacency in the tissue).
- **Graph Neural Network (GNN)**: An AI model that allows each node to update its knowledge by gathering messages from its neighboring nodes.
- **Multi-Task Learning**: Training a single neural network to solve multiple related problems at once (e.g., finding the niche, predicting tumor state, and estimating risk).
- **Macro-F1 Score**: An accuracy score between 0.0 and 1.0 that treats rare and common categories with equal importance.
- **Data Leakage**: An evaluation flaw where information from the test set accidentally leaks into the training set, making an AI look deceptively accurate.
