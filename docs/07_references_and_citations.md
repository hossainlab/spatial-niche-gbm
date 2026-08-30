# Chapter 7: Scientific Evidence, References & Citations

Research is grounded in verifiable evidence. This document provides the complete, authenticated bibliography of peer-reviewed literature, open-access datasets, and foundational algorithmic frameworks supporting **DeepGraph-GBM**.

Every citation below includes official Digital Object Identifiers (DOIs), PubMed IDs (PMIDs), and accession repositories.

---

## 1. Primary Glioblastoma Spatial Transcriptomics Atlas

### [Park & Shah et al., Nature Communications 2026]
- **Full Citation**: Shah, N., Park, S., Kang, H., et al. *A spatially resolved human glioblastoma atlas reveals distinct cellular and molecular patterns of anatomical niches.* **Nature Communications**, 17, 2026.
- **DOI**: [10.1038/s41467-026-69716-2](https://doi.org/10.1038/s41467-026-69716-2)
- **Data Accession**: NCBI BioProject [PRJNA1337938](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1337938) | UCSC Cell Browser: [cells.ucsc.edu/?ds=multiomic-gbm](https://cells.ucsc.edu/?ds=multiomic-gbm)
- **Relevance to Project**: Provides the primary 23 spatial transcriptomics sections (34,988 spots) across 13 glioblastoma patients, anatomical feature annotations, and single-cell metaprogram signatures used for training and testing.

---

## 2. Independent External Spatial Cohort

### [Greenwald et al., Cell 2024]
- **Full Citation**: Greenwald, A. C., Galili Darnell, N., Hoefflin, R., Simkin, D., Mount, C. W., et al. *Integrative spatial analysis reveals a multi-layered organization of glioblastoma.* **Cell**, 187(10): 2485–2501.e26, 2024.
- **DOI**: [10.1016/j.cell.2024.03.029](https://doi.org/10.1016/j.cell.2024.03.029)
- **PMID**: [38657613](https://pubmed.ncbi.nlm.nih.gov/38657613/)
- **Data Repository**: Zenodo Open Data [12624108](https://zenodo.org/records/12624108) (CC-BY 4.0)
- **Relevance to Project**: Serves as our independent external validation cohort (13 Visium samples) to prove cross-hospital and cross-protocol generalizability.

---

## 3. Glioblastoma Single-Cell Cellular States & MES Metaprograms

### [Neftel et al., Cell 2019]
- **Full Citation**: Neftel, C., Laffy, J., Filbin, M. G., Hara, T., Shore, M. E., et al. *An Integrative Model of Cellular States, Plasticity, and Genetics for Glioblastoma.* **Cell**, 178(4): 835–849.e21, 2019.
- **DOI**: [10.1016/j.cell.2019.06.024](https://doi.org/10.1016/j.cell.2019.06.024)
- **PMID**: [31327527](https://pubmed.ncbi.nlm.nih.gov/31327527/) | PMCID: PMC6703903
- **Relevance to Project**: Defines the four core malignant glioblastoma states (MES, NPC, OPC, AC). DeepGraph-GBM's Mesenchymal (MES) regression head is evaluated against Neftel's gold-standard MES meta-module signature ($r = 0.30$).

---

## 4. Clinical Glioblastoma Cohort & Survival Outcomes

### [Brennan et al., Cell 2013 (TCGA Network)]
- **Full Citation**: Brennan, C. W., Verhaak, R. G. W., McKenna, A., Campos, B., Noushmehr, H., et al. (The Cancer Genome Atlas Research Network). *The Somatic Genomic Landscape of Glioblastoma.* **Cell**, 155(2): 462–477, 2013.
- **DOI**: [10.1016/j.cell.2013.09.034](https://doi.org/10.1016/j.cell.2013.09.034)
- **PMID**: [24120142](https://pubmed.ncbi.nlm.nih.gov/24120142/) | PMCID: PMC3910500
- **Data Portal**: NCI Genomic Data Commons (GDC) [TCGA-GBM Project](https://portal.gdc.cancer.gov/projects/TCGA-GBM)
- **Relevance to Project**: Provides the bulk RNA sequencing and clinical survival follow-up ($n=166$ patients, 132 events) used for Kaplan-Meier and Cox proportional hazards survival testing.

### [Stupp et al., NEJM 2005]
- **Full Citation**: Stupp, R., Mason, W. P., van den Bent, M. J., Weller, M., Fisher, B., et al. *Radiotherapy plus concomitant and adjuvant temozolomide for glioblastoma.* **New England Journal of Medicine**, 352(10): 987–996, 2005.
- **DOI**: [10.1056/NEJMoa043330](https://doi.org/10.1056/NEJMoa043330)
- **PMID**: [15758009](https://pubmed.ncbi.nlm.nih.gov/15758009/)
- **Relevance to Project**: Establishes the clinical baseline for glioblastoma prognosis, standard-of-care temozolomide chemoradiation, and the 12–15 month median survival benchmark.

---

## 5. Spatial Transcriptomics Technology

### [Ståhl et al., Science 2016]
- **Full Citation**: Ståhl, P. L., Salmén, F., Vickovic, S., Lundmark, A., Navarro, J. F., et al. *Visualization and analysis of gene expression in tissue sections by spatial transcriptomics.* **Science**, 353(6294): 78–82, 2016.
- **DOI**: [10.1126/science.aaf2403](https://doi.org/10.1126/science.aaf2403)
- **PMID**: [27365449](https://pubmed.ncbi.nlm.nih.gov/27365449/)
- **Relevance to Project**: The foundational paper introducing spatial transcriptomics arrays, the precursor technology commercialized as 10x Genomics Visium.

---

## 6. Graph Neural Networks & Deep Learning Frameworks

### [Hamilton, Ying, & Leskovec, NeurIPS 2017 (GraphSAGE)]
- **Full Citation**: Hamilton, W. L., Ying, R., & Leskovec, J. *Inductive Representation Learning on Large Graphs.* **Advances in Neural Information Processing Systems (NeurIPS)**, 30: 1024–1034, 2017.
- **ArXiv**: [arXiv:1706.02216](https://arxiv.org/abs/1706.02216)
- **Relevance to Project**: Defines the GraphSAGE spatial message passing and mean neighborhood aggregation architecture used as our primary graph encoder backbone.

### [Veličković et al., ICLR 2018 (GAT)]
- **Full Citation**: Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. *Graph Attention Networks.* **International Conference on Learning Representations (ICLR)**, 2018.
- **ArXiv**: [arXiv:1710.10903](https://arxiv.org/abs/1710.10903)
- **Relevance to Project**: Defines the multi-head self-attention graph encoder mechanism implemented in `src/deepgraph_gbm/models/encoder.py`.

### [Fey & Lenssen, ICLR Workshop 2019 (PyTorch Geometric)]
- **Full Citation**: Fey, M., & Lenssen, J. E. *Fast Graph Representation Learning with PyTorch Geometric.* **ICLR Workshop on Representation Learning on Graphs and Manifolds**, 2019.
- **ArXiv**: [arXiv:1903.02428](https://arxiv.org/abs/1903.02428)
- **Relevance to Project**: Provides the graph data structure (`torch_geometric.data.Data`), sparse edge indexing, and neural message passing layers.

---

## 7. Bioinformatic Feature Selection & Single-Cell Analysis

### [Stuart et al., Cell 2019 (Seurat v3 HVG)]
- **Full Citation**: Stuart, T., Butler, A., Hoffman, P., Hafemeister, C., Papalexi, E., et al. *Comprehensive Integration of Single-Cell Data.* **Cell**, 177(7): 1888–1902.e21, 2019.
- **DOI**: [10.1016/j.cell.2019.05.031](https://doi.org/10.1016/j.cell.2019.05.031)
- **PMID**: [31178118](https://pubmed.ncbi.nlm.nih.gov/31178118/)
- **Relevance to Project**: Defines the variance-stabilizing transformation and dispersion-based selection used to pick the top 3,000 Highly Variable Genes (HVGs).

### [Wolf, Angerer, & Theis, Genome Biology 2018 (Scanpy)]
- **Full Citation**: Wolf, F. A., Angerer, P., & Theis, F. J. *SCANPY: large-scale single-cell gene expression data analysis.* **Genome Biology**, 19(1): 15, 2018.
- **DOI**: [10.1186/s13059-017-1382-0](https://doi.org/10.1186/s13059-017-1382-0)
- **PMID**: [29409532](https://pubmed.ncbi.nlm.nih.gov/29409532/)
- **Relevance to Project**: Used for preprocessing, normalization, and expm1 dispersion modeling of Visium count matrices.
