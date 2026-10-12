# Chapter 1: Biology, data and labels

## 1. Gene expression and Visium in plain terms

Every cell carries the same DNA, but at any moment it copies only some genes into messenger RNA.
Counting those RNA copies per gene gives an **expression profile**: a readout of what the cell
is doing. **10x Visium** ([Ståhl et al., *Science* 2016](https://doi.org/10.1126/science.aaf2403))
places a tissue section on a slide covered with barcoded spots about 55 µm across, arranged on a
hexagonal grid. Each spot captures RNA from roughly 1–10 cells, so one section yields thousands of
expression profiles, each tied to an (x, y) position.

A Visium **spot is not a cell**. It already averages a small neighbourhood, and this matters for
the main result (see [Chapter 5 §5](05_benchmark_results.md)).

---

## 2. Glioblastoma niches

GBM tissue is organised into recurring regions with different biology. The Ivy Glioblastoma Atlas
defined anatomic features such as the leading edge, infiltrating tumour, cellular tumour,
pseudopalisading cells around necrosis and microvascular proliferation
([Puchalski et al., *Science* 2018](https://doi.org/10.1126/science.aaf2666)). Spatial
transcriptomics then showed niches shaped by hypoxia and immune composition
([Ravi et al., *Cancer Cell* 2022](https://doi.org/10.1016/j.ccell.2022.05.009);
[Greenwald et al., *Cell* 2024](https://doi.org/10.1016/j.cell.2024.03.029)).

This project predicts four niche classes for every spot of the SNUH cohort:

| Class | Meaning | Rule (from atlas annotations) |
|---|---|---|
| `necrotic` | necrosis and its hypoxic rim | anatomical feature PAN or PNZ |
| `immune_hot` | immune-infiltrated tissue | immune metaprogram, and anatomical feature not PAN/PNZ |
| `normal` | non-neoplastic grey or white matter at the leading edge | anatomical feature LE_GM/LE_WM **and** metaprogram LE_GM/LE_WM |
| `immune_cold` | tumour parenchyma without immune infiltration | anatomical feature CT, IT or IGN **and** metaprogram CT, IT or hypoxic |

Rules are applied in priority order (necrotic → immune-hot → normal → immune-cold; first match
wins). Spots matching no rule, including vascular features, are labelled `exclude` and masked from
training and scoring. The rules live in [`configs/labels.yaml`](../configs/labels.yaml) and
[`src/spatial_niche_gbm/data/labels.py`](../src/spatial_niche_gbm/data/labels.py), and the unit tests in
`tests/test_labels.py` check them.

Of 100,488 SNUH spots, **75,697 are labelled**; about 25% are excluded. The class balance differs
sharply between patients. The three test patients of the primary split contribute 16,377 labelled
spots, of which 67.9% are immune-cold and only 1.1% (185 spots) immune-hot.

### These are silver-standard labels

The labels are derived by rules from the atlas's own expression-based annotations. A model can
therefore score well by reproducing the labelling rule rather than tissue structure. That is why
the project also tests against **neuropathologist H&E annotations** in an independent cohort
([Chapter 6](06_pathologist_external_survival.md)).

### The MES target

A second, binary target marks whether a spot's annotated malignant state is mesenchymal-like
(`MES`, `MES.Hyp` or `MES.Ast`: 1; any other annotated state: 0; missing if unannotated). It is
**binary, not a continuous fraction**. Older documentation described it wrongly.

---

## 3. Features: 3,000 highly variable genes

Spots with fewer than 200 detected genes are removed. The SNUH matrix is stored as
log1p(counts scaled to 100,000 per spot), so counts are reconstructed with `expm1`. Then 3,000
highly variable genes are selected with scanpy's `seurat_v3` method
([Stuart et al., *Cell* 2019](https://doi.org/10.1016/j.cell.2019.05.031);
[Wolf et al., *Genome Biol* 2018](https://doi.org/10.1186/s13059-017-1382-0)). Dispersion-based
selection on the normalised matrix returned mostly housekeeping genes and hurt cross-cohort
transfer, so do not revert this step in `02_build_graphs.py`.

The panel is chosen once, without labels, on the eight training patients of the primary split, and
reused for every fold. Each held-out patient's expression therefore contributed to choosing the
panel, though its labels never did. This affects all models equally.

---

## 4. The four cohorts

| Cohort | Role | Size | Source |
|---|---|---|---|
| **SNUH spatial atlas** ([Park et al., *Nat Commun* 2026](https://doi.org/10.1038/s41467-026-69716-2)) | Training and the primary benchmark | 28 Visium sections (23 FFPE, 5 fresh-frozen), 13 IDH-wildtype GBM patients, 100,488 spots | UCSC Cell Browser `multiomic-gbm`; BioProject PRJNA1337938 |
| **UKF (Freiburg)** ([Ravi et al., *Cancer Cell* 2022](https://doi.org/10.1016/j.ccell.2022.05.009); annotations per [Ritter et al., *Nat Cancer* 2025](https://doi.org/10.1038/s43018-024-00904-z)) | Neuropathologist H&E labels, used within-cohort and as a transfer target | 23 sections (16 tumour, 7 cortex) from 18 patients; 25,374 of 68,015 spots annotated | Dryad [10.5061/dryad.h70rxwdmj](https://doi.org/10.5061/dryad.h70rxwdmj); Zenodo [10.5281/zenodo.14064047](https://doi.org/10.5281/zenodo.14064047) |
| **Greenwald et al.** ([*Cell* 2024](https://doi.org/10.1016/j.cell.2024.03.029)) | External cohort, no retraining | 13 GBM Visium samples, 25,488 spots, annotated by sampling region | Zenodo [10.5281/zenodo.12624108](https://doi.org/10.5281/zenodo.12624108) (CC-BY) |
| **TCGA-GBM** ([TCGA Network, *Nature* 2008](https://doi.org/10.1038/nature07385)) | Bulk survival test of a niche-signature score | 166 tumour samples from 165 patients, 132 deaths | NCI Genomic Data Commons |

For the UKF cohort, pathologist segments are mapped to five histology classes: cellular tumour,
infiltrative tumour, necrosis, microvascular proliferation (MVP) and normal grey/white matter.
Two non-GBM samples are dropped. The per-spot annotations used in the paper are released as
[`results/tables/supplement/TableS9_ukf_pathologist_annotations.csv`](../results/tables/supplement/TableS9_ukf_pathologist_annotations.csv).

---

## Next

[Chapter 2: Graph construction, homophily and patient splits](02_graph_construction.md)
