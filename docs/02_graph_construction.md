# Chapter 2: Spatial graphs, homophily and patient splits

## 1. A section as a graph

Each Visium section becomes one graph:
- **nodes** are spots, each carrying its 3,000-gene feature vector;
- **edges** join each spot to its k nearest spots by coordinate.

Edges are symmetrised, so a spot can end up with more than k neighbours. The builder is
[`src/spatial_niche_gbm/data/graph.py`](../src/spatial_niche_gbm/data/graph.py) (`build_knn_graph`), and
the default is k = 6, because an interior spot on a hexagonal grid has six immediate neighbours.

Gaussian edge weights exp(−d² / 2·median(d)²) are computed and stored, **but neither encoder uses
them**. GraphSAGE and GAT aggregate over the unweighted neighbour set.

## 2. The array-index caveat, and the exact-hex sensitivity analysis

The SNUH atlas supplies spot positions as **Visium array indices**, in which the column index is
doubled. On that grid, four of the six true neighbours are at distance √2 and the other two at
distance 2. The latter tie with two second-ring spots, so **about 24% of 6-NN edges join
second-ring rather than adjacent spots**. Every main analysis uses these graphs, as the manuscript
states in its Methods.

To check whether the geometry matters, the graphs were rebuilt on true hexagonal coordinates
(column × ½, row × √3/2) with everything else unchanged:

```bash
python scripts/23a_build_hex_graphs.py      # data/preprocessed -> data/preprocessed_hex
```

The final GraphSAGE and GAT LOPO runs were repeated on these graphs (78 runs). Exact geometry did
not help: GraphSAGE 0.651 vs 0.652, GAT 0.611 vs 0.649
([Chapter 5 §4](05_benchmark_results.md)). External cohorts (Greenwald, UKF) are built on their
true hexagonal layouts.

## 3. Are the labels spatially clustered? (homophily)

A common reason a GNN fails is **heterophily**: neighbours carrying different labels, so that
averaging them mixes classes ([Zhu et al., *NeurIPS* 2020](https://arxiv.org/abs/2006.11468)).
That is not the case here. The edge homophily of the niche labels on 6-NN graphs is **0.80**.
Per-class homophily is above the level expected after shuffling labels within sections:

| Class | Same-label neighbour fraction | Chance |
|---|---|---|
| immune-cold | 0.82 | 0.67 |
| immune-hot | 0.67 | 0.43 |
| normal | 0.86 | 0.74 |
| necrotic | 0.73 | 0.50 |

Per-patient homophily ranges from 0.58 (SNU23) to 1.00 (SNU46). In principle, then, the graph is
informative about the labels, which makes the null result more informative rather than less.
These values are computed by `scripts/16_label_homophily.py` and stored in
`results/revision/label_homophily.json`.

## 4. Patient-level splitting

Spots from one patient must never appear on both sides of a train/test boundary. Neighbouring
spots are nearly identical, so a random spot-level split lets a model memorise the slide and
report inflated accuracy. Every analysis here splits by **patient**.

Three split schemes are used, from weakest to strongest:

| Scheme | Design | Used for |
|---|---|---|
| **Primary split** (`configs/splits.json`) | train SNU17, 18, 21, 23, 25, 27, 43, 46; validation SNU24, 51; test SNU16, 33, 34 | Fixed-split models, Figures 1–3, Table 1 |
| **Six splits** (`configs/splits_multisplit.json`) | split 0 = primary split; splits 1–5 from generator seeds 1001–1005, same 8/2/3 design | Variance decomposition only (see below) |
| **Leave-one-patient-out** (`results/revision/folds.json`) | 13 folds, one held-out patient each; two validation patients per fold chosen from label counts only | **Primary analysis** |

Two cautions about the six splits:
- Their test sets differ but overlap. Six patients are tested twice and SNU43 never.
- Each split tunes its thresholds on its own two validation patients. Four of the six tuning blocks
  hold fewer than 20 immune-hot spots, and split 4 holds a single one.

For those reasons the six-split benchmark is **not used to rank models**.

In the LOPO folds, validation pairs are chosen without looking at model performance. Among pairs
with at least 100 labelled spots of every class, the pair used least often so far is taken, so
validation duty is spread across patients. Every LOPO tuning block holds at least 183 spots of
every class.

---

## Next

[Chapter 3: The models being compared](03_models_compared.md)
