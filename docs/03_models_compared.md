# Chapter 3: The models being compared

The benchmark compares two models that use the spatial graph with three that do not. Every model
sees the same 3,000-gene input per spot. The only intended difference is whether, and how, a spot's
neighbourhood enters the prediction.

## 1. Graph-free models

**Logistic regression.** A multinomial linear classifier (scikit-learn, L-BFGS, balanced class
weights) on each spot's expression alone. It is deterministic, so it has no seed variance. In the
LOPO benchmark its inverse regularisation strength C is tuned over eight values from 10⁻³ to 3.
It chose strong regularisation (C ≤ 0.003) in 12 of 13 folds.

**MLP.** A multilayer perceptron with the same layer widths, heads and training loop as the graph
encoders, but no edges. This is the cleanest control for message passing: remove the graph and
keep everything else.

**MLP + neighbours.** The MLP's input is the spot's expression concatenated with the
row-normalised mean expression of its k nearest neighbours over one or two hops. It sees the same
neighbourhood information as a message-passing layer, but combines it without learned aggregation.
It separates "access to neighbours" from "learned message passing".

## 2. Graph models

**GraphSAGE** ([Hamilton et al., *NeurIPS* 2017](https://arxiv.org/abs/1706.02216)). Three
`SAGEConv` layers with mean aggregation (3000 → 256 → 256 → 128). Each layer is followed by batch
normalisation, ReLU and dropout (p = 0.3). Built in PyTorch Geometric
([Fey & Lenssen 2019](https://arxiv.org/abs/1903.02428)); 1,751,815 parameters including heads.

**GAT** ([Veličković et al., *ICLR* 2018](https://arxiv.org/abs/1710.10903)). Three `GATConv`
layers with four attention heads and the same dimensions; 985,863 parameters. `GATConv` also
applies dropout to its attention coefficients, a mild extra regulariser that no other model has.
The paper discloses this as limitation (ix).

Code: [`src/spatial_niche_gbm/models/encoder.py`](../src/spatial_niche_gbm/models/encoder.py) and
[`baselines.py`](../src/spatial_niche_gbm/models/baselines.py). The LOPO benchmark builds its models
in [`scripts/15_revision_benchmark.py`](../scripts/15_revision_benchmark.py).

## 3. Heads and the multi-task model

The encoders feed task heads ([`heads.py`](../src/spatial_niche_gbm/models/heads.py),
[`multitask.py`](../src/spatial_niche_gbm/models/multitask.py)):

1. a **niche classifier** (four logits);
2. an **MES head** (sigmoid; binary mesenchymal-state target);
3. a **slide-level risk head** (attention pooling over spots, then an MLP, trained on a
   rule-derived "pseudo-risk" of niche composition).

Which heads are used depends on the analysis:

| Analysis | Heads | Loss |
|---|---|---|
| Fixed-split models (Figures 1–3, Table 1) | niche + MES + slide risk | 1.0·CE(niche) + 0.5·BCE(MES) + 0.5·MSE(risk) |
| LOPO benchmark and UKF experiments | niche + MES only | class-weighted CE(niche) + 0.5·BCE(MES) |

**The slide-level head does not work.** On the five held-out sections of the primary split, the
correlation between predicted and target slide risk was r = 0.40, −0.80 and 0.08 for the three
seeds. No result depends on it, and the survival analysis uses a signature score, not this head.

## 4. Training

All neural models share one training loop:
- **optimiser:** AdamW (learning rate 10⁻³, weight decay 10⁻⁴) with cosine annealing;
- **batching:** one section per step (full-graph batches);
- **class weights:** balanced, then boosted ×3.0 for immune-hot and ×1.5 for necrotic;
- **early stopping:** on validation macro-F1, restoring the best state.

| Setting | Fixed split | LOPO / UKF |
|---|---|---|
| Max epochs | 200 | 120 |
| Early-stopping patience | 40 | 25 |
| Hyperparameters | defaults | LOPO: nested search (Chapter 4); UKF: defaults |
| Seeds | 42, 7, 123 | 42, 7, 123 |

In the UKF experiments, class weights are balanced without the extra boosting.

Hyperparameter grid for the LOPO search:

| Parameter | Values |
|---|---|
| Learning rate | 3×10⁻⁴, 10⁻³, 3×10⁻³ |
| Dropout | 0.1, 0.3, 0.5 |
| Hidden width | 128, 256, 512 |
| Weight decay | 10⁻⁵, 10⁻⁴, 10⁻³ |
| Depth | 1, 2, 3 |
| k (graph models, MLP + neighbours) | 6, 12, 18 |
| Hops (MLP + neighbours) | 1, 2 |

Each family draws eight configurations once (generator seed 2026). In the selected
configurations, GraphSAGE used k = 12 or 18 in all 13 folds and two or three layers in 12. When
tuned, the graph models therefore chose *wider* neighbourhoods than the default, and still did not
win.

---

## Next

[Chapter 4: The evaluation protocol](04_evaluation_protocol.md)
