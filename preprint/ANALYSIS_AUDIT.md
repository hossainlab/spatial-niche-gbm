# Analysis audit, 2026-10-09 (manuscript v5)

This audit reviewed every analysis script (02, 05, 11–23, `src/`) against the code, the result files and the manuscript text.

## Errors found and fixed (with reruns)

| # | Problem | Effect | Fix | Effect on conclusions |
|---|---|---|---|---|
| 1 | The "original protocol" baselines (Table S1: logistic 0.350, MLP 0.449) came from a superseded feature set. Proof: the deterministic logistic model's AUROCs (0.60/0.57) differ from those on the current features (0.77/0.70). | The "scoring rule reverses the ranking" section, Figure 4C, contribution 5 and the first Results sentence of the abstract were unsupported. | Reran the original protocol on the current data (`results/baselines_original_protocol_rerun`): logistic 0.541, MLP 0.618, both above GNN 0.489. One logistic fit scored by argmax vs tuned thresholds: 0.541 vs 0.557. Rewrote section 2.4 ("not reproducible"), Figure 4, Table S1, the abstract, contribution 5 and the Discussion. `14_fig_protocol_reversal.py` now pairs bars from the same fit. | The claim that the decision rule creates the GNN advantage was removed. The graph-free advantage holds under every protocol. |
| 2 | Greenwald external data was normalized to 1e4 counts/spot, but the SNUH training matrix is log1p(CP100k): the expm1 values of every spot sum to 1e5. | External predictions were made on mis-scaled inputs. | `05_external_validation.py --target-sum 1e5` (default) → `results/external_v2`; updated section 2.8, Table S2 and S4 files, and Methods. | Composition changed (77.8% immune-hot vs 56% necrotic before); immune-cold is still near-absent; MES r falls to 0.21. The "no transfer" conclusion holds. |
| 3 | Same scale mismatch in the SNUH→UKF transfer (UKF at CP10k). | E2 numbers were invalid. | `transfer_v2` dataset (exact log1p(10·expm1(x)) rescale); 16 jobs rerun (`results/revision_transfer_v2`); `19_build_histology_graphs.py` patched; `21` uses v2. | **The ranking changed:** GraphSAGE is now first after transfer (0.395 vs MLP 0.380, logistic 0.352). The lead is not significant vs the MLP (p = 0.25) or MLP + neighbors (p = 0.76); the pooled CI excludes 0 vs logistic, but the per-patient test does not (p = 0.30). The abstract, Results, Discussion and Conclusions were softened accordingly. |
| 4 | SNUH (and Greenwald) coordinates are Visium array indices, so the 6-NN graphs include about 24% second-ring edges. | Graph geometry is not exactly hexagonal. | Exact-hex graphs (`data/preprocessed_hex`); reran the 78 final GraphSAGE/GAT LOPO jobs (`results/revision_hex`, `scripts/23_hex_sensitivity.py`). | GraphSAGE 0.651 (vs 0.652), GAT 0.611 (vs 0.649); logistic still beats both (p = 0.002, p = 7e-4). Disclosed in Methods and Results. |
| 5 | Fixed-split tests paired GNN and MLP "by seed" (paired t, df = 2, p = 0.0017), but seeds are not shared units across models. | Inflated significance. | Welch t: p = 0.012 (CI 0.033–0.145); GAT vs GraphSAGE p = 0.30. Updated Table 1 caption, text, Discussion and Methods. | Same direction, weaker evidence. |

## Text errors fixed (no rerun needed)

- The MES target was described as a "fraction of MES states"; it is a binary indicator.
- "No graph model exceeded the best graph-free model on any summary" was false: GraphSAGE had the highest normal-class F1. The claim is now narrowed.
- SNU27 per-patient range corrected from 0.64–0.72 to 0.62–0.72.
- Contribution 1 and section 2.3 said logistic regression shared the optimizer and early stopping; it doesn't. Corrected.
- The multi-split test sets overlap (six patients are tested twice, SNU43 never), so the df = 5 tests are optimistic. Disclosed.
- HVGs were chosen (label-free) on the primary-split training patients but reused for all LOPO and multi-split folds; UKF HVGs were chosen on all UKF patients. Disclosed; this affects every model equally.
- Logistic L-BFGS hits 500 iterations at C ≥ 0.003. At C = 0.001 (9 of 13 folds) it converges; refitting to convergence changes held-out F1 by −0.012 and +0.011. Disclosed.
- TCGA: one patient contributes two aliquots. Deduplicated sensitivity: HR 1.13, p = 0.25. Disclosed.

## Verified correct (numbers match result files)

LOPO tables and tests, variance ratios, chosen configurations, homophily, ablations and marker removal, UKF E1 (all values and confusion fractions), multi-split statistics, fixed-split Table 1 values, the MLP argmax control, and the survival statistics. There is no test-label leakage: thresholds, early stopping, configuration selection and marker ranking use only training or validation patients.

## v6 note
Section 2.4, Figure 4 and Table S1 (finding 1) were later removed from the manuscript because the earlier analysis was never public; the corrected numbers are kept in results/tables/archive/ (moved to `results/archive/tables/` in v7.2).

## Still open

- The UKF data-source placeholder.
- The abstract is about 600 words.
- Fixed-split figures (Figures 1–3 and 8) use the original array-index graphs; they are qualitative.
