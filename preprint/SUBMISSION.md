# Submission checklist

Route: **bioRxiv preprint first, then journal.** Posting the preprint fixes a citable version
and a DOI that the journal submission can reference.

## State of the manuscript

| | |
|---|---|
| Title | *Spatial Graph Structure Does Not Improve Supervised Niche Prediction in Glioblastoma* (11 words) |
| Abstract | 353 words |
| Build | 40 pages, 53 references, 10 main + 7 supplementary figures, 3 main + 13 supplementary tables |
| LaTeX log | no undefined or multiply-defined references, no missing graphics, no overfull boxes |
| Figures | every figure ≥ 300 dpi |
| Verification | 35/35 manuscript numbers, 99/99 independent re-derivations, 18/18 unit tests |

## Ready

- [x] Data and code availability, with the repository URL and all four cohort sources
- [x] Ethics and data-use declaration (secondary analysis of published, de-identified, public data)
- [x] Funding statement (none)
- [x] Author contributions (sole author)
- [x] Competing interests (none)
- [x] Acknowledgments
- [x] Released per-spot pathologist annotations (Supplementary Table S9)
- [x] Repository committed, documented and reproducible without retraining

## Needs you before posting

1. **ORCID.** Not in the manuscript and not supplied. bioRxiv will ask for it, and most journals
   require it for the corresponding author. Add to the author block once you have the identifier.
2. **Push the repository.** Four commits are staged locally on `main`; `origin` is
   `https://github.com/hossainlab/deepgraph-gbm.git`, which matches the URL printed in the
   manuscript. Nothing is pushed yet — the data-availability statement is only true once it is.
   Note the repository is ~96 MB tracked; confirm that is acceptable before pushing.
3. **Confirm the licence of the released UKF annotation table.** Supplementary Table S9
   redistributes annotations derived from the Ravi et al. and Ritter et al. deposits. Check the
   terms of those deposits allow redistribution, or link rather than redistribute.
4. **bioRxiv category and licence.** Suggested category: *Bioinformatics* (alternatively
   *Cancer Biology*). Licence: CC-BY 4.0 is the usual choice and is compatible with the MIT code.

## Then, for the journal

- **Candidate venues**, in order of fit: *Genome Biology*, *Briefings in Bioinformatics*,
  *Bioinformatics*, *NAR Genomics and Bioinformatics*, *PLoS Computational Biology*. All accept
  benchmark and negative-result papers; all permit a 350-word or longer abstract except where
  noted below.
- **Abstract cap.** 353 words is fine for *Genome Biology* and *PLoS Comput Biol*; trim three
  words if the target enforces 350.
- **Figure files.** Journals usually want figures separately at 300–600 dpi (TIFF or EPS). The
  PNG sources are in `preprint/figures/`; only `Figure_split_robustness` currently has a vector
  version, so regenerate as SVG/EPS from the numbered scripts if vector artwork is required.
- **Reference style.** Currently `unsrtnat` numeric. Convert to the target journal's style.
- **Likely referee requests**, all already addressed in the text but worth having ready:
  the six-split threshold defect (disclosed, analysis demoted — a full repair needs 72 retrained
  models), the transductive gene-panel selection, and the GAT attention-dropout asymmetry.

## One open scientific item

The six-split threshold-tuning protocol is disclosed and demoted rather than repaired. The
leave-one-patient-out benchmark carries the conclusion without it. If a referee insists, the
repair is to re-tune thresholds on pooled out-of-fold predictions from patient-grouped
cross-validation inside each split's training pool and retrain all 72 models.
