#!/usr/bin/env bash
# Download the public datasets for spatial-niche-gbm into data/raw (the location the
# pipeline scripts expect). Usage: bash scripts/01_download_data.sh [data/raw]
#
# The UKF (Freiburg) neuropathologist-annotated cohort is not fetched here: download
# the SPATA objects from Dryad (doi:10.5061/dryad.h70rxwdmj) and Zenodo
# (doi:10.5281/zenodo.14064047), then export them with scripts/18_export_spata_objects.R
# into data/raw/ukf_spata (see docs/WORKFLOW.md, step B1).
set -euo pipefail
DATA_DIR=${1:-data/raw}

echo "=== SNUH 2026 GBM atlas (UCSC Cell Browser) ==="
python -m spatial_niche_gbm.data.download_snuh "$DATA_DIR/snuh"

echo "=== Greenwald et al. Cell 2024 (Zenodo) ==="
python -m spatial_niche_gbm.data.download_greenwald "$DATA_DIR/greenwald"

echo "=== TCGA-GBM expression + clinical (GDC) ==="
python -m spatial_niche_gbm.data.download_tcga "$DATA_DIR/tcga"

echo "All downloads complete under $DATA_DIR"
