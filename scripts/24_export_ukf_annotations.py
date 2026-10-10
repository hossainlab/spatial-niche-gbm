"""Export the neuropathologist H&E region annotations of the UKF cohort as a
machine-readable per-spot table.

The annotations live in the `segmentation` field of the SPATA objects of the
Freiburg (UKF) cohort (Ravi et al. 2022) as annotated for Ritter et al. 2025.
`scripts/18_export_spata_objects.R` writes them to `<sample>/spots.csv`; this
script applies exactly the sample filtering, spot filtering and class mapping of
`scripts/19_build_histology_graphs.py` and writes one row per retained spot, so
that the labels used in the paper can be obtained without re-deriving them from
the source objects.

Usage: python scripts/24_export_ukf_annotations.py \
           data/raw/ukf_spata results/tables/supplement/TableS9_ukf_pathologist_annotations.csv
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

CLASSES = ["cellular", "infiltrative", "necrotic", "mvp", "normal"]
UKF_MAP = {"cellular": "cellular", "infiltrative": "infiltrative", "infiltartive": "infiltrative",
           "necrosis": "necrotic", "necrotic_edge": "necrotic", "vascular_hyper": "mvp",
           "grey_matter": "normal", "white_matter": "normal", "wite_matter": "normal"}
UKF_NOT_GBM = {"270_T", "268_T"}
MIN_GENES = 200


def main(in_dir: str, out_csv: str) -> None:
    ukf = Path(in_dir)
    rows = []
    dropped = []
    for sd in sorted(p for p in ukf.iterdir() if (p / "spots.csv").exists()):
        if sd.name in UKF_NOT_GBM:
            dropped.append((sd.name, "not GBM"))
            continue
        spots = pd.read_csv(sd / "spots.csv").set_index("barcodes")
        bcs = [l.strip() for l in open(sd / "barcodes.txt")]
        seg = spots.loc[bcs, "segmentation"].astype(str).str.strip().str.lower()
        cls = [UKF_MAP.get(s, "unannotated") for s in seg]
        if not any(c in CLASSES for c in cls):
            dropped.append((sd.name, "no pathologist annotation"))
            continue
        C = sp.csr_matrix(sio.mmread(sd / "counts.mtx")).T.tocsr()   # spots x genes
        n_genes = np.diff(C.indptr)
        keep = n_genes >= MIN_GENES
        xy = spots.loc[bcs, ["x", "y"]].values.astype(float)
        pid = re.match(r"#?(\d+)", sd.name).group(1)
        for i in np.flatnonzero(keep):
            rows.append((f"UKF{sd.name.lstrip('#')}", f"UKF{pid}", bcs[i],
                         xy[i, 0], xy[i, 1], int(n_genes[i]),
                         spots.loc[bcs[i], "segmentation"], cls[i]))
    df = pd.DataFrame(rows, columns=["section", "patient", "barcode", "x", "y",
                                     "n_genes_detected", "segmentation_raw", "class"])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    ann = df[df["class"].isin(CLASSES)]
    print(f"sections {df.section.nunique()}  patients {df.patient.nunique()}  "
          f"spots {len(df)}  annotated {len(ann)}")
    print(ann["class"].value_counts().reindex(CLASSES).to_string())
    for name, why in dropped:
        print(f"dropped {name}: {why}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
