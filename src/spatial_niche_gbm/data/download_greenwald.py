"""Download Greenwald et al. (Cell 2024) GBM Visium samples from Zenodo.

Record: https://zenodo.org/records/12624108 (CC-BY-4.0)
Each GBM tarball contains SpaceRanger outputs (filtered_feature_bc_matrix.h5,
tissue_positions_list.csv, scalefactors_json.json, images).
"""

from __future__ import annotations

import logging
import tarfile
import urllib.request
from pathlib import Path

log = logging.getLogger("spatial_niche_gbm")

ZENODO_API = "https://zenodo.org/api/records/12624108"

GBM_SAMPLES = [
    "GBM_MGH258.tar.gz",
    "GBM_ZH881_inf.tar.gz",
    "GBM_ZH881T1.tar.gz",
    "GBM_ZH8811Abulk.tar.gz",
    "GBM_ZH8811Bbulk.tar.gz",
    "GBM_ZH8812bulk.tar.gz",
    "GBM_ZH916bulk.tar.gz",
    "GBM_ZH916inf.tar.gz",
    "GBM_ZH916T1.tar.gz",
    "GBM_ZH1007inf.tar.gz",
    "GBM_ZH1007nec.tar.gz",
    "GBM_ZH1019inf.tar.gz",
    "GBM_ZH1019T1.tar.gz",
]


def _file_urls() -> dict[str, str]:
    import json

    with urllib.request.urlopen(ZENODO_API, timeout=60) as r:
        rec = json.load(r)
    return {f["key"]: f["links"]["self"] for f in rec["files"]}


def download_greenwald(out_dir: str | Path, samples: list[str] | None = None) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    urls = _file_urls()
    samples = samples or GBM_SAMPLES
    extracted = []
    for s in samples:
        if s not in urls:
            log.warning("sample %s not found in Zenodo record, skipping", s)
            continue
        dest = out_dir / s
        sample_dir = out_dir / s.replace(".tar.gz", "")
        if sample_dir.exists() and any(sample_dir.iterdir()):
            log.info("exists, skipping: %s", sample_dir)
            extracted.append(sample_dir)
            continue
        log.info("downloading %s", s)
        urllib.request.urlretrieve(urls[s], dest)
        sample_dir.mkdir(exist_ok=True)
        with tarfile.open(dest) as tf:
            tf.extractall(sample_dir, filter="data")
        dest.unlink()  # remove tarball after extraction
        extracted.append(sample_dir)
        log.info("extracted -> %s", sample_dir)
    return extracted


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    download_greenwald(sys.argv[1] if len(sys.argv) > 1 else "data/greenwald")
