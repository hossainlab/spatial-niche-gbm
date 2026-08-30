"""Download SNUH 2026 GBM atlas spatial data from the UCSC Cell Browser.

Source: https://cells.ucsc.edu/?ds=multiomic-gbm+spatial
Files: matrix.mtx.gz (27,573 genes x 115,914 spots, log1p-normalized values),
features.tsv.gz (gene symbols), meta.tsv (per-spot annotations incl. AF,
metaprogram, greenwald_metaprograms), Visium.coords.tsv.gz (array coordinates).
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

log = logging.getLogger("deepgraph_gbm")

BASE = "https://cells.ucsc.edu/multiomic-gbm/spatial"
FILES = [
    "matrix.mtx.gz",
    "features.tsv.gz",
    "meta.tsv",
    "Visium.coords.tsv.gz",
    "UMAP.coords.tsv.gz",
]


def download_snuh(out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fname in FILES:
        dest = out_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            log.info("exists, skipping: %s", dest)
            paths.append(dest)
            continue
        url = f"{BASE}/{fname}"
        log.info("downloading %s -> %s", url, dest)
        urllib.request.urlretrieve(url, dest)
        paths.append(dest)
    return paths


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    download_snuh(sys.argv[1] if len(sys.argv) > 1 else "data/snuh")
