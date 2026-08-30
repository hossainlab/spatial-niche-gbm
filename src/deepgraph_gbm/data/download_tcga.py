"""Download TCGA-GBM gene expression (STAR counts) + clinical via the GDC API.

Used for external survival validation of the niche-composition risk score.
"""

from __future__ import annotations

import gzip
import json
import logging
import urllib.request
from pathlib import Path

import pandas as pd

log = logging.getLogger("deepgraph_gbm")

GDC_FILES = "https://api.gdc.cancer.gov/files"
GDC_DATA = "https://api.gdc.cancer.gov/data"
GDC_CASES = "https://api.gdc.cancer.gov/cases"


def _post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def query_gbm_star_counts(max_files: int = 200) -> list[dict]:
    """Find TCGA-GBM primary-tumor STAR gene-count files."""
    filt = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-GBM"}},
            {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
            {"op": "=", "content": {"field": "analysis.workflow_type", "value": "STAR - Counts"}},
            {"op": "=", "content": {"field": "cases.samples.sample_type", "value": "Primary Tumor"}},
        ],
    }
    payload = {
        "filters": filt,
        "fields": "file_id,file_name,cases.submitter_id,cases.samples.submitter_id",
        "format": "json",
        "size": str(max_files),
    }
    res = _post(GDC_FILES, payload)
    return res["data"]["hits"]


def download_expression(out_dir: str | Path, max_files: int = 200) -> Path:
    """Download STAR count files and build a gene x sample TPM-ish matrix.

    Saves tcga_gbm_expression.tsv.gz (genes x samples, TPM unstranded) and
    returns its path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hits = query_gbm_star_counts(max_files)
    log.info("found %d TCGA-GBM STAR-count files", len(hits))

    series = {}
    for h in hits:
        fid = h["file_id"]
        case = h["cases"][0]["submitter_id"]
        sample = h["cases"][0].get("samples", [{}])[0].get("submitter_id", case)
        cache = out_dir / f"{fid}.tsv"
        if not cache.exists():
            try:
                urllib.request.urlretrieve(f"{GDC_DATA}/{fid}", cache)
            except Exception as e:  # noqa: BLE001
                log.warning("failed %s: %s", fid, e)
                continue
        # GDC serves plain-text TSV (despite any .gz assumption); read generically
        opener = gzip.open if cache.read_bytes()[:2] == b"\x1f\x8b" else open
        with opener(cache, "rt") as f:
            df = pd.read_csv(f, sep="\t", comment="#", usecols=["gene_name", "tpm_unstranded"])
        s = df.drop_duplicates("gene_name").set_index("gene_name")["tpm_unstranded"]
        s.name = sample
        series[sample] = s
        cache.unlink()

    mat = pd.DataFrame(series)
    mat = mat[~mat.index.str.startswith("N_")]  # drop PAR_Y/novel placeholders
    out = out_dir / "tcga_gbm_expression.tsv.gz"
    mat.to_csv(out, sep="\t", compression="gzip")
    log.info("expression matrix %s -> %s", mat.shape, out)
    return out


def download_clinical(out_dir: str | Path) -> Path:
    """Fetch TCGA-GBM clinical (days_to_death/last_follow_up, age, MGMT, IDH)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filt = {"op": "=", "content": {"field": "project.project_id", "value": "TCGA-GBM"}}
    payload = {
        "filters": filt,
        # NOTE: days_to_last_follow_up lives under `diagnoses`, not `demographic`.
        # Requesting it from demographic returns null and silently drops every
        # censored (alive) patient -> deaths-only biased cohort. Pull from diagnoses.
        "fields": (
            "submitter_id,demographic.days_to_death,"
            "demographic.age_at_index,demographic.vital_status,"
            "diagnoses.days_to_last_follow_up,"
            "samples.submitter_id,samples.sample_type"
        ),
        "format": "json",
        "size": "1000",
    }
    res = _post(GDC_CASES, payload)
    rows = []
    for h in res["data"]["hits"]:
        dem = h.get("demographic", {})
        diags = h.get("diagnoses", []) or [{}]
        # a case can have several diagnoses; use the longest recorded follow-up
        dlfu = [d.get("days_to_last_follow_up") for d in diags if d.get("days_to_last_follow_up") is not None]
        dlfu = max(dlfu) if dlfu else None
        for s in h.get("samples", []):
            if s.get("sample_type") != "Primary Tumor":
                continue
            rows.append(
                {
                    "case": h["submitter_id"],
                    "sample": s["submitter_id"],
                    "days_to_death": dem.get("days_to_death"),
                    "days_to_last_follow_up": dlfu,
                    "age": dem.get("age_at_index"),
                    "vital_status": dem.get("vital_status"),
                }
            )
    clin = pd.DataFrame(rows)
    clin["os_days"] = clin["days_to_death"].fillna(clin["days_to_last_follow_up"])
    clin["os_event"] = (clin["vital_status"] == "Dead").astype(int)
    out = out_dir / "tcga_gbm_clinical.tsv"
    clin.to_csv(out, sep="\t", index=False)
    log.info("clinical n=%d -> %s", len(clin), out)
    return out


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    d = sys.argv[1] if len(sys.argv) > 1 else "data/tcga"
    download_expression(d)
    download_clinical(d)
