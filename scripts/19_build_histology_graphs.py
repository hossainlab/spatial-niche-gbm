"""Build graphs for the HISTOLOGY task with a shared 5-class label set
{cellular, infiltrative, necrotic, mvp, normal}, on two cohorts:

* SNUH atlas -- labels from the atlas anatomical-feature (AF) annotation, which
  the atlas derived from Ivy GAP marker-gene expression (EXPRESSION-DERIVED):
  CT -> cellular, IT -> infiltrative, PAN/PNZ -> necrotic, MVP -> mvp,
  LE_GM/LE_WM -> normal; IGN/BV/HBV excluded.
* UKF cohort (Ravi et al. 2022; Ritter et al. 2025) -- labels from the
  NEUROPATHOLOGIST's manual H&E segmentation: Cellular -> cellular,
  Infiltrative -> infiltrative, Necrosis/Necrotic_Edge -> necrotic,
  Vascular_Hyper -> mvp, Grey_Matter/White_Matter -> normal; unannotated excluded.
  Non-GBM samples (270_T astrocytoma, 268_T oligodendroglioma) are dropped.

Two datasets are written (features = log1p of counts scaled to 10,000/spot):
  <out>/ukf       UKF only; 3,000 HVGs (seurat_v3) selected on UKF counts
                  (label-free) -- for within-cohort LOPO on pathologist labels.
  <out>/transfer  SNUH + UKF; genes restricted to those measured in BOTH cohorts,
                  3,000 HVGs selected on SNUH only -- for SNUH -> UKF transfer.
                  UKF counts are scaled to 100,000/spot here to match the SNUH matrix.
SNUH spots/edges are identical to data/preprocessed (same barcodes, 6-NN graph).

Usage: python scripts/19_build_histology_graphs.py --snuh-raw data/raw/snuh \
          --snuh data/preprocessed --ukf data/raw/ukf_spata --out data/preprocessed_histology
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio
import scipy.sparse as sp
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deepgraph_gbm.data.graph import build_knn_graph  # noqa: E402

CLASSES = ["cellular", "infiltrative", "necrotic", "mvp", "normal"]
SNUH_AF_MAP = {"CT": "cellular", "IT": "infiltrative", "PAN": "necrotic", "PNZ": "necrotic",
               "MVP": "mvp", "LE_GM": "normal", "LE_WM": "normal"}
UKF_MAP = {"cellular": "cellular", "infiltrative": "infiltrative", "infiltartive": "infiltrative",
           "necrosis": "necrotic", "necrotic_edge": "necrotic", "vascular_hyper": "mvp",
           "grey_matter": "normal", "white_matter": "normal", "wite_matter": "normal"}
UKF_NOT_GBM = {"270_T", "268_T"}
N_HVG = 3000
SNUH_TARGET_SUM = 1e5  # expm1 of every SNUH atlas spot sums to 100,000


def encode(labels):
    idx = {c: i for i, c in enumerate(CLASSES)}
    return np.array([idx.get(l, -1) for l in labels], dtype=np.int64)


def lognorm(counts_spots_x_genes: sp.csr_matrix, target_sum: float = 1e4) -> sp.csr_matrix:
    X = counts_spots_x_genes.astype(np.float32)
    lib = np.asarray(X.sum(1)).ravel()
    X = sp.diags((target_sum / np.maximum(lib, 1)).astype(np.float32)) @ X
    X = sp.csr_matrix(X)
    X.data = np.log1p(X.data)
    return X


def hvg(counts_like: sp.csr_matrix, genes: list[str], n: int, allowed: set | None = None) -> list[str]:
    a = ad.AnnData(X=counts_like, var=pd.DataFrame(index=genes))
    a.var_names_make_unique()
    if allowed is not None:
        a = a[:, [g in allowed for g in a.var_names]].copy()
    sc.pp.highly_variable_genes(a, n_top_genes=n, flavor="seurat_v3", subset=False)
    return a.var.index[a.var["highly_variable"]].tolist()


# --------------------------------------------------------------------------- UKF

def load_ukf(ukf_dir: Path):
    samples = []
    for sd in sorted(p for p in ukf_dir.iterdir() if (p / "spots.csv").exists()):
        if sd.name in UKF_NOT_GBM:
            continue
        spots = pd.read_csv(sd / "spots.csv").set_index("barcodes")
        bcs = [l.strip() for l in open(sd / "barcodes.txt")]
        seg = spots.loc[bcs, "segmentation"].astype(str).str.strip().str.lower()
        y = encode([UKF_MAP.get(s, "exclude") for s in seg])
        if (y >= 0).sum() == 0:
            print(sd.name, "no pathologist annotation -> dropped")
            continue
        genes = [l.strip() for l in open(sd / "genes.txt")]
        C = sp.csr_matrix(sio.mmread(sd / "counts.mtx")).T.tocsr()  # spots x genes
        keep = np.diff(C.indptr) >= 200
        xy = spots.loc[bcs, ["x", "y"]].values.astype(float)
        pid = re.match(r"#?(\d+)", sd.name).group(1)
        samples.append(dict(name=sd.name, section=f"UKF{sd.name.lstrip('#')}", patient=f"UKF{pid}",
                            genes=genes, C=C[keep], y=y[keep], xy=xy[keep]))
    return samples


def ukf_features(s, gene_list, target_sum=1e4):
    gi = {g: i for i, g in enumerate(s["genes"])}
    X = lognorm(s["C"], target_sum)
    cols = [gi.get(g, -1) for g in gene_list]
    out = np.zeros((X.shape[0], len(gene_list)), dtype=np.float32)
    have = [(j, c) for j, c in enumerate(cols) if c >= 0]
    out[:, [j for j, _ in have]] = X[:, [c for _, c in have]].toarray()
    return out, len(have)


def ukf_graph(s, X):
    ei, _ = build_knn_graph(s["xy"], k=6)
    return Data(x=torch.from_numpy(X), edge_index=ei, niche_y=torch.from_numpy(s["y"]),
                mes_y=torch.full((len(s["y"]),), float("nan")), risk_y=0.0, section=s["section"],
                patient=s["patient"], n_spots=int(len(s["y"])), cohort="UKF")


# --------------------------------------------------------------------------- SNUH

def load_snuh(raw: Path, pre: Path):
    meta = pd.read_csv(raw / "meta.tsv", sep="\t", index_col=0, low_memory=False)
    genes = pd.read_csv(raw / "features.tsv.gz", header=None)[0].astype(str).tolist()
    print("loading SNUH matrix ...", flush=True)
    X = sp.csc_matrix(sio.mmread(raw / "matrix.mtx.gz"))  # genes x spots, log1p-normalized
    assert X.shape[1] == len(meta), (X.shape, len(meta))
    pos = pd.Series(np.arange(len(meta)), index=meta.index)
    graphs = torch.load(pre / "graphs.pt", weights_only=False)
    secs = []
    for g in graphs:
        bc = pd.read_csv(pre / f"{g.section}_spot_labels.csv", index_col=0).index
        cols = pos.loc[bc].values
        af = meta.loc[bc, "AF"].astype(str).values
        secs.append(dict(g=g, Xlog=X[:, cols].T.tocsr(),
                         y=encode([SNUH_AF_MAP.get(a, "exclude") for a in af])))
    return secs, genes


def save(dirpath: Path, graphs, coords):
    dirpath.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, dirpath / "graphs.pt")
    for sec, xy in coords.items():
        np.save(dirpath / f"{sec}_coords.npy", xy)
    rows = [{"section": g.section, "patient": g.patient, "cohort": g.cohort, "n_spots": g.n_spots,
             **{c: int((g.niche_y.numpy() == i).sum()) for i, c in enumerate(CLASSES)}} for g in graphs]
    m = pd.DataFrame(rows)
    m.to_csv(dirpath / "sections_meta.csv", index=False)
    print(dirpath, "\n", m.groupby("cohort")[CLASSES + ["n_spots"]].sum(), "\n",
          m.groupby("cohort").patient.nunique().to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snuh-raw", default="data/raw/snuh")
    ap.add_argument("--snuh", default="data/preprocessed")
    ap.add_argument("--ukf", default="data/raw/ukf_spata")
    ap.add_argument("--out", default="data/preprocessed_histology")
    args = ap.parse_args()
    out = Path(args.out)

    ukf = load_ukf(Path(args.ukf))
    # ---- (1) UKF-only dataset: HVGs on UKF counts (label-free), common gene universe
    ukf_genes = sorted(set.intersection(*[set(s["genes"]) for s in ukf]))
    gi = [{g: i for i, g in enumerate(s["genes"])} for s in ukf]
    Cstack = sp.vstack([s["C"][:, [m[g] for g in ukf_genes]] for s, m in zip(ukf, gi)]).tocsr()
    ukf_hvg = hvg(Cstack.astype(np.float32), ukf_genes, N_HVG)
    del Cstack
    graphs, coords = [], {}
    for s in ukf:
        X, n = ukf_features(s, ukf_hvg)
        graphs.append(ukf_graph(s, X)); coords[s["section"]] = s["xy"][:, ::-1].copy()
    save(out / "ukf", graphs, coords)
    json.dump(ukf_hvg, open(out / "ukf" / "hvg_genes.json", "w"))

    # ---- (2) transfer dataset: shared genes, HVGs chosen on SNUH only
    snuh, snuh_genes = load_snuh(Path(args.snuh_raw), Path(args.snuh))
    shared = set(snuh_genes) & set(ukf_genes)
    Xall = sp.vstack([s["Xlog"] for s in snuh]).tocsr()
    Xall.data = np.expm1(Xall.data)  # approximate normalized counts, as in 02_build_graphs
    tr_hvg = hvg(Xall, snuh_genes, N_HVG, allowed=shared)
    del Xall
    gidx = {g: i for i, g in enumerate(snuh_genes)}
    hidx = [gidx[g] for g in tr_hvg]
    graphs, coords = [], {}
    for s in snuh:
        g = s["g"]
        graphs.append(Data(x=torch.from_numpy(s["Xlog"][:, hidx].toarray().astype(np.float32)),
                           edge_index=g.edge_index, niche_y=torch.from_numpy(s["y"]), mes_y=g.mes_y,
                           risk_y=0.0, section=g.section, patient=g.patient, n_spots=g.n_spots,
                           cohort="SNUH"))
        coords[g.section] = np.load(Path(args.snuh) / f"{g.section}_coords.npy")
    for s in ukf:
        # SNUH atlas values are log1p(counts scaled to 1e5/spot): match that scale
        X, n = ukf_features(s, tr_hvg, SNUH_TARGET_SUM)
        assert n == len(tr_hvg), (s["name"], n)
        graphs.append(ukf_graph(s, X)); coords[s["section"]] = s["xy"][:, ::-1].copy()
    save(out / "transfer", graphs, coords)
    json.dump(tr_hvg, open(out / "transfer" / "hvg_genes.json", "w"))
    markers = ["GFAP", "MBP", "CD74", "SPP1", "EGFR", "CHI3L1", "VEGFA", "PTPRZ1", "SOX2", "OLIG2"]
    print("marker genes in UKF HVGs:", {m: m in ukf_hvg for m in markers})
    print("marker genes in transfer HVGs:", {m: m in tr_hvg for m in markers})
    print("shared genes:", len(shared))


if __name__ == "__main__":
    main()
