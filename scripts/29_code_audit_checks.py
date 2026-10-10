"""Independent re-derivation of the quantities the manuscript reports.

Every check recomputes a value from raw data or from the saved run records using
its own implementation (scikit-learn directly, not the project's evaluate.py) and
compares it with the value printed in the manuscript. Writes
results/tables/main/code_audit_checks.csv.

Usage: python scripts/29_code_audit_checks.py
"""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "results/revision"
MODELS = ["logistic", "mlp", "mlpnbr", "sage", "gat"]
SEEDS = [42, 7, 123]
CHECKS: list[dict] = []


def chk(section, quantity, published, recomputed, tol=5e-4, source=""):
    try:
        ok = abs(float(published) - float(recomputed)) <= tol
        pub, rec = f"{float(published):.4g}", f"{float(recomputed):.4g}"
    except (TypeError, ValueError):
        ok = str(published) == str(recomputed)
        pub, rec = str(published), str(recomputed)
    CHECKS.append(dict(section=section, quantity=quantity, published=pub,
                       recomputed=rec, agrees=ok, source=source))


def mf1(y, yh):
    return float(f1_score(y, yh, average="macro", labels=sorted(set(y.tolist())), zero_division=0))


# --------------------------------------------------------------- LOPO from preds
def lopo():
    pats = sorted({re.match(r"final_\w+?_(SNU\d+)_s\d+\.npz", os.path.basename(f)).group(1)
                   for f in glob.glob(str(REV / "preds/final_*.npz"))})
    store = {}
    for m in MODELS:
        for s in SEEDS:
            for p in pats:
                z = np.load(REV / f"preds/final_{m}_{p}_s{s}.npz")
                k = z["y"] >= 0
                store[(m, s, p)] = (z["y"][k], (z["prob"][k] / z["thr"]).argmax(1), z["prob"][k])
    pooled, per = {}, {}
    for m in MODELS:
        v = []
        for s in SEEDS:
            Y = np.concatenate([store[(m, s, p)][0] for p in pats])
            H = np.concatenate([store[(m, s, p)][1] for p in pats])
            v.append(mf1(Y, H))
        pooled[m] = float(np.mean(v))
        per[m] = {p: float(np.mean([mf1(*store[(m, s, p)][:2]) for s in SEEDS])) for p in pats}
    pub = {"logistic": 0.691, "mlp": 0.655, "mlpnbr": 0.637, "sage": 0.652, "gat": 0.649}
    for m in MODELS:
        chk("Table 2", f"pooled LOPO macro-F1, {m}", pub[m], pooled[m], 5e-4, "results/revision/preds")
    pub_auc = {"logistic": [0.82, 0.97, 0.90, 0.96], "sage": [0.77, 0.92, 0.83, 0.92],
               "mlp": [0.80, 0.89, 0.86, 0.91], "mlpnbr": [0.74, 0.93, 0.79, 0.92],
               "gat": [0.75, 0.90, 0.80, 0.92]}
    for m in MODELS:
        au = np.mean([[roc_auc_score((np.concatenate([store[(m, s, p)][0] for p in pats]) == c).astype(int),
                                     np.concatenate([store[(m, s, p)][2] for p in pats])[:, c])
                       for c in range(4)] for s in SEEDS], axis=0)
        for c, name in enumerate(["cold", "hot", "normal", "necrotic"]):
            chk("Table 2", f"pooled AUROC {m} / {name}", pub_auc[m][c], au[c], 5e-3, "results/revision/preds")
    base = np.array([per["sage"][p] for p in pats])
    pubt = {"logistic": (12, 7e-4), "mlp": (9, 0.048), "mlpnbr": (8, 0.27), "gat": (5, 0.34)}
    for m, (w, p_) in pubt.items():
        a = np.array([per[m][p] for p in pats])
        chk("Table 2", f"{m} vs GraphSAGE: patients won", w, int((a > base).sum()), 0, "results/revision/preds")
        chk("Table 2", f"{m} vs GraphSAGE: Wilcoxon p", p_, stats.wilcoxon(a, base).pvalue, 5e-3,
            "results/revision/preds")
    # patient bootstrap of the pooled difference
    rng = np.random.default_rng(0)
    idx = {p: np.where(np.concatenate([np.full((store[("sage", 42, q)][0]).size, q) for q in pats]) == p)[0]
           for p in pats}
    data = {(m, s): (np.concatenate([store[(m, s, p)][0] for p in pats]),
                     np.concatenate([store[(m, s, p)][1] for p in pats])) for m in ("logistic", "sage") for s in SEEDS}
    diffs = []
    for _ in range(2000):
        sel = np.concatenate([idx[p] for p in rng.choice(pats, len(pats), replace=True)])
        diffs.append(np.mean([mf1(data[("logistic", s)][0][sel], data[("logistic", s)][1][sel]) for s in SEEDS])
                     - np.mean([mf1(data[("sage", s)][0][sel], data[("sage", s)][1][sel]) for s in SEEDS]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    chk("Table 2", "logistic - GraphSAGE pooled difference", 0.039, pooled["logistic"] - pooled["sage"],
        5e-4, "results/revision/preds")
    chk("Table 2", "bootstrap CI lower", 0.009, lo, 5e-3, "results/revision/preds")
    chk("Table 2", "bootstrap CI upper", 0.076, hi, 5e-3, "results/revision/preds")


# --------------------------------------------------------------- homophily
def homophily():
    import torch
    sys_path = str(ROOT / "src")
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from deepgraph_gbm.data.graph import build_knn_graph
    lab = {os.path.basename(f).replace("_spot_labels.csv", ""): pd.read_csv(f, index_col=0)
           for f in glob.glob(str(ROOT / "data/preprocessed/*_spot_labels.csv"))}
    classes = ["immune_cold", "immune_hot", "normal", "necrotic"]
    same = tot = 0
    per = {c: [] for c in classes}
    rng = np.random.default_rng(0)
    chance = {c: [0, 0] for c in classes}
    for sec, df in lab.items():
        coords = np.load(ROOT / f"data/preprocessed/{sec}_coords.npy")
        ei, _ = build_knn_graph(coords, k=6)
        y = df["niche"].values
        src, dst = ei.numpy()
        m = (y[src] != "exclude") & (y[dst] != "exclude")
        same += int((y[src][m] == y[dst][m]).sum()); tot += int(m.sum())
        for c in classes:
            sel = m & (y[src] == c)
            if sel.any():
                # per-spot mean over its labeled neighbors, pooled over spots (Methods definition)
                agree = (y[dst][sel] == c).astype(float)
                num = np.bincount(src[sel], weights=agree, minlength=len(y))
                den = np.bincount(src[sel], minlength=len(y))
                per[c].extend((num[den > 0] / den[den > 0]).tolist())
        yp = y.copy(); keep = y != "exclude"
        yp[keep] = rng.permutation(y[keep])
        for c in classes:
            s = m & (yp[src] == c)
            chance[c][0] += int((yp[dst][s] == c).sum()); chance[c][1] += int(s.sum())
    stored = json.loads((REV / "label_homophily.json").read_text())
    chk("Section 2.5", "edge homophily, overall", 0.80, same / tot, 5e-3, "recomputed from coords + labels")
    chk("Section 2.5", "edge homophily vs stored", stored["overall_edge_homophily"], same / tot, 5e-3,
        "label_homophily.json")
    for c in classes:
        chk("Section 2.5", f"homophily {c} (per-spot mean)", stored["per_class"][c]["homophily"],
            float(np.mean(per[c])), 5e-3, "recomputed")
        chk("Section 2.5", f"chance homophily {c}", stored["per_class"][c]["chance"], chance[c][0] / chance[c][1],
            2e-2, "recomputed (one shuffle)")


# --------------------------------------------------------------- other records
def records():
    s = json.loads((REV / "revision_summary.json").read_text())
    ab = s["ablation"]
    sage = [v["mean"] for k, v in ab.items() if k.startswith("abl_sage_")]
    chk("Section 2.5", "GraphSAGE ablation min", 0.382, min(sage), 5e-4, "revision_summary.json")
    chk("Section 2.5", "GraphSAGE ablation max", 0.425, max(sage), 5e-4, "revision_summary.json")
    nbr = [v["mean"] for k, v in ab.items() if k.startswith("abl_mlpnbr_")]
    chk("Section 2.5", "MLP+neighbors ablation min", 0.383, min(nbr), 5e-4, "revision_summary.json")
    chk("Section 2.5", "MLP+neighbors ablation max", 0.425, max(nbr), 5e-4, "revision_summary.json")
    chk("Section 2.5", "one-layer MLP", 0.460, ab["abl_mlp_L1"]["mean"], 5e-4, "revision_summary.json")
    chk("Section 2.5", "two-layer MLP", 0.424, ab["abl_mlp_L2"]["mean"], 5e-4, "revision_summary.json")
    chk("Section 2.5", "three-layer MLP", 0.419, ab["abl_mlp_L3"]["mean"], 5e-4, "revision_summary.json")
    mr = s["marker_removal"]["per_patient_mean_macro_f1"]
    chk("Section 2.5", "logistic at default regularization", 0.448, mr["logistic"]["0"], 5e-4, "revision_summary.json")
    chk("Section 2.5", "GraphSAGE before marker removal", 0.404, mr["sage"]["0"], 5e-4, "revision_summary.json")
    chk("Section 2.5", "GraphSAGE after removing 100 genes", 0.340, mr["sage"]["100"], 5e-4, "revision_summary.json")
    chk("Section 2.5", "logistic after removing 500 genes", 0.293, mr["logistic"]["500"], 5e-4, "revision_summary.json")
    chk("Section 2.5", "MLP after removing 500 genes", 0.253, mr["mlp"]["500"], 5e-4, "revision_summary.json")
    chk("Section 2.5", "MLP+neighbors after removing 500", 0.271, mr["mlpnbr"]["500"], 5e-4, "revision_summary.json")
    chk("Section 2.5", "GraphSAGE after removing 500 genes", 0.258, mr["sage"]["500"], 5e-4, "revision_summary.json")
    cc = s["chosen_configs"]
    ks = [c["k"] for c in cc["sage"].values()]
    chk("Section 2.5", "GraphSAGE folds with k in {12,18}", 13, sum(k in (12, 18) for k in ks), 0,
        "chosen_configs")
    chk("Section 2.5", "GraphSAGE folds with 2 or 3 layers", 12,
        sum(c["layers"] in (2, 3) for c in cc["sage"].values()), 0, "chosen_configs")
    chk("Section 2.5", "logistic folds with C <= 0.003", 12,
        sum(c["C"] <= 0.003 for c in cc["logistic"].values()), 0, "chosen_configs")
    hx = json.loads((ROOT / "results/revision_hex/hex_sensitivity.json").read_text())
    chk("Section 2.5", "exact-hex GraphSAGE pooled", 0.651, hx["sage"]["pooled_hex"][0], 5e-4, "hex_sensitivity.json")
    chk("Section 2.5", "exact-hex GAT pooled", 0.611, hx["gat"]["pooled_hex"][0], 5e-4, "hex_sensitivity.json")
    hs = json.loads((ROOT / "results/revision_ukf/histology_summary.json").read_text())
    for m, v in [("logistic", 0.419), ("sage", 0.346), ("mlpnbr2", 0.346), ("mlp", 0.336),
                 ("mlpnbr", 0.311), ("gat", 0.291)]:
        chk("Section 2.8 (E1)", f"UKF within-cohort macro-F1, {m}", v,
            hs["E1_ukf_lopo"]["models"][m]["pooled_macro_f1"], 5e-4, "histology_summary.json")
    for m, v in [("sage", 0.395), ("mlp", 0.380), ("mlpnbr", 0.377), ("logistic", 0.352),
                 ("mlpnbr2", 0.315), ("gat", 0.315)]:
        chk("Section 2.8 (E2)", f"UKF transfer macro-F1, {m}", v,
            hs["E2_transfer"][m]["macro_f1_tuned"], 5e-4, "histology_summary.json")
    t1 = json.loads((ROOT / "results/tables/main/Table1_model_comparison.json").read_text())
    for key, name, v in [("gnn_multiseed", "GraphSAGE", 0.489), ("mlp_multiseed", "MLP", 0.578),
                         ("logistic_multiseed", "Logistic", 0.554)]:
        chk("Table 1", f"fixed-split macro-F1, {name}", v, t1[key]["macro_f1"]["mean"], 5e-4,
            "Table1_model_comparison.json")
    ex = json.loads((ROOT / "results/external_v2/greenwald_summary.json").read_text())
    n = sum(v["n_spots"] for v in ex.values())
    comp = {c: sum(v[f"frac_{c}"] * v["n_spots"] for v in ex.values()) / n
            for c in ["immune_cold", "immune_hot", "normal", "necrotic"]}
    chk("Section 2.7", "external predicted immune-hot fraction", 0.778, comp["immune_hot"], 5e-3,
        "greenwald_summary.json")
    chk("Section 2.7", "external predicted necrotic fraction", 0.200, comp["necrotic"], 5e-3,
        "greenwald_summary.json")
    chk("Section 2.7", "external predicted normal fraction", 0.014, comp["normal"], 5e-3, "greenwald_summary.json")
    chk("Section 2.7", "external predicted immune-cold fraction", 0.008, comp["immune_cold"], 5e-3,
        "greenwald_summary.json")
    chk("Section 2.7", "external samples with <1% immune-cold", 12,
        sum(v["frac_immune_cold"] < 0.01 for v in ex.values()), 0, "greenwald_summary.json")
    r = [v["mes_pearson_vs_neftel"] for v in ex.values()]
    chk("Section 2.7", "external MES r, mean", 0.21, float(np.mean(r)), 5e-3, "greenwald_summary.json")
    chk("Section 2.7", "external MES r, median", 0.15, float(np.median(r)), 5e-3, "greenwald_summary.json")
    chk("Section 2.7", "external spots", 25488, n, 0, "greenwald_summary.json")
    sv = json.loads((ROOT / "results/tables/main/Table3_survival_results.json").read_text())
    chk("Table 3", "Cox HR per SD", 1.12, sv["cox_univariate_per_sd"]["HR"], 5e-3, "Table3_survival_results.json")
    chk("Table 3", "age positive control HR per SD", 1.55, sv["positive_control_age_per_sd"]["HR"], 5e-3,
        "Table3_survival_results.json")
    chk("Table 3", "smallest detectable HR per SD", 1.28, sv["min_detectable_HR_per_sd_80pct_power"], 5e-3,
        "Table3_survival_results.json")


def architecture():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from deepgraph_gbm.models.multitask import DeepGraphGBM
    for enc, v in [("graphsage", 1751815), ("gat", 985863)]:
        m = DeepGraphGBM(in_dim=3000, encoder_type=enc)
        chk("Methods", f"{enc} parameter count", v, sum(p.numel() for p in m.parameters()), 0,
            "src/deepgraph_gbm/models")


def labels_and_geometry():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from deepgraph_gbm.data.graph import build_knn_graph
    from deepgraph_gbm.data.labels import assign_niche_labels, mes_target
    meta = pd.read_csv(ROOT / "data/raw/snuh/meta.tsv", sep="\t", index_col=0, low_memory=False)
    rules = yaml.safe_load(open(ROOT / "configs/labels.yaml"))
    lab = assign_niche_labels(meta, rules)
    stored = pd.concat([pd.read_csv(f, index_col=0) for f in
                        glob.glob(str(ROOT / "data/preprocessed/*_spot_labels.csv"))])
    common = stored.index.intersection(lab.index)
    chk("Methods", "rule-derived niche labels reproduce stored labels", 1.0,
        float((lab.loc[common].values == stored.loc[common, "niche"].values).mean()), 1e-6,
        "configs/labels.yaml + data/raw/snuh/meta.tsv")
    mes = mes_target(meta, rules).loc[common].fillna(0.0).values
    chk("Methods", "rule-derived MES target reproduces stored", 1.0,
        float((mes == stored.loc[common, "mes"].fillna(0.0).values).mean()), 1e-6, "configs/labels.yaml")
    chk("Section 2.1", "labeled spots", 75697, int((stored.niche != "exclude").sum()), 0,
        "data/preprocessed/*_spot_labels.csv")
    chk("Section 2.1", "total spots", 100488, len(stored), 0, "data/preprocessed/*_spot_labels.csv")
    # Visium array indices with a doubled column: the six true neighbours are at
    # (|dcol|,|drow|) = (1,1) or (2,0); the second ring is the next shell out.
    SECOND = {(0, 2), (2, 2), (3, 1), (1, 3), (4, 0), (3, 3)}
    tot = nonadj = second = 0
    for f in sorted(glob.glob(str(ROOT / "data/preprocessed/*_coords.npy"))):
        c = np.load(f)
        ei, _ = build_knn_graph(c, k=6)
        e = ei.numpy(); d = np.abs(c[e[0]] - c[e[1]])
        first = ((d[:, 0] == 1) & (d[:, 1] == 1)) | ((d[:, 0] == 2) & (d[:, 1] == 0))
        sec = np.zeros(len(first), bool)
        for a, b in SECOND:
            sec |= (d[:, 0] == a) & (d[:, 1] == b)
        tot += len(first); nonadj += int((~first).sum()); second += int((sec & ~first).sum())
    chk("Methods", "fraction of 6-NN edges joining non-adjacent spots", 0.24, nonadj / tot, 1e-2,
        "data/preprocessed/*_coords.npy")
    chk("Methods", "of those, fraction of all edges in the second ring", nonadj / tot, second / tot, 5e-3,
        "data/preprocessed/*_coords.npy")


def main():
    labels_and_geometry()
    architecture()
    lopo()
    homophily()
    records()
    df = pd.DataFrame(CHECKS)
    out = ROOT / "results/tables/main/code_audit_checks.csv"
    df.to_csv(out, index=False)
    bad = df[~df.agrees]
    print(df.to_string(index=False))
    print(f"\n{len(df) - len(bad)}/{len(df)} checks agree")
    if len(bad):
        print("\nDISAGREEMENTS:")
        print(bad.to_string(index=False))


if __name__ == "__main__":
    main()
