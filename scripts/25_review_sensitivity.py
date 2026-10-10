"""Robustness analyses requested at review of manuscript v6.

Three analyses, all computed from records that already exist in `results/`; no
model is retrained.

1. Threshold-tuning support of the six-split benchmark.  The per-class decision
   thresholds of each split are tuned on that split's two validation patients.
   This writes the labelled class support of every tuning block and re-scores
   each split over the three classes whose support is adequate in every block,
   so the published four-class contrasts can be compared with contrasts that do
   not depend on the degenerate immune-hot threshold.

2. Support floor for the per-patient LOPO tests.  Per-patient macro-F1 averages
   F1 over the classes present in that patient, with no minimum support; several
   patients carry single-digit counts of a class.  This recomputes every paired
   contrast under minimum per-class supports of 1 (as published), 50 and 200
   labelled spots, and adds Holm correction across the four-contrast family.

3. Decision rule.  Pooled LOPO macro-F1 under the validation-tuned thresholds
   used throughout the paper and under plain argmax, for every model.

Usage: python scripts/25_review_sensitivity.py [--rev results/revision] [--out results/tables/supplement]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

MODELS = ["logistic", "mlp", "mlpnbr", "sage", "gat"]
NICE = {"logistic": "Logistic", "mlp": "MLP", "mlpnbr": "MLP + neighbors",
        "sage": "GraphSAGE", "gat": "GAT"}
SEEDS = [42, 7, 123]
CLASSES = ["immune_cold", "immune_hot", "normal", "necrotic"]


def macro_f1(y, yhat, labels=None):
    labels = sorted(set(y.tolist())) if labels is None else labels
    if not labels:
        return np.nan
    return float(f1_score(y, yhat, average="macro", labels=labels, zero_division=0))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


# --------------------------------------------------------------- 1. six splits

def split_support(data_dir: Path, splits_json: Path):
    lab = pd.concat([pd.read_csv(f).assign(section=Path(f).name.replace("_spot_labels.csv", ""))
                     for f in glob.glob(str(data_dir / "*_spot_labels.csv"))], ignore_index=True)
    lab["patient"] = lab.section.str.extract(r"^(SNU\d+)")
    cnt = (lab[lab.niche != "exclude"].groupby(["patient", "niche"]).size()
           .unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0))
    sp = json.loads(splits_json.read_text())["splits"]
    rows = []
    for k in sorted(sp):
        v = cnt.loc[sp[k]["val"], CLASSES].sum()
        rows.append(dict(split=k, val_patients=" + ".join(sp[k]["val"]),
                         **{f"val_{c}": int(v[c]) for c in CLASSES}))
    return pd.DataFrame(rows), cnt


def multisplit_scores(ms_dir: Path):
    rows = []
    for f in sorted(ms_dir.glob("*.json")):
        d = json.loads(f.read_text())
        if "test" not in d:
            continue
        t = d["test"]
        enc = d.get("encoder")
        model = d["model"] if d["model"] != "gnn" else ("gat" if enc and "gat" in str(enc).lower() else "sage")
        rows.append(dict(model=model, split=d["split"], seed=d["seed"],
                         macro4=t["macro_f1"],
                         macro3=np.mean([t["f1_immune_cold"], t["f1_normal"], t["f1_necrotic"]])))
    return pd.DataFrame(rows)


def six_split_analysis(ms: pd.DataFrame, support: pd.DataFrame):
    piv4 = ms.groupby(["split", "model"])["macro4"].mean().unstack()
    piv3 = ms.groupby(["split", "model"])["macro3"].mean().unstack()
    out = []
    for a, b in [("logistic", "sage"), ("mlp", "sage"), ("gat", "sage"),
                 ("gat", "mlp"), ("gat", "logistic"), ("logistic", "mlp")]:
        d4, d3 = piv4[a] - piv4[b], piv3[a] - piv3[b]
        out.append(dict(
            contrast=f"{NICE[a]} - {NICE[b]}",
            published_diff=round(float(d4.mean()), 4),
            published_wins=f"{int((d4 > 0).sum())}/6",
            published_p_t=round(float(stats.ttest_rel(piv4[a], piv4[b]).pvalue), 4),
            restricted_diff=round(float(d3.mean()), 4),
            restricted_wins=f"{int((d3 > 0).sum())}/6",
            restricted_p_t=round(float(stats.ttest_rel(piv3[a], piv3[b]).pvalue), 4)))
    tab = support.copy()
    for m in ["logistic", "mlp", "sage", "gat"]:
        tab[f"macro4_{m}"] = [round(float(piv4.loc[s, m]), 3) for s in tab.split]
        tab[f"macro3_{m}"] = [round(float(piv3.loc[s, m]), 3) for s in tab.split]
    return tab, pd.DataFrame(out)


# ------------------------------------------------------------ 2 & 3. LOPO LOPO

def load_lopo(rev: Path):
    pats = sorted({re.match(r"final_\w+?_(SNU\d+)_s\d+\.npz", os.path.basename(f)).group(1)
                   for f in glob.glob(str(rev / "preds" / "final_*.npz"))})
    store = {}
    for m in MODELS:
        for s in SEEDS:
            for p in pats:
                z = np.load(rev / "preds" / f"final_{m}_{p}_s{s}.npz")
                msk = z["y"] >= 0
                store[(m, s, p)] = (z["y"][msk],
                                    (z["prob"][msk] / z["thr"]).argmax(1),
                                    z["prob"][msk].argmax(1))
    return pats, store


def support_floor_analysis(pats, store):
    rows = []
    for floor in [1, 50, 200]:
        per = {}
        for m in MODELS:
            per[m] = {}
            for p in pats:
                vals = []
                for s in SEEDS:
                    y, yh, _ = store[(m, s, p)]
                    labels = [c for c in range(4) if int((y == c).sum()) >= floor]
                    vals.append(macro_f1(y, yh, labels))
                per[m][p] = float(np.nanmean(vals))
        base = np.array([per["sage"][p] for p in pats])
        raw = {}
        for m in ["logistic", "mlp", "mlpnbr", "gat"]:
            a = np.array([per[m][p] for p in pats])
            raw[m] = (float(np.mean(a - base)), int((a > base).sum()),
                      float(stats.wilcoxon(a, base).pvalue))
        adj = holm([raw[m][2] for m in ["logistic", "mlp", "mlpnbr", "gat"]])
        for m, q in zip(["logistic", "mlp", "mlpnbr", "gat"], adj):
            d, w, p = raw[m]
            rows.append(dict(floor=("none (published)" if floor == 1 else f">= {floor} spots"),
                             contrast=f"{NICE[m]} - GraphSAGE",
                             mean_per_patient_diff=round(d, 4), wins=f"{w}/13",
                             wilcoxon_p=float(f"{p:.3g}"), wilcoxon_p_holm=float(f"{q:.3g}"),
                             classes_retained=int(np.mean([
                                 len([c for c in range(4) if int((store[('sage', 42, p)][0] == c).sum()) >= floor])
                                 for p in pats]) * 100) / 100))
    return pd.DataFrame(rows)


def decision_rule_analysis(pats, store):
    rows = []
    pooled = {}
    for m in MODELS:
        tuned, arg = [], []
        for s in SEEDS:
            Y = np.concatenate([store[(m, s, p)][0] for p in pats])
            tuned.append(macro_f1(Y, np.concatenate([store[(m, s, p)][1] for p in pats])))
            arg.append(macro_f1(Y, np.concatenate([store[(m, s, p)][2] for p in pats])))
        pooled[m] = (float(np.mean(tuned)), float(np.mean(arg)))
        rows.append(dict(model=NICE[m],
                         pooled_macro_f1_tuned=round(np.mean(tuned), 6),
                         pooled_macro_f1_argmax=round(np.mean(arg), 6),
                         tuned_minus_argmax=round(np.mean(tuned) - np.mean(arg), 6)))
    df = pd.DataFrame(rows)
    gaps = pd.DataFrame([
        dict(model="Logistic - GraphSAGE",
             pooled_macro_f1_tuned=round(pooled["logistic"][0] - pooled["sage"][0], 6),
             pooled_macro_f1_argmax=round(pooled["logistic"][1] - pooled["sage"][1], 6),
             tuned_minus_argmax=round((pooled["logistic"][0] - pooled["sage"][0])
                                      - (pooled["logistic"][1] - pooled["sage"][1]), 6)),
        dict(model="MLP - GraphSAGE",
             pooled_macro_f1_tuned=round(pooled["mlp"][0] - pooled["sage"][0], 6),
             pooled_macro_f1_argmax=round(pooled["mlp"][1] - pooled["sage"][1], 6),
             tuned_minus_argmax=round((pooled["mlp"][0] - pooled["sage"][0])
                                      - (pooled["mlp"][1] - pooled["sage"][1]), 6))])
    return pd.concat([df, gaps], ignore_index=True)


# ------------------------------------------------------- 4. slide-level risk head

def slide_risk_analysis(eval_glob: str, sections_meta: Path):
    """Does the attention-pooled slide head recover its own pseudo-risk target
    on held-out sections?  Target is the rule-derived niche-composition risk."""
    tgt = pd.read_csv(sections_meta).set_index("section")["risk"]
    rows = []
    for d in sorted(glob.glob(eval_glob)):
        seed = int(os.path.basename(d).replace("eval_seed", ""))
        t = pd.read_csv(Path(d) / "test_predictions.csv")
        pred = t.groupby("section")["risk"].mean()
        y = tgt.loc[pred.index]
        pr, sr = stats.pearsonr(pred.values, y.values), stats.spearmanr(pred.values, y.values)
        rows.append(dict(seed=seed, n_sections=len(pred),
                         pearson_r=round(float(pr.statistic), 3), pearson_p=round(float(pr.pvalue), 3),
                         spearman_rho=round(float(sr.statistic), 3), spearman_p=round(float(sr.pvalue), 3),
                         rmse=round(float(np.sqrt(np.mean((pred.values - y.values) ** 2))), 3)))
    return pd.DataFrame(rows).sort_values("seed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", default="results/revision")
    ap.add_argument("--multisplit", default="results/multisplit")
    ap.add_argument("--data", default="data/preprocessed")
    ap.add_argument("--splits", default="configs/splits_multisplit.json")
    ap.add_argument("--out", default="results/tables/supplement")
    ap.add_argument("--eval-glob", default="results/rerun_20261009/eval_seed*")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    support, _ = split_support(Path(a.data), Path(a.splits))
    ms = multisplit_scores(Path(a.multisplit))
    s10, s10b = six_split_analysis(ms, support)
    s10.to_csv(out / "TableS10_split_tuning_support.csv", index=False)
    s10b.to_csv(out / "TableS11_split_contrasts_restricted.csv", index=False)

    pats, store = load_lopo(Path(a.rev))
    s11 = support_floor_analysis(pats, store)
    s11.to_csv(out / "TableS12_lopo_support_floor.csv", index=False)
    s12 = decision_rule_analysis(pats, store)
    s12.to_csv(out / "TableS13_decision_rule.csv", index=False)
    risk = slide_risk_analysis(a.eval_glob, Path(a.out) / "TableS2_sections_meta.csv")
    risk.to_csv(Path(a.out).parent / "main" / "slide_risk_head_eval.csv", index=False)

    print(s10.to_string(index=False)); print()
    print(s10b.to_string(index=False)); print()
    print(s11.to_string(index=False)); print()
    print(s12.to_string(index=False)); print()
    print(risk.to_string(index=False))


if __name__ == "__main__":
    main()
