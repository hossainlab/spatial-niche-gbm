"""Analyse the neuropathologist-label experiments (scripts/20_make_histology_jobs.py).

E1 (results/revision_ukf): pooled leave-one-patient-out macro-F1 per model
   (mean +/- SD over seeds), per-patient paired tests vs GraphSAGE (n = 18
   patients), patient-bootstrap 95% CI of pooled differences.
E2 (results/revision_transfer_v2): SNUH-trained models tested on all UKF spots:
   macro-F1 with SNUH-tuned thresholds and with argmax, per-class AUROC.

Usage: python scripts/21_histology_analysis.py --figdir results/figures/revision
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CLASSES = ["cellular", "infiltrative", "necrotic", "mvp", "normal"]
CLASS_LABELS = ["cellular tumor", "infiltrative", "necrotic", "microvascular\nproliferation", "normal brain"]
VARIANTS = ["logistic", "mlp", "mlpnbr", "mlpnbr2", "sage", "gat"]
NAMES = {"logistic": "Logistic", "mlp": "MLP", "mlpnbr": "MLP + nbr (1 hop)", "mlpnbr2": "MLP + nbr (2 hops)",
         "sage": "GraphSAGE", "gat": "GAT"}
COLORS = {"logistic": "#009E73", "mlp": "#0072B2", "mlpnbr": "#56B4E9", "mlpnbr2": "#9AD0F0",
          "sage": "#000000", "gat": "#E69F00"}
SEEDS = [42, 7, 123]


def f1m(y, yh):
    from sklearn.metrics import f1_score
    return float(f1_score(y, yh, average="macro", labels=sorted(set(y.tolist())), zero_division=0))


def load_pred(d: Path, variant, fold, seed):
    z = np.load(d / "preds" / f"final_{variant}_{fold}_s{seed}.npz")
    m = z["y"] >= 0
    return z["y"][m], z["prob"][m], z["thr"], z["sections"][m]


def seeds_for(v):
    return [42] if v == "logistic" else SEEDS


def e1(d: Path, n_boot=2000):
    from scipy import stats
    folds = json.loads((d / "folds.json").read_text())
    pats = list(folds)
    res, perpat, pooled = {}, defaultdict(dict), defaultdict(dict)
    for v in VARIANTS:
        vals = []
        for s in seeds_for(v):
            Y, YH, PAT = [], [], []
            for p in pats:
                y, prob, thr, _ = load_pred(d, v, p, s)
                yh = (prob / thr).argmax(1)
                Y.append(y); YH.append(yh); PAT.append(np.array([p] * len(y)))
                perpat[v].setdefault(p, []).append(f1m(y, yh))
            Y, YH, PAT = map(np.concatenate, (Y, YH, PAT))
            pooled[v][s] = (Y, YH, PAT)
            vals.append(f1m(Y, YH))
        from sklearn.metrics import f1_score
        cls = {c: float(np.mean([f1_score(pooled[v][s][0] == i, pooled[v][s][1] == i, zero_division=0)
                                 for s in seeds_for(v)])) for i, c in enumerate(CLASSES)}
        res[v] = {"pooled_macro_f1": float(np.mean(vals)),
                  "pooled_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                  "per_class_f1": cls,
                  "per_patient_mean": float(np.mean([np.mean(perpat[v][p]) for p in pats]))}
    rng = np.random.default_rng(0)
    tests = {}
    for v in VARIANTS:
        if v == "sage":
            continue
        a = np.array([np.mean(perpat[v][p]) for p in pats]); b = np.array([np.mean(perpat["sage"][p]) for p in pats])
        boots = []
        for _ in range(n_boot):
            pick = rng.choice(pats, len(pats), replace=True)
            def pf(var):
                fs = []
                for s in seeds_for(var):
                    Y, YH, PAT = pooled[var][s]
                    idx = np.concatenate([np.where(PAT == p)[0] for p in pick])
                    fs.append(f1m(Y[idx], YH[idx]))
                return np.mean(fs)
            boots.append(pf(v) - pf("sage"))
        tests[f"{v}_vs_sage"] = {
            "pooled_diff": res[v]["pooled_macro_f1"] - res["sage"]["pooled_macro_f1"],
            "boot95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "per_patient_diff": float((a - b).mean()), "wins": int((a > b).sum()), "n": len(pats),
            "wilcoxon_p": float(stats.wilcoxon(a, b).pvalue), "t_p": float(stats.ttest_rel(a, b).pvalue)}
    return {"models": res, "tests_vs_sage": tests, "folds": folds,
            "per_patient": {v: {p: float(np.mean(x)) for p, x in perpat[v].items()} for v in VARIANTS}}, pooled


def e2(d: Path):
    from sklearn.metrics import roc_auc_score, f1_score
    res = {}
    for v in VARIANTS:
        tuned, arg, auc, cls = [], [], defaultdict(list), defaultdict(list)
        for s in seeds_for(v):
            y, prob, thr, _ = load_pred(d, v, "SNUH_to_UKF", s)
            yh = (prob / thr).argmax(1)
            tuned.append(f1m(y, yh)); arg.append(f1m(y, prob.argmax(1)))
            for i, c in enumerate(CLASSES):
                auc[c].append(roc_auc_score(y == i, prob[:, i]))
                cls[c].append(f1_score(y == i, yh == i, zero_division=0))
        res[v] = {"macro_f1_tuned": float(np.mean(tuned)), "macro_f1_tuned_sd": float(np.std(tuned, ddof=1)) if len(tuned) > 1 else 0.0,
                  "macro_f1_argmax": float(np.mean(arg)),
                  "auroc": {c: float(np.mean(x)) for c, x in auc.items()},
                  "mean_auroc": float(np.mean([np.mean(x) for x in auc.values()])),
                  "per_class_f1": {c: float(np.mean(x)) for c, x in cls.items()}}
    return res


def figures(r1, r2, pooled1, figdir: Path, data_ukf: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    xs = np.arange(len(VARIANTS))
    ax = axes[0]
    m = [r1["models"][v]["pooled_macro_f1"] for v in VARIANTS]
    e = [r1["models"][v]["pooled_sd"] for v in VARIANTS]
    ax.bar(xs, m, yerr=e, capsize=2, color=[COLORS[v] for v in VARIANTS], edgecolor="black", linewidth=0.5)
    for x, val, sd in zip(xs, m, e):
        ax.text(x, val + sd + 0.02, f"{val:.2f}", ha="center", fontsize=6)
    ax.set_xticks(xs, [NAMES[v] for v in VARIANTS], rotation=50, ha="right", fontsize=6.5)
    ax.set_ylabel("Pooled macro-F1"); ax.set_ylim(0, 1)
    ax.set_title("A  Within UKF (LOPO, n = 18)", loc="left", fontsize=8, fontweight="bold")
    ax = axes[1]
    m = [r2[v]["macro_f1_tuned"] for v in VARIANTS]
    a = [r2[v]["mean_auroc"] for v in VARIANTS]
    ax.bar(xs - 0.2, m, 0.4, color=[COLORS[v] for v in VARIANTS], edgecolor="black", linewidth=0.5, label="macro-F1")
    ax.bar(xs + 0.2, a, 0.4, color=[COLORS[v] for v in VARIANTS], edgecolor="black", linewidth=0.5, hatch="///",
           alpha=0.6, label="mean AUROC")
    ax.set_xticks(xs, [NAMES[v] for v in VARIANTS], rotation=50, ha="right", fontsize=6.5)
    ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=6)
    ax.set_title("B  SNUH → UKF transfer", loc="left", fontsize=8, fontweight="bold")
    for a_ in axes[:2]:
        for s in ["top", "right"]:
            a_.spines[s].set_visible(False)
    # C: pooled LOPO confusion, best graph-free vs GraphSAGE (seed 42)
    from sklearn.metrics import confusion_matrix
    best = max(["logistic", "mlp", "mlpnbr", "mlpnbr2"], key=lambda v: r1["models"][v]["pooled_macro_f1"])
    Y, YH, _ = pooled1["sage"][42]
    cm = confusion_matrix(Y, YH, labels=range(5), normalize="true")
    ax = axes[2]
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    short = ["cell.", "infil.", "necr.", "MVP", "normal"]
    ax.set_xticks(range(5), short, rotation=45, ha="right"); ax.set_yticks(range(5), short)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if cm[i, j] > 0.5 else "black")
    ax.set_title("C  GraphSAGE, UKF LOPO", loc="left", fontsize=8, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Pathologist")
    fig.tight_layout()
    fig.savefig(figdir / "FigR5_pathologist_labels.png", dpi=300, bbox_inches="tight")
    fig.savefig(figdir / "FigR5_pathologist_labels.svg", bbox_inches="tight")
    plt.close(fig)

    # D: example maps (pathologist vs LOPO predictions), two informative sections
    import torch
    graphs = {g.section: g for g in torch.load(data_ukf / "graphs.pt", weights_only=False)}
    pal = {0: "#D55E00", 1: "#E69F00", 2: "#CC79A7", 3: "#0072B2", 4: "#009E73", -1: "#E5E5E5"}
    for sec in ["UKF269_T", "UKF260_T", "UKF243_T", "UKF251_T"]:
        if sec not in graphs:
            continue
        g = graphs[sec]; xy = np.load(data_ukf / f"{sec}_coords.npy"); y = g.niche_y.numpy()
        pat = g.patient
        panels = [("Neuropathologist (H&E)", y)]
        for v in [best, "sage"]:
            yy, prob, thr, secs = load_pred(Path("results/revision_ukf"), v, pat, 42)
            full = np.full(len(y), -1); full[y >= 0] = (prob / thr).argmax(1)[secs == sec]
            panels.append((f"{NAMES[v]} (LOPO)", full))
        fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
        for ax, (t, yy) in zip(axs, panels):
            ax.scatter(xy[:, 1], -xy[:, 0], c=[pal[int(v)] for v in yy], s=1.5, linewidths=0)
            ax.set_aspect("equal"); ax.axis("off"); ax.set_title(t, fontsize=7)
        h = [plt.Line2D([], [], marker="o", ls="", color=pal[i]) for i in [0, 1, 2, 3, 4, -1]]
        fig.legend(h, ["cellular", "infiltrative", "necrotic", "MVP", "normal", "unannotated"],
                   loc="lower center", ncol=6, frameon=False, fontsize=6.5)
        fig.suptitle(sec, fontsize=8); fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig(figdir / f"maps_path_{sec}.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdir", default="results/figures/revision")
    ap.add_argument("--data-ukf", default="data/preprocessed_histology/ukf")
    ap.add_argument("--transfer", default="results/revision_transfer_v2",
                    help="E2 run dir (v2: UKF inputs scaled to 1e5/spot like the SNUH matrix)")
    args = ap.parse_args()
    figdir = Path(args.figdir); figdir.mkdir(parents=True, exist_ok=True)
    r1, pooled1 = e1(Path("results/revision_ukf"))
    r2 = e2(Path(args.transfer))
    out = {"E1_ukf_lopo": r1, "E2_transfer": r2}
    Path("results/revision_ukf/histology_summary.json").write_text(json.dumps(out, indent=1))
    figures(r1, r2, pooled1, figdir, Path(args.data_ukf))
    for v in VARIANTS:
        m = r1["models"][v]
        print(f"E1 {v:9s} pooled {m['pooled_macro_f1']:.3f}±{m['pooled_sd']:.3f}  per-pt {m['per_patient_mean']:.3f} | "
              f"E2 tuned {r2[v]['macro_f1_tuned']:.3f} argmax {r2[v]['macro_f1_argmax']:.3f} AUROC {r2[v]['mean_auroc']:.3f}")
    print(json.dumps(r1["tests_vs_sage"], indent=1))


if __name__ == "__main__":
    main()
