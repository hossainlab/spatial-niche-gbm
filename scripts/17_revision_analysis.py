"""Aggregate the revision benchmark (scripts/15_revision_benchmark.py) into
tables and figures.

Primary metric: POOLED leave-one-patient-out (LOPO) macro-F1 -- every labeled
spot is predicted exactly once, by a model that never saw its patient, using
thresholds tuned on that fold's validation patients. Reported as mean +/- SD over
three seeds; uncertainty from a patient-level bootstrap (resampling the 13
patients with replacement). Model comparisons: paired Wilcoxon signed-rank and
paired t-tests on per-patient macro-F1 (n = 13 held-out patients).

Usage: python scripts/17_revision_analysis.py --rev results/revision \
           --data data/preprocessed --figdir results/figures/revision
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CLASSES = ["immune_cold", "immune_hot", "normal", "necrotic"]
LABELS = {"immune_cold": "immune cold", "immune_hot": "immune hot", "normal": "normal", "necrotic": "necrotic"}
NICHE_COLORS = {"immune_cold": "#D55E00", "immune_hot": "#0072B2", "normal": "#009E73",
                "necrotic": "#CC79A7", "exclude": "#E5E5E5"}
MODELS = ["logistic", "mlp", "mlpnbr", "sage", "gat"]
MODEL_NAMES = {"logistic": "Logistic", "mlp": "MLP", "mlpnbr": "MLP + neighbor features",
               "sage": "GraphSAGE", "gat": "GAT"}
MODEL_COLORS = {"logistic": "#009E73", "mlp": "#0072B2", "mlpnbr": "#56B4E9",
                "sage": "#000000", "gat": "#E69F00"}
SEEDS = [42, 7, 123]


def load_runs(rev: Path, prefix: str):
    return [json.loads(p.read_text()) for p in sorted((rev / "runs").glob(f"{prefix}*.json"))]


def macro_f1(y, yhat):
    from sklearn.metrics import f1_score
    present = sorted(set(y.tolist()))
    return float(f1_score(y, yhat, average="macro", labels=present, zero_division=0))


def pooled(rev: Path, model: str, seed: int):
    """Concatenate LOPO test predictions over folds for one model/seed."""
    Y, YH, P, S, PAT = [], [], [], [], []
    folds = json.loads((rev / "folds.json").read_text())
    for fold in folds:
        f = rev / "preds" / f"final_{model}_{fold}_s{seed}.npz"
        d = np.load(f, allow_pickle=False)
        y, prob, thr = d["y"], d["prob"], d["thr"]
        m = y >= 0
        Y.append(y[m]); P.append(prob[m]); YH.append((prob[m] / thr).argmax(1))
        S.append(d["sections"][m]); PAT.append(np.array([fold] * m.sum()))
    return (np.concatenate(Y), np.concatenate(YH), np.concatenate(P),
            np.concatenate(S), np.concatenate(PAT))


def bootstrap_diff(rev, a, b, n_boot=2000, rng=None):
    """Patient-level bootstrap of the pooled macro-F1 difference a - b (seed-averaged)."""
    rng = rng or np.random.default_rng(0)
    data = {}
    for m in (a, b):
        for s in SEEDS:
            data[(m, s)] = pooled(rev, m, s)
    pats = np.unique(data[(a, SEEDS[0])][4])
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(pats, len(pats), replace=True)
        vals = {}
        for m in (a, b):
            fs = []
            for s in SEEDS:
                y, yh, _, _, pat = data[(m, s)]
                idx = np.concatenate([np.where(pat == p)[0] for p in pick])
                fs.append(macro_f1(y[idx], yh[idx]))
            vals[m] = np.mean(fs)
        diffs.append(vals[a] - vals[b])
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", default="results/revision")
    ap.add_argument("--data", default="data/preprocessed")
    ap.add_argument("--figdir", default="results/figures/revision")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--figures-only", action="store_true",
                    help="re-render figures from an existing revision_summary.json")
    args = ap.parse_args()
    if args.figures_only:
        rev = Path(args.rev)
        summary = json.loads((rev / "revision_summary.json").read_text())
        patients = list(summary["folds"])
        per_patient = {m: {p: [v] for p, v in summary["lopo"][m]["per_patient_macro_f1"].items()}
                       for m in MODELS}
        make_figures(rev, Path(args.figdir), summary, per_patient, patients, Path(args.data))
        return
    from scipy import stats
    from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

    rev, figdir = Path(args.rev), Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    folds = json.loads((rev / "folds.json").read_text())
    patients = list(folds)
    summary = {"folds": folds}

    # ---------------- 1. final LOPO comparison
    final = load_runs(rev, "final_")
    per_patient = defaultdict(lambda: defaultdict(list))   # model -> patient -> [f1 per seed]
    for r in final:
        per_patient[r["model"]][r["fold"]].append(r["test"]["macro_f1"])
    comp = {}
    for m in MODELS:
        seeds_pooled, cls_f1, aurocs, mes_auc = [], defaultdict(list), defaultdict(list), []
        for s in SEEDS:
            y, yh, P, _, _ = pooled(rev, m, s)
            seeds_pooled.append(macro_f1(y, yh))
            for i, c in enumerate(CLASSES):
                cls_f1[c].append(f1_score(y == i, yh == i, zero_division=0))
                aurocs[c].append(roc_auc_score(y == i, P[:, i]))
        mes = [r["test"].get("mes_auroc") for r in final if r["model"] == m and r["test"].get("mes_auroc")]
        pp = np.array([np.mean(per_patient[m][p]) for p in patients])
        comp[m] = {
            "pooled_macro_f1_mean": float(np.mean(seeds_pooled)),
            "pooled_macro_f1_sd": float(np.std(seeds_pooled, ddof=1)),
            "per_class_f1": {c: float(np.mean(v)) for c, v in cls_f1.items()},
            "per_class_auroc": {c: float(np.mean(v)) for c, v in aurocs.items()},
            "per_patient_macro_f1": dict(zip(patients, pp.tolist())),
            "per_patient_mean": float(pp.mean()), "per_patient_sd": float(pp.std(ddof=1)),
            "per_patient_min": float(pp.min()), "per_patient_max": float(pp.max()),
            "mes_auroc_mean_over_folds": float(np.mean(mes)) if mes else None,
        }
    summary["lopo"] = comp

    # paired tests on per-patient macro-F1 (n = 13) + patient-bootstrap CI of pooled diff
    tests = {}
    pairs = [("logistic", "sage"), ("mlp", "sage"), ("mlpnbr", "sage"), ("gat", "sage"),
             ("mlpnbr", "mlp"), ("gat", "mlp"), ("gat", "logistic"), ("mlp", "logistic"),
             ("mlpnbr", "logistic")]
    rng = np.random.default_rng(0)
    for a, b in pairs:
        xa = np.array([np.mean(per_patient[a][p]) for p in patients])
        xb = np.array([np.mean(per_patient[b][p]) for p in patients])
        d = xa - xb
        t = stats.ttest_rel(xa, xb)
        w = stats.wilcoxon(xa, xb)
        lo, hi = bootstrap_diff(rev, a, b, args.n_boot, rng)
        tests[f"{a}_vs_{b}"] = {
            "mean_per_patient_diff": float(d.mean()),
            "pooled_diff": comp[a]["pooled_macro_f1_mean"] - comp[b]["pooled_macro_f1_mean"],
            "pooled_diff_boot95": [lo, hi],
            "t": float(t.statistic), "t_p": float(t.pvalue), "wilcoxon_p": float(w.pvalue),
            "a_better_patients": int((d > 0).sum()), "n": len(d),
        }
    summary["lopo_tests"] = tests

    # variance: patient vs seed (per model)
    var = {}
    for m in MODELS:
        arr = np.array([per_patient[m][p] for p in patients])  # 13 x 3
        between = arr.mean(1).var(ddof=1)
        within = arr.var(1, ddof=1).mean()
        var[m] = {"sd_between_patients": float(np.sqrt(between)), "sd_within_seed": float(np.sqrt(within)),
                  "ratio": float(between / within) if within > 1e-12 else None}
    summary["variance"] = var

    # selected hyperparameters
    chosen = json.loads((rev / "chosen_configs.json").read_text())
    cfgs = json.loads((rev / "search_configs.json").read_text())
    summary["chosen_configs"] = {m: {p: cfgs[m][i] for p, i in chosen[m].items()} for m in MODELS}

    # ablation / marker-removal sections need the whole extra queue; without it,
    # only the LOPO comparison is summarized (rerun once jobs_extra.jsonl is done)
    extra = [json.loads(l)["name"] for l in (rev / "jobs_extra.jsonl").read_text().splitlines() if l.strip()]
    extra_done = all((rev / "runs" / n).exists() for n in extra)
    summary["extra_complete"] = extra_done
    if not extra_done:
        log_missing = sum(not (rev / "runs" / n).exists() for n in extra)
        print(f"extra queue incomplete ({log_missing}/{len(extra)} missing): skipping ablations")
    else:
        ablation_and_markers(rev, summary, patients)

    homo = rev / "label_homophily.json"
    if homo.exists():
        summary["homophily"] = json.loads(homo.read_text())
    (rev / "revision_summary.json").write_text(json.dumps(summary, indent=1))
    make_figures(rev, figdir, summary, per_patient, patients, Path(args.data))
    print(json.dumps({k: summary[k] for k in ["lopo_tests", "variance", "marker_removal"] if k in summary}, indent=1))
    for m in MODELS:
        c = comp[m]
        print(f"{m:9s} pooled {c['pooled_macro_f1_mean']:.3f}±{c['pooled_macro_f1_sd']:.3f} "
              f"per-patient {c['per_patient_mean']:.3f} [{c['per_patient_min']:.2f}-{c['per_patient_max']:.2f}]")


def ablation_and_markers(rev, summary, patients):
    # ---------------- 2. ablations (seed 42, default hyperparameters)
    abl = load_runs(rev, "abl_")
    ab = defaultdict(list)
    for r in abl:
        key = r["name"].rsplit("_", 2)[0]   # e.g. abl_sage_L3_k6
        ab[key].append(r["test"]["macro_f1"])
    summary["ablation"] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "n": len(v)}
                           for k, v in sorted(ab.items())}

    # ---------------- 3. marker-gene removal (seed 42, default hyperparameters)
    tune = load_runs(rev, "tune_logistic_c4_")  # logistic default C = 0.1 == config index 4
    base = {"logistic": {r["fold"]: r["test"]["macro_f1"] for r in tune},
            "mlp": {r["fold"]: r["test"]["macro_f1"] for r in abl if r["name"].startswith("abl_mlp_L3_")},
            "mlpnbr": {r["fold"]: r["test"]["macro_f1"] for r in abl if r["name"].startswith("abl_mlpnbr_h1_k6_")},
            "sage": {r["fold"]: r["test"]["macro_f1"] for r in abl if r["name"].startswith("abl_sage_L3_k6_")}}
    drop = defaultdict(dict)
    for r in load_runs(rev, "drop"):
        n = int(r["name"].split("_")[0].replace("drop", ""))
        drop[(r["model"], n)][r["fold"]] = r["test"]["macro_f1"]
    dg = {}
    for m in ["logistic", "mlp", "mlpnbr", "sage"]:
        dg[m] = {"0": float(np.mean([base[m][p] for p in patients]))}
        for n in [100, 500]:
            dg[m][str(n)] = float(np.mean([drop[(m, n)][p] for p in patients]))
    # paired: does removing markers hurt graph-free models more than SAGE?
    gap = {}
    for n in [0, 100, 500]:
        get = (lambda m, p: base[m][p]) if n == 0 else (lambda m, p, n=n: drop[(m, n)][p])
        d = np.array([get("logistic", p) - get("sage", p) for p in patients])
        d2 = np.array([get("mlp", p) - get("sage", p) for p in patients])
        gap[str(n)] = {"logistic_minus_sage": float(d.mean()), "logistic_wins": int((d > 0).sum()),
                       "mlp_minus_sage": float(d2.mean()), "mlp_wins": int((d2 > 0).sum())}
    summary["marker_removal"] = {"per_patient_mean_macro_f1": dg, "gaps": gap}


# --------------------------------------------------------------------------- figures

def _style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def _save(fig, figdir, name):
    fig.savefig(figdir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figdir / f"{name}.svg", bbox_inches="tight")


def make_figures(rev, figdir, summary, per_patient, patients, data_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.linewidth": 0.8})

    # Fig A: LOPO comparison -- per-patient lines + pooled bars
    comp = summary["lopo"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 2.2]})
    ax = axes[0]
    xs = np.arange(len(MODELS))
    ax.bar(xs, [comp[m]["pooled_macro_f1_mean"] for m in MODELS],
           yerr=[comp[m]["pooled_macro_f1_sd"] for m in MODELS], capsize=2,
           color=[MODEL_COLORS[m] for m in MODELS], edgecolor="black", linewidth=0.5)
    for x, m in zip(xs, MODELS):
        top = comp[m]["pooled_macro_f1_mean"] + comp[m]["pooled_macro_f1_sd"]
        ax.text(x, top + 0.02, f"{comp[m]['pooled_macro_f1_mean']:.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(xs, [MODEL_NAMES[m].replace(" + ", "\n+ ") for m in MODELS], rotation=45, ha="right")
    ax.set_ylabel("Pooled LOPO macro-F1"); ax.set_ylim(0, 0.8); _style(ax)
    ax.set_title("A", loc="left", fontweight="bold")
    ax = axes[1]
    order = sorted(patients, key=lambda p: np.mean(per_patient["logistic"][p]))
    for m in MODELS:
        v = [np.mean(per_patient[m][p]) for p in order]
        ax.plot(range(len(order)), v, "o-", ms=3, lw=1, color=MODEL_COLORS[m], label=MODEL_NAMES[m])
    ax.set_xticks(range(len(order)), order, rotation=60, ha="right")
    ax.set_xlabel("Held-out patient"); ax.set_ylabel("Macro-F1 (mean of 3 seeds)")
    ax.set_ylim(0, 1); _style(ax); ax.legend(frameon=False, fontsize=6.5, ncol=3, loc="upper left")
    ax.set_title("B", loc="left", fontweight="bold")
    fig.tight_layout(); _save(fig, figdir, "FigR1_lopo_comparison"); plt.close(fig)

    if summary["extra_complete"]:
        ablation_figures(figdir, summary, plt)

    # Fig D: pooled confusion matrices (tuned thresholds), best graph-free vs GraphSAGE
    confusion_and_maps(rev, figdir, summary, comp, data_dir, plt)


def ablation_figures(figdir, summary, plt):
    # Fig B: ablations
    ab = summary["ablation"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax = axes[0]
    for k, mk in zip([6, 12, 18], ["o", "s", "^"]):
        ys = [ab[f"abl_sage_L{L}_k{k}"]["mean"] for L in [1, 2, 3]]
        ax.plot([1, 2, 3], ys, marker=mk, color="black", lw=1, ms=4, label=f"GraphSAGE, k = {k}",
                alpha={6: 1, 12: 0.65, 18: 0.35}[k])
    ax.plot([1, 2, 3], [ab[f"abl_mlp_L{L}"]["mean"] for L in [1, 2, 3]], "o--", color=MODEL_COLORS["mlp"],
            lw=1, ms=4, label="MLP (no graph)")
    ax.set_xticks([1, 2, 3]); ax.set_xlabel("Number of layers (= hops for GraphSAGE)")
    ax.set_ylabel("Mean per-patient macro-F1"); _style(ax); ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("A", loc="left", fontweight="bold")
    ax = axes[1]
    labels, vals, errs, cols = [], [], [], []
    labels.append("MLP\n(0 hops)"); vals.append(ab["abl_mlp_L3"]["mean"]); errs.append(0); cols.append(MODEL_COLORS["mlp"])
    for h in [1, 2]:
        for k in [6, 12, 18]:
            labels.append(f"{h} hop{'s' if h > 1 else ''}\nk={k}")
            vals.append(ab[f"abl_mlpnbr_h{h}_k{k}"]["mean"]); errs.append(0); cols.append(MODEL_COLORS["mlpnbr"])
    ax.bar(range(len(vals)), vals, color=cols, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(vals)), labels, fontsize=6.5)
    ax.set_ylabel("Mean per-patient macro-F1"); ax.set_ylim(0, max(vals) * 1.25); _style(ax)
    ax.set_title("B  MLP given averaged neighbor expression", loc="left", fontweight="bold", fontsize=8)
    fig.tight_layout(); _save(fig, figdir, "FigR2_ablation"); plt.close(fig)

    # Fig C: marker-gene removal
    dg = summary["marker_removal"]["per_patient_mean_macro_f1"]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for m in ["logistic", "mlp", "mlpnbr", "sage"]:
        ax.plot([0, 100, 500], [dg[m][k] for k in ["0", "100", "500"]], "o-", color=MODEL_COLORS[m],
                lw=1.2, ms=4, label=MODEL_NAMES[m])
    ax.set_xticks([0, 100, 500]); ax.set_xlabel("Most label-associated genes removed")
    ax.set_ylabel("Mean per-patient macro-F1"); _style(ax); ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout(); _save(fig, figdir, "FigR3_marker_removal"); plt.close(fig)


def confusion_and_maps(rev, figdir, summary, comp, data_dir, plt):
    from sklearn.metrics import confusion_matrix
    best_free = max(["logistic", "mlp", "mlpnbr"], key=lambda m: comp[m]["pooled_macro_f1_mean"])
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
    for ax, m in zip(axes, [best_free, "sage"]):
        cms = []
        for s in SEEDS:
            y, yh, _, _, _ = pooled(rev, m, s)
            cms.append(confusion_matrix(y, yh, labels=range(4), normalize="true"))
        cm = np.mean(cms, 0)
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(4), [LABELS[c] for c in CLASSES], rotation=45, ha="right")
        ax.set_yticks(range(4), [LABELS[c] for c in CLASSES])
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > 0.5 else "black")
        ax.set_title(MODEL_NAMES[m], fontsize=8); ax.set_xlabel("Predicted")
        if ax is axes[0]:
            ax.set_ylabel("True")
        else:
            ax.tick_params(labelleft=False)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Proportion of true class")
    _save(fig, figdir, "FigR4_confusion_pooled"); plt.close(fig)
    summary["confusion_best_graph_free"] = best_free

    # Fig E: spatial maps (LOPO, tuned thresholds, seed 42) for every section
    secs_meta = []
    preds = {}
    for m in [best_free, "sage"]:
        y, yh, P, S, _ = pooled(rev, m, 42)
        preds[m] = (S, yh)
    import pandas as pd
    for g_sec in sorted(set(preds["sage"][0].tolist())):
        secs_meta.append(g_sec)
    import torch
    graphs = {g.section: g for g in torch.load(data_dir / "graphs.pt", weights_only=False)}
    for sec in secs_meta:
        g = graphs[sec]
        xy = np.load(data_dir / f"{sec}_coords.npy")
        ytrue = g.niche_y.numpy()
        lab = ytrue >= 0
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
        panels = [("Silver-standard label", ytrue)]
        for m in [best_free, "sage"]:
            S, yh = preds[m]
            full = np.full(len(ytrue), -1)
            full[lab] = yh[S == sec]
            panels.append((f"{MODEL_NAMES[m]} (LOPO)", full))
        for ax, (title, yy) in zip(axes, panels):
            cols = [NICHE_COLORS["exclude"] if v < 0 else NICHE_COLORS[CLASSES[v]] for v in yy]
            ax.scatter(xy[:, 1], -xy[:, 0], c=cols, s=1.2, linewidths=0)
            ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=7)
        handles = [plt.Line2D([], [], marker="o", ls="", color=NICHE_COLORS[c], label=LABELS.get(c, c))
                   for c in CLASSES + ["exclude"]]
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=6.5,
                   labels=[LABELS.get(c, "unlabeled") for c in CLASSES + ["exclude"]])
        fig.suptitle(sec, fontsize=8)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig(figdir / f"maps_{sec.replace(' ', '_')}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    (rev / "revision_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
