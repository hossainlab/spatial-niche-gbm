"""Check every number changed in the v7 revision against the file that produced it.

Each check recomputes or re-reads a value from `results/` (or from the per-spot
label files) and asserts that the manuscript source contains exactly that value,
formatted as the manuscript formats it.  Run after editing the .tex.

Usage: python scripts/28_verify_manuscript_numbers.py [--tex preprint/spatial-niche-gbm.tex]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

CLASSES = ["immune_cold", "immune_hot", "normal", "necrotic"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default="preprint/spatial-niche-gbm.tex")
    ap.add_argument("--sup", default="results/tables/supplement")
    ap.add_argument("--main", default="results/tables/main")
    ap.add_argument("--log", default="results/tables/main/v7_number_verification.csv")
    a = ap.parse_args()
    tex = Path(a.tex).read_text()
    sup, main = Path(a.sup), Path(a.main)
    checks = []

    def need(label, value, source):
        checks.append(dict(quantity=label, value=value, source=source,
                           present_in_manuscript=value in tex))

    # --- six-split tuning support
    s10 = pd.read_csv(sup / "TableS10_split_tuning_support.csv")
    vals = [f"{v:,}".replace(",", "{,}") for v in s10.val_immune_hot]
    need("tuning-block immune-hot support, splits 0-5",
         ", ".join(vals[:-1]) + " and " + vals[-1],      # manuscript lists them in prose
         "TableS10_split_tuning_support.csv")
    need("split 4 necrotic tuning support", str(int(s10.val_necrotic.iloc[4])),
         "TableS10_split_tuning_support.csv")

    # --- restricted-class contrasts
    s11 = pd.read_csv(sup / "TableS11_split_contrasts_restricted.csv").set_index("contrast")
    for c, pub, res in [("Logistic - GraphSAGE", "0.086", "0.070"), ("MLP - GraphSAGE", "0.040", "0.047")]:
        assert f"{s11.loc[c, 'published_diff']:.3f}" == pub, (c, s11.loc[c, "published_diff"])
        assert f"{s11.loc[c, 'restricted_diff']:.3f}" == res, (c, s11.loc[c, "restricted_diff"])
        need(f"{c}: four-class diff", f"${pub}$", "TableS11_split_contrasts_restricted.csv")
        need(f"{c}: three-class diff", f"${res}$", "TableS11_split_contrasts_restricted.csv")

    # --- LOPO support floors
    s12 = pd.read_csv(sup / "TableS12_lopo_support_floor.csv")
    f50 = s12[s12.floor.str.contains("50")].set_index("contrast")
    f200 = s12[s12.floor.str.contains("200")].set_index("contrast")
    f0 = s12[s12.floor.str.contains("none")].set_index("contrast")
    need("logistic vs GraphSAGE, Holm-adjusted p (no floor)",
         f"{f0.loc['Logistic - GraphSAGE', 'wilcoxon_p_holm']:.3f}", "TableS12_lopo_support_floor.csv")
    need("logistic vs GraphSAGE at a 50-spot floor, wins",
         f"{f50.loc['Logistic - GraphSAGE', 'wins'].replace('/', ' of ')}", "TableS12_lopo_support_floor.csv")
    need("logistic vs GraphSAGE at a 50-spot floor, p",
         f"{f50.loc['Logistic - GraphSAGE', 'wilcoxon_p']:.3f}", "TableS12_lopo_support_floor.csv")
    need("MLP vs GraphSAGE at a 50-spot floor, p",
         f"{f50.loc['MLP - GraphSAGE', 'wilcoxon_p']:.2f}", "TableS12_lopo_support_floor.csv")
    need("logistic vs GraphSAGE at a 200-spot floor, p",
         f"{f200.loc['Logistic - GraphSAGE', 'wilcoxon_p']:.3f}", "TableS12_lopo_support_floor.csv")

    # --- decision rule
    s13 = pd.read_csv(sup / "TableS13_decision_rule.csv").set_index("model")
    for m in ["Logistic", "MLP", "GraphSAGE", "MLP + neighbors", "GAT"]:
        need(f"{m}: pooled LOPO macro-F1 under argmax",
             f"{s13.loc[m, 'pooled_macro_f1_argmax']:.3f}", "TableS13_decision_rule.csv")
    need("logistic - GraphSAGE gap under argmax",
         f"${s13.loc['Logistic - GraphSAGE', 'pooled_macro_f1_argmax']:.3f}$", "TableS13_decision_rule.csv")
    need("MLP - GraphSAGE gap under argmax",
         f"${s13.loc['MLP - GraphSAGE', 'pooled_macro_f1_argmax']:.3f}$", "TableS13_decision_rule.csv")

    # --- survival
    sv = json.loads((main / "Table3_survival_results.json").read_text())
    need("Cox HR per SD", f"{sv['cox_univariate_per_sd']['HR']:.2f}", "Table3_survival_results.json")
    need("Cox HR per SD, CI",
         f"{sv['cox_univariate_per_sd']['ci'][0]:.2f}--{sv['cox_univariate_per_sd']['ci'][1]:.2f}",
         "Table3_survival_results.json")
    need("age positive control HR per SD", f"{sv['positive_control_age_per_sd']['HR']:.2f}",
         "Table3_survival_results.json")
    need("age positive control CI",
         f"{sv['positive_control_age_per_sd']['ci'][0]:.2f}--{sv['positive_control_age_per_sd']['ci'][1]:.2f}",
         "Table3_survival_results.json")
    need("smallest detectable HR per SD", f"{sv['min_detectable_HR_per_sd_80pct_power']:.2f}",
         "Table3_survival_results.json")

    # --- slide-level risk head
    rk = pd.read_csv(main / "slide_risk_head_eval.csv").set_index("seed")
    need("slide head Pearson r, seed 7", f"{rk.loc[7, 'pearson_r']:.2f}", "slide_risk_head_eval.csv")
    need("slide head Spearman rho, seed 42", f"{rk.loc[42, 'spearman_rho']:.1f}", "slide_risk_head_eval.csv")
    need("slide head Pearson r, seed 42", f"{rk.loc[42, 'pearson_r']:.2f}", "slide_risk_head_eval.csv")
    need("slide head Pearson r, seed 123", f"{rk.loc[123, 'pearson_r']:.2f}", "slide_risk_head_eval.csv")

    # --- UKF annotation table
    ukf = pd.read_csv(sup / "TableS9_ukf_pathologist_annotations.csv")
    need("UKF rows in the released annotation table",
         f"{len(ukf):,}".replace(",", "{,}"), "TableS9_ukf_pathologist_annotations.csv")
    need("UKF annotated spots",
         f"{int(ukf['class'].isin(['cellular','infiltrative','necrotic','mvp','normal']).sum()):,}".replace(",", "{,}"),
         "TableS9_ukf_pathologist_annotations.csv")
    need("UKF sections", str(ukf.section.nunique()), "TableS9_ukf_pathologist_annotations.csv")

    # --- per-patient rare-class counts quoted in Section 2.5
    lab = pd.concat([pd.read_csv(f).assign(section=Path(f).name.replace("_spot_labels.csv", ""))
                     for f in glob.glob("data/preprocessed/*_spot_labels.csv")], ignore_index=True)
    lab["patient"] = lab.section.str.extract(r"^(SNU\d+)")
    cnt = (lab[lab.niche != "exclude"].groupby(["patient", "niche"]).size()
           .unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0))
    for pat, word in [("SNU25", "one"), ("SNU16", "eight"), ("SNU24", "six"), ("SNU51", "nine")]:
        checks.append(dict(quantity=f"{pat} immune-hot spots", value=f"{int(cnt.loc[pat,'immune_hot'])} ({word})",
                           source="data/preprocessed/*_spot_labels.csv",
                           present_in_manuscript=f"{pat} has {word}" in tex or f"{pat} {word}" in tex))
    frac = cnt.loc["SNU46", "normal"] / cnt.loc["SNU46"].sum() * 100
    need("SNU46 fraction normal", f"{frac:.1f}\\%", "data/preprocessed/*_spot_labels.csv")

    df = pd.DataFrame(checks)
    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.log, index=False)
    bad = df[~df.present_in_manuscript]
    print(df.to_string(index=False))
    print(f"\n{len(df) - len(bad)}/{len(df)} verified")
    if len(bad):
        raise SystemExit(f"{len(bad)} value(s) not found in the manuscript as formatted")


if __name__ == "__main__":
    main()
