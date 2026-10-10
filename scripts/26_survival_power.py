"""Re-express the TCGA-GBM survival test on a per-SD scale, add a positive
control and bound the effect size the cohort could have detected.

The v6 manuscript reported a hazard ratio per unit of the niche-signature score
and described the cohort as providing "moderate power", which is not falsifiable
as written.  This script recomputes the same model with the score standardised,
runs age through the identical pipeline as a positive control, and computes the
smallest per-SD hazard ratio detectable at 80% power and two-sided alpha = 0.05
from the observed number of events (Schoenfeld).  It rewrites
results/tables/main/Table3_survival_results.json.

Usage: python scripts/26_survival_power.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from scipy.stats import norm


def cox(df, cols):
    m = CoxPHFitter().fit(df[["os_days", "os_event"] + cols],
                          duration_col="os_days", event_col="os_event")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/tables/supplement/TableS4_tcga_scores_clinical.csv")
    ap.add_argument("--out", default="results/tables/main/Table3_survival_results.json")
    a = ap.parse_args()

    d = pd.read_csv(a.scores)
    n, events = len(d), int(d.os_event.sum())
    sd_risk, sd_age = float(d.risk.std()), float(d.age.std())
    d["risk_z"] = (d.risk - d.risk.mean()) / sd_risk
    d["age_z"] = (d.age - d.age.mean()) / sd_age

    uni_unit = cox(d, ["risk"])
    uni_z = cox(d, ["risk_z"])
    adj = cox(d, ["risk_z", "age_z"])
    pos = cox(d, ["age_z"])

    ci = float(concordance_index(d.os_days, -uni_z.predict_partial_hazard(d), d.os_event))
    # median split exactly as in the Kaplan-Meier figure (scripts/07_make_figures.py)
    km_df = d.dropna(subset=["os_days", "risk"])
    km_df = km_df[km_df.os_days > 0]
    hi = km_df.risk > km_df.risk.median()
    lr = logrank_test(km_df.os_days[hi], km_df.os_days[~hi],
                      km_df.os_event[hi], km_df.os_event[~hi])
    km_med = {}
    for lbl, grp in [("high", km_df[hi]), ("low", km_df[~hi])]:
        km_med[lbl] = float(KaplanMeierFitter().fit(grp.os_days, grp.os_event).median_survival_time_)
        km_med[f"n_{lbl}"] = int(len(grp))

    # Schoenfeld: log(HR) detectable = (z_{1-a/2} + z_{1-b}) / sqrt(d * var(x)); var(z) = 1
    hr_min = float(np.exp((norm.ppf(0.975) + norm.ppf(0.80)) / np.sqrt(events)))

    dedup = d.sort_values("sample").drop_duplicates("case")
    ded = cox(dedup, ["risk_z"])
    hi2 = dedup.risk > dedup.risk.median()
    lr2 = logrank_test(dedup.os_days[hi2], dedup.os_days[~hi2],
                       dedup.os_event[hi2], dedup.os_event[~hi2])

    def row(m, c):
        s = m.summary.loc[c]
        return dict(HR=round(float(s["exp(coef)"]), 3), p=round(float(s["p"]), 4),
                    ci=[round(float(s["exp(coef) lower 95%"]), 3),
                        round(float(s["exp(coef) upper 95%"]), 3)])

    out = {
        "n": n, "n_patients": int(d.case.nunique()), "events": events,
        "score_sd": round(sd_risk, 4), "age_sd": round(sd_age, 4),
        "cox_univariate": row(uni_unit, "risk"),
        "cox_univariate_per_sd": row(uni_z, "risk_z"),
        "cox_age_adjusted_per_sd": row(adj, "risk_z"),
        "positive_control_age_per_sd": row(pos, "age_z"),
        "c_index": round(ci, 4),
        "km_logrank_p": round(float(lr.p_value), 4),
        "km_median_os_high": km_med["high"], "km_median_os_low": km_med["low"],
        "km_n_high": km_med["n_high"], "km_n_low": km_med["n_low"],
        "min_detectable_HR_per_sd_80pct_power": round(hr_min, 3),
        "deduplicated_one_aliquot_per_patient": {
            "n": len(dedup), "events": int(dedup.os_event.sum()),
            "cox_univariate_per_sd": row(ded, "risk_z"),
            "km_logrank_p": round(float(lr2.p_value), 4)},
    }
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
