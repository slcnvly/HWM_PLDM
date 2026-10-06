# Metric A before (raw inputs, *_v1.json) vs after the input-normalization fix.
# F1 / AUROC per candidate (original labels), F1-minus-random with paired bootstrap
# (original and coarse-collapsed labels), Spearman(Metric A F1, Metric B) using the
# corrected Metric B for every candidate, and the dp-smoothing stability check.
import json
import os

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["signal_6", "signal_7", "signal_8", "signal_10", "signal_11", "signal_12", "signal_13",
         "top_down", "bottom_up", "fixed", "random", "oracle"]


def j(n):
    return json.load(open(os.path.join(HERE, n)))


def metric_b(sp, am):
    s = sp["summary"]
    return {"signal_6": s["signal_6_bocpd"]["improvement_over_fixed"], "signal_7": s["signal_7_speed"]["improvement_over_fixed"],
            "signal_8": s["signal_8_direction"]["improvement_over_fixed"], "signal_10": s["signal_10_action_delta"]["improvement_over_fixed"],
            "signal_11": am["signal_11_curvature"]["improvement_over_fixed"], "signal_12": am["signal_12_norm_curvature"]["improvement_over_fixed"],
            "signal_13": am["signal_13_chord_dev"]["improvement_over_fixed"], "top_down": am["top_down_minseg8"]["improvement_over_fixed"],
            "bottom_up": am["bottom_up_minseg8"]["improvement_over_fixed"], "fixed": 0.0,
            "random": s["random"]["improvement_over_fixed"], "oracle": s["oracle"]["improvement_over_fixed"]}


def boot_vs_random(prf1_raw, label_set=None):
    p = prf1_raw[label_set] if label_set else prf1_raw
    r = np.array(p["random"]["overall"]["f1"])
    idx = np.random.default_rng(0).integers(0, len(r), (10000, len(r)))
    out = {}
    for n in NAMES:
        if n == "random":
            continue
        x = np.array(p[n]["overall"]["f1"]) - r
        b = x[idx].mean(1)
        out[n] = [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    return out


def main():
    res = {}
    for tag, ma, mc, s16, sp, am in (
        ("v1_raw_inputs", "results_metric_a_v1.json", "results_metric_a_coarse_v1.json", "results_metric_a_signal16v2_v1.json",
         "results_stage2_predictor_free_v1.json", "results_stage2_amendment3_v1.json"),
        ("fixed", "results_metric_a.json", "results_metric_a_coarse.json", "results_metric_a_signal16v2.json",
         "results_stage2_predictor_free.json", "results_stage2_amendment3.json"),
    ):
        A, C, S = j(ma), j(mc), j(s16)
        B = metric_b(j(sp), j(am))
        f1 = {n: A["summary"]["prf1"][n]["overall"]["f1"] for n in NAMES}
        au = {n: A["summary"]["auroc_ap"][n]["auroc_mean"] for n in NAMES if n in A["summary"]["auroc_ap"]}
        f1c = {n: C["summary"]["prf1"]["coarse_collapsed"][n]["overall"]["f1"] for n in NAMES}
        rho, p = spearmanr([f1[n] for n in NAMES], [B[n] for n in NAMES])
        rhoc, pc = spearmanr([f1c[n] for n in NAMES], [B[n] for n in NAMES])
        res[tag] = {
            "f1_original_labels": f1, "auroc_original_labels": au,
            "f1_coarse_collapsed": f1c,
            "f1_minus_random_original": boot_vs_random(A["prf1_raw"]),
            "f1_minus_random_coarse_collapsed": boot_vs_random(C["prf1_raw"], "coarse_collapsed"),
            "metric_b_improvement": B,
            "spearman_f1_vs_b_original": [float(rho), float(p)],
            "spearman_f1_vs_b_coarse_collapsed": [float(rhoc), float(pc)],
            "signal_16v2": {k: {"f1": S["summary"]["prf1"][k]["overall"]["f1"], "auroc": S["summary"]["auroc_ap"][k]["auroc_mean"]}
                            for k in S["summary"]["prf1"]},
            "dp_smoothing": C["summary"]["smoothing"],
        }
    res["note"] = ("v1 Spearman uses the raw-input Metric B of the same files (bottom_up/top_down from amendment3_v1, i.e. the "
                   "pre-merge-fix values), so the v1 rho here differs from the -0.769 in RESULTS.md SS8, which used the min_seg sweep.")
    json.dump(res, open(os.path.join(HERE, "results_metric_a_bugfix_comparison.json"), "w"), indent=2)
    for tag in ("v1_raw_inputs", "fixed"):
        r = res[tag]
        print(tag, "rho(F1,B) orig", np.round(r["spearman_f1_vs_b_original"], 3), "coarse", np.round(r["spearman_f1_vs_b_coarse_collapsed"], 3))
        for n in NAMES:
            m = r["f1_minus_random_original"].get(n)
            mc = r["f1_minus_random_coarse_collapsed"].get(n)
            print(f"  {n:10s} F1 {r['f1_original_labels'][n]:.3f} AUROC {r['auroc_original_labels'].get(n, float('nan')):.3f} "
                  f"F1-rand {m[0]:+.3f}[{m[1]:+.3f},{m[2]:+.3f}] coarse {mc[0]:+.3f}[{mc[1]:+.3f},{mc[2]:+.3f}]" if m else f"  {n} F1 {r['f1_original_labels'][n]:.3f}")
        print("  16v2", r["signal_16v2"], "dp smoothing", {k: round(v, 3) for k, v in r["dp_smoothing"].items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
