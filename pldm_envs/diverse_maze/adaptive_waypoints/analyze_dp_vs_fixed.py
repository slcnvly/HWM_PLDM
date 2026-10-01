# SS9 analysis: dp_segmentation-cache fine-tune vs the SS6b fixed-interval
# control on the same 120 hard instances. The control was re-evaluated
# (experiments/kaggle_fixed_control_reeval) to obtain per-trial outcomes;
# historic SS6b aggregates are shown alongside for reference.
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest, mannwhitneyu, norm, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))


def wilson(k, n, alpha=0.05):
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def load(path, key):
    d = json.load(open(path))[key]
    s, t = d["per_trial_success"], d["per_trial_steps"]
    assert s is not None and len(s) == 120, f"{path}: per-trial data missing"
    return np.array(s, dtype=int), np.array(t, dtype=float)


def main(dp_path, fixed_path):
    dp_s, dp_t = load(dp_path, "dp_segmentation_minseg8_finetuned")
    fx_s, fx_t = load(fixed_path, "fixed_stride_finetuned_reeval")
    rows = {
        "baseline (SS6b)": (96, 120, 169.6),
        "fixed-interval control (SS6b, historic)": (110, 120, 169.5),
        "adaptive signal-1 min_seg=8 (SS6b)": (111, 120, 155.1),
        "fixed-interval control (re-eval, same instances)": (int(fx_s.sum()), 120, float(fx_t[fx_s == 1].mean())),
        "adaptive dp_segmentation min_seg=8 (new)": (int(dp_s.sum()), 120, float(dp_t[dp_s == 1].mean())),
    }
    table = {name: {"k": k, "n": n, "rate": k / n, "wilson95": wilson(k, n), "avg_steps_successes": st}
             for name, (k, n, st) in rows.items()}

    dp_only = int(((dp_s == 1) & (fx_s == 0)).sum())
    fx_only = int(((dp_s == 0) & (fx_s == 1)).sum())
    both = int(((dp_s == 1) & (fx_s == 1)).sum())
    neither = int(((dp_s == 0) & (fx_s == 0)).sum())
    mcn_p = binomtest(dp_only, dp_only + fx_only, 0.5).pvalue if dp_only + fx_only > 0 else 1.0

    a, b = dp_t[dp_s == 1], fx_t[fx_s == 1]
    mw = mannwhitneyu(a, b, alternative="two-sided")
    both_mask = (dp_s == 1) & (fx_s == 1)
    paired_diff = dp_t[both_mask] - fx_t[both_mask]

    out = {
        "table": table,
        "mcnemar_dp_vs_fixed_reeval": {
            "dp_success_fixed_fail": dp_only, "dp_fail_fixed_success": fx_only,
            "both_success": both, "both_fail": neither, "exact_p_two_sided": float(mcn_p),
        },
        "mann_whitney_steps_successes": {
            "n_dp": int(len(a)), "n_fixed": int(len(b)),
            "median_dp": float(np.median(a)), "median_fixed": float(np.median(b)),
            "U": float(mw.statistic), "p_two_sided": float(mw.pvalue),
        },
        "paired_steps_on_both_success": {
            "n": int(both_mask.sum()), "mean_diff_dp_minus_fixed": float(paired_diff.mean()),
            "median_diff": float(np.median(paired_diff)),
            "n_dp_faster": int((paired_diff < 0).sum()), "n_fixed_faster": int((paired_diff > 0).sum()),
            "wilcoxon_p_two_sided": float(wilcoxon(paired_diff).pvalue),
        },
        "sanity": {"max_success_steps": float(max(a.max(), b.max())), "fail_steps_unique": sorted(set(np.concatenate([dp_t[dp_s == 0], fx_t[fx_s == 0]]).tolist()))},
    }
    json.dump(out, open(os.path.join(HERE, "results_hard_dp_vs_fixed_stats.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
