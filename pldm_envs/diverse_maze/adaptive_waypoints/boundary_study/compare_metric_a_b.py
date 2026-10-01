# Boundary-signal study: compare Metric A (event alignment) and Metric B
# (latent reconstruction) rankings, all at min_seg=8, same 300-episode
# sample (12 candidates) + signal_16v2 (its own 50-episode sample, noted
# separately). Spearman correlation + flag large rank disagreements.
import json
import os

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(__file__)


def main():
    with open(os.path.join(HERE, "results_metric_a.json")) as f:
        metric_a = json.load(f)["summary"]
    with open(os.path.join(HERE, "results_stage2_predictor_free.json")) as f:
        metric_b_stage2 = json.load(f)["summary"]
    with open(os.path.join(HERE, "results_stage2_amendment3.json")) as f:
        metric_b_amend3 = json.load(f)
    with open(os.path.join(HERE, "results_min_seg_sweep.json")) as f:
        metric_b_sweep = json.load(f)["summary"]["8"]  # post-bugfix, min_seg=8

    # Metric B (% improvement over fixed) for the 12 main-sample candidates,
    # min_seg=8 versions specifically for top_down/bottom_up.
    metric_b_improvement = {
        "signal_6": metric_b_stage2["signal_6_bocpd"]["improvement_over_fixed"],
        "signal_7": metric_b_stage2["signal_7_speed"]["improvement_over_fixed"],
        "signal_8": metric_b_stage2["signal_8_direction"]["improvement_over_fixed"],
        "signal_10": metric_b_stage2["signal_10_action_delta"]["improvement_over_fixed"],
        "signal_11": metric_b_amend3["signal_11_curvature"]["improvement_over_fixed"],
        "signal_12": metric_b_amend3["signal_12_norm_curvature"]["improvement_over_fixed"],
        "signal_13": metric_b_amend3["signal_13_chord_dev"]["improvement_over_fixed"],
        # bottom_up/top_down pulled from the POST-BUGFIX min_seg=8 sweep
        # (results_stage2_amendment3.json's bottom_up_minseg8 is stale --
        # computed before the bottom_up_merge min_seg bug was fixed, see
        # PROGRESS.md "bottom_up_merge bug: found, fixed, full sweep rerun").
        "top_down": metric_b_sweep["top_down"]["improvement_over_fixed"],
        "bottom_up": metric_b_sweep["bottom_up"]["improvement_over_fixed"],
        "fixed": 0.0,
        "random": metric_b_stage2["random"]["improvement_over_fixed"],
        "oracle": metric_b_stage2["oracle"]["improvement_over_fixed"],
    }

    metric_a_f1 = {name: metric_a["prf1"][name]["overall"]["f1"] for name in metric_b_improvement}

    names = list(metric_b_improvement.keys())
    a_vals = [metric_a_f1[n] for n in names]
    b_vals = [metric_b_improvement[n] for n in names]

    a_rank = {n: r for r, n in enumerate(sorted(names, key=lambda n: -metric_a_f1[n]), start=1)}
    b_rank = {n: r for r, n in enumerate(sorted(names, key=lambda n: -metric_b_improvement[n]), start=1)}

    rho, p = spearmanr(a_vals, b_vals)

    print(f"{'name':<12} {'MetricA_F1':>10} {'A_rank':>7} {'MetricB_%improve':>17} {'B_rank':>7} {'rank_diff':>10}")
    rows = []
    for n in sorted(names, key=lambda n: a_rank[n]):
        diff = abs(a_rank[n] - b_rank[n])
        rows.append((n, metric_a_f1[n], a_rank[n], metric_b_improvement[n] * 100, b_rank[n], diff))
        print(f"{n:<12} {metric_a_f1[n]:>10.4f} {a_rank[n]:>7} {metric_b_improvement[n]*100:>16.1f}% {b_rank[n]:>7} {diff:>10}")

    print(f"\nSpearman rho (Metric A F1 vs Metric B % improvement) = {rho:.4f} (p={p:.4f})")

    large_disagreements = [r for r in rows if r[5] >= 5]
    print(f"\nLarge rank disagreements (|rank_diff|>=5): {len(large_disagreements)}")
    for r in large_disagreements:
        print(f"  {r[0]}: A_rank={r[2]}, B_rank={r[4]}, diff={r[5]}")

    out = {
        "spearman_rho": float(rho), "spearman_p": float(p),
        "table": [{"name": r[0], "metric_a_f1": r[1], "a_rank": r[2], "metric_b_pct_improve": r[3], "b_rank": r[4], "rank_diff": r[5]} for r in rows],
    }
    with open(os.path.join(HERE, "results_metric_a_vs_b.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results_metric_a_vs_b.json")


if __name__ == "__main__":
    main()
