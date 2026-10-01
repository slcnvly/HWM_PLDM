# Boundary-signal study, item: trajectory-level bootstrap 95% CI (n=10000)
# for Metric A's overall F1 (all 12 main-sample candidates) and AUROC
# (7 signals), reusing results_metric_a.json's per-episode raw arrays.
import json
import os

import numpy as np

HERE = os.path.dirname(__file__)
N_BOOT = 10000


def bootstrap_mean_ci(a, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(a)
    n = len(a)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = a[idx].mean()
    return float(a.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    with open(os.path.join(HERE, "results_metric_a.json")) as f:
        data = json.load(f)
    prf1_raw = data["prf1_raw"]
    auroc_ap_raw = data["auroc_ap_raw"]

    results = {"f1_overall": {}, "auroc": {}}
    print("=== Overall F1, 95% CI (n=10000 bootstrap) ===")
    for name in prf1_raw:
        vals = prf1_raw[name]["overall"]["f1"]
        mean, lo, hi = bootstrap_mean_ci(vals)
        results["f1_overall"][name] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
        print(f"{name}: F1={mean:.4f}, 95% CI [{lo:.4f}, {hi:.4f}]")

    print("\n=== AUROC, 95% CI (n=10000 bootstrap) ===")
    for name in auroc_ap_raw:
        vals = auroc_ap_raw[name]["auroc"]
        if not vals:
            continue
        mean, lo, hi = bootstrap_mean_ci(vals)
        results["auroc"][name] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
        print(f"{name}: AUROC={mean:.4f}, 95% CI [{lo:.4f}, {hi:.4f}]")

    with open(os.path.join(HERE, "results_bootstrap_ci_metric_a.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote results_bootstrap_ci_metric_a.json")


if __name__ == "__main__":
    main()
