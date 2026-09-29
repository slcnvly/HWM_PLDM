# Boundary-signal study, item 4: trajectory-level bootstrap 95% CIs
# (n=10000, percentile method -- same convention PREREGISTRATION.md SS6
# and confidence_intervals.py already use, though that file's own
# functions are for binary success/failure outcomes and don't apply to
# these continuous per-episode Metric B scores, so this is a small
# standalone implementation, not a reuse of that file's functions).
#
# Run AFTER min_seg_sweep.py has written results_min_seg_sweep.json (needs
# its saved per-episode score arrays).
import json
import os

import numpy as np

HERE = os.path.dirname(__file__)
N_BOOT = 10000


def bootstrap_diff_ci(a, b, n_boot=N_BOOT, seed=0):
    """a, b: (N,) paired per-episode scores (SAME episodes, e.g. condition A
    vs condition B on the same 100-episode sample). Returns (mean_diff,
    ci_lo, ci_hi) for mean(a)-mean(b), resampling episode indices jointly
    (paired bootstrap, since a[i] and b[i] are the same episode)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    return float((a - b).mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


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
    with open(os.path.join(HERE, "results_min_seg_sweep.json")) as f:
        data = json.load(f)
    per_ep = data["per_episode_scores"]

    results = {}

    print("=== Mean Metric B, 95% CI (n=10000 bootstrap) ===")
    for ms in ("8", "1"):
        for name in ("bottom_up", "fixed"):
            vals = np.array(per_ep[ms][name])
            mean, lo, hi = bootstrap_mean_ci(vals)
            results[f"{name}_minseg{ms}"] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
            print(f"{name} (min_seg={ms}): mean={mean:.5f}, 95% CI [{lo:.5f}, {hi:.5f}]")

    print("\n=== Paired difference (bottom_up - fixed), 95% CI ===")
    for ms in sorted(per_ep.keys(), key=int):
        bu = np.array(per_ep[ms]["bottom_up"])
        fx = np.array(per_ep[ms]["fixed"])
        diff, lo, hi = bootstrap_diff_ci(bu, fx)
        significant = not (lo <= 0 <= hi)
        results[f"bottom_up_minus_fixed_minseg{ms}"] = {"diff": diff, "ci_lo": lo, "ci_hi": hi, "significant_at_5pct": significant}
        print(f"min_seg={ms}: bottom_up - fixed = {diff:.5f}, 95% CI [{lo:.5f}, {hi:.5f}] {'(significant, CI excludes 0)' if significant else '(NOT significant, CI includes 0)'}")

    with open(os.path.join(HERE, "results_bootstrap_ci.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote results_bootstrap_ci.json")


if __name__ == "__main__":
    main()
