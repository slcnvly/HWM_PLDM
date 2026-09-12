"""
95% confidence intervals (binomial + bootstrap) for the three hard-difficulty
(D 13-16) planning-success conditions, and pairwise significance between them.
Computed at both n=40 (original curated starts_targets_13_16.pt, seed=42) and
n=120 (n=40 + 80 new trials, seed=20260910, combined per RESULTS.md SS6b).

Also computed (RESULTS.md SS7): a fourth "n120_l1adaptive" scale crossing the
two checkpoints that got a second hard-difficulty eval with error-adaptive L1
resource allocation (baseline, adaptive_minseg8) against their existing
fixed-allocation SS6b numbers -- 4 conditions, all 6 pairwise comparisons,
with the two same-checkpoint pairs (fixed vs. error-adaptive) reported first
and most prominently since those are the actual question this experiment
asks; the cross-checkpoint pairs are secondary context.

Run: python confidence_intervals.py
Writes confidence_intervals_results.json alongside this script.

Binomial CI: Wilson score interval (better-behaved than the normal
approximation at n=40 and success rates away from 0.5 -- doesn't produce
bounds outside [0,1] and doesn't assume a symmetric distribution).

Bootstrap CI: percentile bootstrap on the raw pass/fail outcomes, 10000
resamples, for a distribution-free cross-check on the same data.

Pairwise significance: two-sided permutation test using each condition's
raw outcome vector (assembled as k successes among n trials, order doesn't
matter for a difference-of-proportions permutation test), 10000 permutations,
plus a Wilson CI on the *difference* in proportions (Newcombe's method) as a
second check. A condition's own trial-level pass/fail sequence isn't
available (only k/n survived from the original run) -- both tests below are
computed from k/n, which is what a permutation test on grouped binomial
counts amounts to; noted explicitly rather than silently assumed.
"""
import itertools
import json
import math
from dataclasses import dataclass, asdict

import numpy as np

RNG = np.random.default_rng(0)
N_BOOTSTRAP = 10_000
N_PERMUTATIONS = 10_000

# ---- data: hard difficulty (D 13-16) -----------------------------------
CONDITIONS_N40 = {
    "baseline": {"k": 33, "n": 40, "label": "baseline (pretrained, no fine-tune)"},
    "adaptive_minseg8": {"k": 37, "n": 40, "label": "adaptive min_seg=8 (2-epoch fine-tune)"},
    "fixed_stride_finetuned": {"k": 35, "n": 40, "label": "fixed-stride, 2-epoch fine-tune (control)"},
}

# n=120 = n40 above + 80 new trials (seed=20260910, disjoint start/target
# instances from the original 40) per RESULTS.md SS6b / results_hard_n120_final.json
CONDITIONS_N120 = {
    "baseline": {"k": 96, "n": 120, "label": "baseline (pretrained, no fine-tune)"},
    "adaptive_minseg8": {"k": 111, "n": 120, "label": "adaptive min_seg=8 (2-epoch fine-tune)"},
    "fixed_stride_finetuned": {"k": 110, "n": 120, "label": "fixed-stride, 2-epoch fine-tune (control)"},
}

# SS7: baseline and adaptive_minseg8 each re-evaluated at hard difficulty,
# n=120 (same 120 trials as SS6b), with error-adaptive L1 resource
# allocation instead of fixed -- from results_hard_l1adaptive_final.json.
CONDITIONS_N120_L1ADAPTIVE = {
    "baseline": {"k": 96, "n": 120, "label": "baseline, fixed L1 allocation (SS6b)"},
    "baseline_error_adaptive": {"k": 96, "n": 120, "label": "baseline, error-adaptive L1 allocation (SS7)"},
    "adaptive_minseg8": {"k": 111, "n": 120, "label": "adaptive min_seg=8, fixed L1 allocation (SS6b)"},
    "adaptive_minseg8_error_adaptive": {"k": 115, "n": 120, "label": "adaptive min_seg=8, error-adaptive L1 allocation (SS7)"},
}

# Reported first/most prominently within the n120_l1adaptive scale -- these
# are the pairs this experiment actually asks about (same checkpoint, fixed
# vs. error-adaptive L1 allocation). The other 4 of the 6 total pairs (cross-
# checkpoint) are still computed, just printed/reported after.
PRIORITY_PAIRS_L1ADAPTIVE = [
    ("baseline", "baseline_error_adaptive"),
    ("adaptive_minseg8", "adaptive_minseg8_error_adaptive"),
]

SCALES = {"n40": CONDITIONS_N40, "n120": CONDITIONS_N120}


@dataclass
class CIResult:
    label: str
    k: int
    n: int
    rate: float
    wilson_95: tuple
    bootstrap_95: tuple


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def bootstrap_interval(k: int, n: int, n_boot: int = N_BOOTSTRAP) -> tuple:
    """Percentile bootstrap CI: resample n Bernoulli outcomes with the
    observed rate as the population, n_boot times, take the 2.5/97.5
    percentiles of the resampled proportion."""
    outcomes = np.array([1] * k + [0] * (n - k))
    resampled_rates = RNG.choice(outcomes, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(resampled_rates, [2.5, 97.5])
    return (float(lo), float(hi))


def newcombe_diff_interval(k1: int, n1: int, k2: int, n2: int, z: float = 1.959963984540054) -> tuple:
    """95% CI for (p1 - p2) via Newcombe's method (combines each side's
    Wilson interval -- standard approach for a difference of two independent
    binomial proportions without assuming normality)."""
    l1, u1 = wilson_interval(k1, n1, z)
    l2, u2 = wilson_interval(k2, n2, z)
    p1, p2 = k1 / n1, k2 / n2
    lo = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (lo, hi)


def permutation_test(k1: int, n1: int, k2: int, n2: int, n_perm: int = N_PERMUTATIONS) -> float:
    """Two-sided permutation test on grouped binomial counts: pool the
    n1+n2 pass/fail outcomes, repeatedly re-split into groups of size
    n1/n2, and ask how often the resampled |difference in proportions|
    meets or exceeds the observed one."""
    pooled = np.array([1] * k1 + [0] * (n1 - k1) + [1] * k2 + [0] * (n2 - k2))
    observed_diff = abs(k1 / n1 - k2 / n2)
    n_total = n1 + n2
    count_ge = 0
    for _ in range(n_perm):
        RNG.shuffle(pooled)
        p1 = pooled[:n1].mean()
        p2 = pooled[n1:].mean()
        if abs(p1 - p2) >= observed_diff - 1e-12:
            count_ge += 1
    return count_ge / n_perm


def _report_pair(pairwise: dict, conditions: dict, name_a: str, name_b: str):
    a, b = conditions[name_a], conditions[name_b]
    diff = a["k"] / a["n"] - b["k"] / b["n"]
    ci = newcombe_diff_interval(a["k"], a["n"], b["k"], b["n"])
    p = permutation_test(a["k"], a["n"], b["k"], b["n"])
    sig = ci[0] > 0 or ci[1] < 0  # CI excludes 0
    key = f"{name_a}_vs_{name_b}"
    pairwise[key] = {
        "diff": diff,
        "newcombe_95_ci": ci,
        "permutation_p_two_sided": p,
        "significant_at_0.05": bool(sig),
    }
    label = f"{name_a} vs {name_b}"
    print(
        f"{label:<48} {diff:>+7.3f}   [{ci[0]:+.3f}, {ci[1]:+.3f}]      "
        f"{p:>18.4f}  {'yes' if sig else 'no'}"
    )


def run_scale(conditions: dict, priority_pairs: list = None) -> tuple:
    results = {}
    print(f"{'condition':<28} {'k/n':>8} {'rate':>8}   {'wilson 95% CI':<22} {'bootstrap 95% CI':<22}")
    for name, c in conditions.items():
        k, n = c["k"], c["n"]
        w = wilson_interval(k, n)
        b = bootstrap_interval(k, n)
        res = CIResult(label=c["label"], k=k, n=n, rate=k / n, wilson_95=w, bootstrap_95=b)
        results[name] = asdict(res)
        print(
            f"{name:<28} {k:>3}/{n:<4} {k/n:>7.3f}   "
            f"[{w[0]:.3f}, {w[1]:.3f}]        [{b[0]:.3f}, {b[1]:.3f}]"
        )

    pairwise = {}
    all_pairs = list(itertools.combinations(conditions.keys(), 2))

    if priority_pairs:
        print(f"\n--- priority pairs ---")
        print(f"{'pair':<48} {'diff':>7}   {'newcombe 95% CI':<20} {'perm. p (2-sided)':>18}  significant@.05")
        for name_a, name_b in priority_pairs:
            _report_pair(pairwise, conditions, name_a, name_b)
        remaining_pairs = [p for p in all_pairs if p not in priority_pairs and (p[1], p[0]) not in priority_pairs]
        if remaining_pairs:
            print(f"\n--- other pairs ---")
            print(f"{'pair':<48} {'diff':>7}   {'newcombe 95% CI':<20} {'perm. p (2-sided)':>18}  significant@.05")
            for name_a, name_b in remaining_pairs:
                _report_pair(pairwise, conditions, name_a, name_b)
    else:
        print(f"\n{'pair':<48} {'diff':>7}   {'newcombe 95% CI':<20} {'perm. p (2-sided)':>18}  significant@.05")
        for name_a, name_b in all_pairs:
            _report_pair(pairwise, conditions, name_a, name_b)

    return results, pairwise


def main():
    out = {}
    for scale_name, conditions in SCALES.items():
        print(f"\n{'=' * 20} {scale_name} {'=' * 20}")
        results, pairwise = run_scale(conditions)
        out[scale_name] = {"conditions": results, "pairwise": pairwise}

    print(f"\n{'=' * 20} n120_l1adaptive {'=' * 20}")
    results, pairwise = run_scale(CONDITIONS_N120_L1ADAPTIVE, priority_pairs=PRIORITY_PAIRS_L1ADAPTIVE)
    out["n120_l1adaptive"] = {"conditions": results, "pairwise": pairwise}

    out["n_bootstrap"] = N_BOOTSTRAP
    out["n_permutations"] = N_PERMUTATIONS
    out_path = "confidence_intervals_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
