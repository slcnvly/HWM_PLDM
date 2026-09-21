# HWM_PLDM 원본 대비 추가된 실험 코드 (fork: slcnvly/HWM_PLDM, branch: causal-waypoint-student).
# 도달 기반 종료 실험의 최종 분석. numpy/matplotlib만 사용 (scipy 없음 -- 이 프로젝트의
# confidence_intervals.py와 같은 관례: exact McNemar's는 math.comb로 정확한 이항분포
# 꼬리확률을 직접 계산, Wilcoxon 부호순위는 표준 정규근사(continuity correction)로 계산).
#
# 입력: 같은 120개 인스턴스에 대해 순서가 동일한 per-trial success/steps 리스트
# (baseline=시간 기반 종료, arrival=도달 기반 종료, paper_baseline은 선택).
import argparse
import json
import math
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def exact_mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar's test p-value: b, c are the two
    discordant-pair counts (A-succ/B-fail and A-fail/B-succ). Under H0,
    b ~ Binomial(b+c, 0.5); two-sided p = 2 * min(P(X<=min(b,c)), P(X>=max(b,c))),
    capped at 1.0 (standard exact-binomial McNemar's, matches R's
    mcnemar.test(..., correct=FALSE) exact variant for small n)."""
    n = b + c
    if n == 0:
        return 1.0
    lo, hi = min(b, c), max(b, c)

    def binom_cdf_le(k, n):
        return sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)

    p_lo = binom_cdf_le(lo, n)
    p_hi = 1.0 - binom_cdf_le(hi - 1, n) if hi > 0 else 1.0
    return min(1.0, 2 * min(p_lo, p_hi))


def wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray):
    """Paired Wilcoxon signed-rank test, normal approximation with
    continuity correction (standard for n not tiny). x, y: 1D arrays of
    equal length (paired steps-to-goal on the dual-success intersection).
    Returns (statistic W, two-sided p-value, n_used_after_dropping_ties)."""
    d = x - y
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 0.0, 1.0, 0
    abs_d = np.abs(d)
    ranks = np.argsort(np.argsort(abs_d)) + 1  # average-rank tie handling below
    # handle ties in |d| by averaging ranks within tie groups
    order = np.argsort(abs_d)
    sorted_abs = abs_d[order]
    avg_ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_abs[j + 1] == sorted_abs[i]:
            j += 1
        avg_ranks[i : j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    ranks_full = np.empty(n)
    ranks_full[order] = avg_ranks

    w_pos = ranks_full[d > 0].sum()
    w_neg = ranks_full[d < 0].sum()
    W = min(w_pos, w_neg)

    mean_w = n * (n + 1) / 4.0
    std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if std_w == 0:
        return W, 1.0, n
    z = (W - mean_w + 0.5) / std_w  # continuity correction
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return float(W), float(min(1.0, p)), n


def summarize(name: str, success: list, steps: list):
    n = len(success)
    k = int(sum(success))
    succ_steps = [s for s, st in zip(steps, success) if st]
    avg_steps = float(np.mean(succ_steps)) if succ_steps else -1.0
    return {"label": name, "n": n, "k": k, "rate": k / n, "avg_steps": avg_steps}


def paired_compare(name_a, a_succ, a_steps, name_b, b_succ, b_steps):
    a_succ, b_succ = np.array(a_succ, dtype=bool), np.array(b_succ, dtype=bool)
    a_steps, b_steps = np.array(a_steps, dtype=float), np.array(b_steps, dtype=float)
    assert len(a_succ) == len(b_succ), "instance count mismatch -- not paired correctly"

    b_disc = int(np.sum(a_succ & ~b_succ))  # A succeeded, B failed
    c_disc = int(np.sum(~a_succ & b_succ))  # A failed, B succeeded
    mcnemar_p = exact_mcnemar_p(b_disc, c_disc)

    both_succ = a_succ & b_succ
    n_both = int(both_succ.sum())
    if n_both > 0:
        W, wilcoxon_p, n_used = wilcoxon_signed_rank(a_steps[both_succ], b_steps[both_succ])
    else:
        W, wilcoxon_p, n_used = 0.0, 1.0, 0

    return {
        "pair": f"{name_a}_vs_{name_b}",
        "mcnemar": {"b_a_succ_b_fail": b_disc, "c_a_fail_b_succ": c_disc, "p_value": mcnemar_p},
        "wilcoxon_steps_dual_success": {
            "n_both_succeeded": n_both, "n_used_nonzero_diff": n_used,
            "statistic": W, "p_value": wilcoxon_p,
            "mean_steps_a": float(a_steps[both_succ].mean()) if n_both else None,
            "mean_steps_b": float(b_steps[both_succ].mean()) if n_both else None,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_pertrial", type=str, required=True)
    parser.add_argument("--arrival_pertrial", type=str, required=True)
    parser.add_argument("--arrival_stats", type=str, required=True)
    parser.add_argument("--epsilon_exploration", type=str, required=True)
    parser.add_argument("--epsilon_candidates", type=str, required=True)
    parser.add_argument("--paper_baseline_pertrial", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args()

    with open(args.baseline_pertrial) as f:
        baseline = json.load(f)
    with open(args.arrival_pertrial) as f:
        arrival = json.load(f)
    with open(args.arrival_stats) as f:
        arrival_stats = json.load(f)
    with open(args.epsilon_exploration) as f:
        eps_explore = json.load(f)
    with open(args.epsilon_candidates) as f:
        eps_candidates = json.load(f)

    conditions = {
        "baseline_time_based": summarize("baseline (time-based)", baseline["success"], baseline["steps"]),
        "arrival_based": summarize("arrival-based", arrival["success"], arrival["steps"]),
    }
    comparisons = [
        paired_compare("baseline_time_based", baseline["success"], baseline["steps"],
                        "arrival_based", arrival["success"], arrival["steps"])
    ]

    paper_baseline = None
    if args.paper_baseline_pertrial and os.path.exists(args.paper_baseline_pertrial):
        with open(args.paper_baseline_pertrial) as f:
            paper_baseline = json.load(f)
        conditions["paper_baseline"] = summarize("paper baseline", paper_baseline["success"], paper_baseline["steps"])
        comparisons.append(
            paired_compare("paper_baseline", paper_baseline["success"], paper_baseline["steps"],
                            "baseline_time_based", baseline["success"], baseline["steps"])
        )
        comparisons.append(
            paired_compare("paper_baseline", paper_baseline["success"], paper_baseline["steps"],
                            "arrival_based", arrival["success"], arrival["steps"])
        )

    total_advances = sum(arrival_stats.get(k, 0) for k in ("arrival", "forced", "initial"))
    non_initial = arrival_stats.get("arrival", 0) + arrival_stats.get("forced", 0)
    arrival_frac = arrival_stats.get("arrival", 0) / non_initial if non_initial > 0 else None

    result = {
        "conditions": conditions,
        "comparisons": comparisons,
        "epsilon_candidates": eps_candidates,
        "epsilon_exploration": eps_explore,
        "chosen_epsilon": eps_explore.get("chosen_epsilon"),
        "arrival_advance_stats": {
            **arrival_stats,
            "total_replan_events": total_advances,
            "fraction_arrival_triggered_excl_initial": arrival_frac,
        },
    }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "termination_comparison.json"), "w") as f:
        json.dump(result, f, indent=2)

    labels = list(conditions.keys())
    rates = [conditions[k]["rate"] * 100 for k in labels]
    steps = [conditions[k]["avg_steps"] for k in labels]
    colors = ["#8C8C8C", "#4C72B0", "#55A868"][: len(labels)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(labels, rates, color=colors)
    for i, r in enumerate(rates):
        axes[0].text(i, r + 1, f"{r:.1f}%", ha="center")
    axes[0].set_ylabel("success rate (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("Hard difficulty success rate (n=120)")

    axes[1].bar(labels, steps, color=colors)
    for i, s in enumerate(steps):
        axes[1].text(i, s + 2, f"{s:.1f}", ha="center")
    axes[1].set_ylabel("avg steps-to-goal (successes)")
    axes[1].set_title("Planning cost")

    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "termination_comparison.png"), dpi=150)
    plt.close(fig)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
