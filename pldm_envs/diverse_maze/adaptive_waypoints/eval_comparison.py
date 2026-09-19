# HWM_PLDM 원본 대비 추가된 실험 코드 (fork: slcnvly/HWM_PLDM, branch: causal-waypoint-student).
# 로컬(GPU 불필요) 4단계: (a)/(b)/(c) 세 waypoint 배치 방식의 hard-difficulty
# n=120 planning 성능을 outputs/eval_comparison.json + outputs/eval_comparison.png
# 로 정리한다.
#
# (a) baseline과 (b) oracle은 재학습/재평가 없이 RESULTS.md에 이미 기록된
# 검증된 수치를 그대로 인용한다 (SS1 표, SS6b 표 -- 아래 주석의 파일:줄 참고).
# 재실행하지 않는 이유: 이미 n=120으로 통계적으로 검증된 값이라 다시 도는 건
# GPU 시간 낭비이고, "오늘 안에 끝내야 한다"는 시간 제약과도 맞음.
#   - baseline (원논문 방식, 파인튜닝 없음): RESULTS.md SS6b 표
#     (pldm_envs/diverse_maze/adaptive_waypoints/RESULTS.md:255) "Baseline"
#     행 -- 80.0% (96/120), avg 169.6 steps.
#   - oracle (pick_changepoints, raw error, min_seg=8, 파인튜닝됨): 같은 표
#     "Adaptive min_seg=8" 행 -- 92.5% (111/120), avg 155.1 steps.
# (c) causal-student는 이 파이프라인이 새로 만든 결과
# (outputs/causal_student_eval_final.json, Kaggle에서 생성)를 읽는다.
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "outputs")
CAUSAL_RESULT_PATH = os.path.join(OUT_DIR, "causal_student_eval_final.json")

# RESULTS.md SS6b, n=120 hard difficulty -- see module docstring for citation.
BASELINE = {"label": "(a) baseline\n(fixed stride, no fine-tune)", "success_rate": 96 / 120, "n_success": 96, "n_total": 120, "avg_steps": 169.6}
ORACLE = {"label": "(b) oracle\n(pick_changepoints, non-causal)", "success_rate": 111 / 120, "n_success": 111, "n_total": 120, "avg_steps": 155.1}


def load_causal_result():
    if not os.path.exists(CAUSAL_RESULT_PATH):
        raise FileNotFoundError(
            f"{CAUSAL_RESULT_PATH} not found -- run the causal-waypoint-student Kaggle "
            "pipeline (stage 5: eval) first."
        )
    with open(CAUSAL_RESULT_PATH) as f:
        d = json.load(f)
    return {
        "label": "(c) causal student\n(online trigger)",
        "success_rate": d["k"] / d["n"],
        "n_success": d["k"],
        "n_total": d["n"],
        "avg_steps": d["avg_steps"],
    }


def main():
    causal = load_causal_result()
    conditions = [BASELINE, ORACLE, causal]

    gap_to_baseline = causal["success_rate"] - BASELINE["success_rate"]
    gap_closed_frac = (
        (causal["success_rate"] - BASELINE["success_rate"]) / (ORACLE["success_rate"] - BASELINE["success_rate"])
        if ORACLE["success_rate"] != BASELINE["success_rate"]
        else float("nan")
    )
    gap_to_oracle = ORACLE["success_rate"] - causal["success_rate"]

    comparison = {
        "n": 120,
        "conditions": {
            "a_baseline": BASELINE,
            "b_oracle": ORACLE,
            "c_causal_student": causal,
        },
        "headline": {
            "c_minus_a_success_rate_pp": round((causal["success_rate"] - BASELINE["success_rate"]) * 100, 2),
            "b_minus_c_success_rate_pp": round((ORACLE["success_rate"] - causal["success_rate"]) * 100, 2),
            "pct_of_a_to_b_gap_closed_by_c": (
                round(gap_closed_frac * 100, 1) if gap_closed_frac == gap_closed_frac else None
            ),
            "c_beats_a": bool(causal["success_rate"] > BASELINE["success_rate"]),
        },
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "eval_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    labels = [c["label"] for c in conditions]
    rates = [c["success_rate"] * 100 for c in conditions]
    steps = [c["avg_steps"] for c in conditions]
    colors = ["#8C8C8C", "#55A868", "#4C72B0"]

    axes[0].bar(labels, rates, color=colors)
    for i, r in enumerate(rates):
        axes[0].text(i, r + 1, f"{r:.1f}%", ha="center")
    axes[0].set_ylabel("success rate (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title(f"Hard difficulty success rate (n=120)")

    axes[1].bar(labels, steps, color=colors)
    for i, s in enumerate(steps):
        axes[1].text(i, s + 2, f"{s:.1f}", ha="center")
    axes[1].set_ylabel("avg steps-to-goal (successes)")
    axes[1].set_title("Planning cost")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "eval_comparison.png"), dpi=150)
    plt.close(fig)

    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
