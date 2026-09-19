# HWM_PLDM 원본 대비 추가된 실험 코드 (fork: slcnvly/HWM_PLDM, branch: adaptive-waypoints).
# 로컬 WSL(GPU 없음)에서 실행. kaggle/extract_embeddings.py가 만든
# outputs/embeddings.npz만 읽어서 waypoint 배치가 구조적으로 타당한지
# 진단한다. numpy/matplotlib 외 의존성 없음 (torch, 모델 코드 import 안 함).
#
# 신호: surprise = err^2 / (predicted_variance + eps) (RESULTS.md SS8, embeddings.npz
# 생성 시 사용자가 선택한 신호). 실제 waypoint 배치 알고리즘은 이 신호에 대한
# 단순 threshold가 아니라 min_seg 최소 간격 제약을 둔 top-n_boundaries 그리디
# peak-picking(segmentation.py::pick_changepoints)이라, "threshold 민감도"
# 그림의 세로 점선은 진짜 설정값이 아니라 "실제 채택된 boundary들 중 가장 낮은
# surprise 값" -- 즉 이 알고리즘이 암묵적으로 쓰고 있는 유효 컷오프 -- 로 표시한다.
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IN_PATH = os.path.join(HERE, "..", "outputs", "embeddings.npz")
OUT_DIR = os.path.join(HERE, "..", "outputs")


def gaussian_kde(samples: np.ndarray, grid: np.ndarray, bandwidth: float) -> np.ndarray:
    """Manual Gaussian KDE (Silverman bandwidth by default) -- avoids a scipy
    dependency so this script stays numpy/matplotlib only."""
    diffs = (grid[:, None] - samples[None, :]) / bandwidth
    density = np.exp(-0.5 * diffs**2).sum(axis=1)
    return density / (samples.shape[0] * bandwidth * np.sqrt(2 * np.pi))


def silverman_bandwidth(samples: np.ndarray) -> float:
    n = samples.shape[0]
    std = samples.std()
    iqr = np.percentile(samples, 75) - np.percentile(samples, 25)
    sigma = min(std, iqr / 1.349) if iqr > 0 else std
    sigma = sigma if sigma > 0 else (std if std > 0 else 1.0)
    return 0.9 * sigma * n ** (-1 / 5)


def find_local_extrema(y: np.ndarray, min_prominence_frac: float = 0.02):
    """Indices of local maxima and minima in a 1D curve. Filters out maxima
    whose height is below `min_prominence_frac` of the global max -- without
    this, KDE noise in a sparse tail produces spurious tiny bumps that get
    counted as separate modes even when the distribution is really unimodal."""
    raw_maxima, raw_minima = [], []
    for i in range(1, len(y) - 1):
        if y[i - 1] < y[i] > y[i + 1]:
            raw_maxima.append(i)
        elif y[i - 1] > y[i] < y[i + 1]:
            raw_minima.append(i)

    y_max = y.max()
    if y_max <= 0:
        return [], []
    maxima = [i for i in raw_maxima if y[i] >= min_prominence_frac * y_max]
    # keep only minima that sit between two surviving (significant) maxima --
    # a minimum next to a filtered-out noise bump isn't a real valley
    minima = []
    for i in raw_minima:
        left = [m for m in maxima if m < i]
        right = [m for m in maxima if m > i]
        if left and right:
            minima.append(i)
    return maxima, minima


def bimodality_coefficient(samples: np.ndarray) -> float:
    """Pearson's bimodality coefficient (SAS convention): (skew^2 + 1) /
    excess_kurtosis, sample-size-corrected. > 0.555 (the value for a uniform
    distribution) is the usual rule-of-thumb "possibly bimodal" cutoff."""
    n = samples.shape[0]
    m = samples.mean()
    s = samples.std(ddof=1)
    if s == 0:
        return float("nan")
    skew = ((samples - m) ** 3).mean() / s**3
    kurt = ((samples - m) ** 4).mean() / s**4  # not excess yet
    excess_kurt = kurt - 3
    g1 = skew * np.sqrt(n * (n - 1)) / (n - 2) if n > 2 else skew
    correction = 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)) if n > 3 else 1.0
    g2 = excess_kurt * correction if n > 3 else excess_kurt
    bc = (g1**2 + 1) / (g2 + 3)
    return float(bc)


def plot_waypoint_intervals(boundaries: np.ndarray, T: int, out_path: str):
    """(1) 궤적별 인접 waypoint 간격 히스토그램 (0, T를 양끝 경계로 포함 --
    boundaries_to_segments와 동일한 관례)."""
    n_ep = boundaries.shape[0]
    gaps = []
    for row in range(n_ep):
        edges = [0] + sorted(boundaries[row].tolist()) + [T]
        gaps.extend(np.diff(edges).tolist())
    gaps = np.array(gaps, dtype=np.float64)

    stats = {
        "n_gaps": int(gaps.size),
        "mean": float(gaps.mean()),
        "median": float(np.median(gaps)),
        "std": float(gaps.std()),
        "min": float(gaps.min()),
        "max": float(gaps.max()),
        "frac_gap_1_to_2": float(((gaps >= 1) & (gaps <= 2)).mean()),
        "cv": float(gaps.std() / gaps.mean()) if gaps.mean() != 0 else float("nan"),
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(gaps, bins=np.arange(gaps.min(), gaps.max() + 2) - 0.5, color="#4C72B0", edgecolor="white")
    ax.axvline(stats["mean"], color="#C44E52", linestyle="--", linewidth=1.5, label=f"mean={stats['mean']:.1f}")
    ax.set_xlabel("waypoint interval (raw steps)")
    ax.set_ylabel("count")
    ax.set_title("Waypoint interval distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return stats, gaps


def plot_surprise_distribution(surprise: np.ndarray, out_path: str):
    """(2) 인접 스텝 간 surprise score 히스토그램 + KDE (단봉/이봉 판단용)."""
    flat = surprise.flatten().astype(np.float64)
    flat = flat[np.isfinite(flat)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts, bin_edges, _ = ax.hist(
        flat, bins=60, color="#4C72B0", edgecolor="white", alpha=0.75, density=True, label="histogram"
    )

    bw = silverman_bandwidth(flat)
    grid = np.linspace(flat.min(), flat.max(), 512)
    density = gaussian_kde(flat, grid, bw)
    ax.plot(grid, density, color="#C44E52", linewidth=2, label="KDE")

    maxima_idx, minima_idx = find_local_extrema(density)
    for mi in minima_idx:
        ax.axvline(grid[mi], color="#55A868", linestyle=":", linewidth=1.2)

    ax.set_xlabel("surprise score (err^2 / sigma^2)")
    ax.set_ylabel("density")
    ax.set_title("Surprise score distribution (adjacent-step)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    bc = bimodality_coefficient(flat)
    stats = {
        "n_samples": int(flat.size),
        "mean": float(flat.mean()),
        "median": float(np.median(flat)),
        "std": float(flat.std()),
        "kde_bandwidth_silverman": float(bw),
        "kde_n_local_maxima": len(maxima_idx),
        "kde_n_local_minima": len(minima_idx),
        "kde_local_minima_x": [float(grid[i]) for i in minima_idx],
        "bimodality_coefficient": bc,
        "bimodality_coefficient_note": "SAS rule of thumb: >0.555 suggests possible bimodality",
    }
    return stats


def plot_threshold_sensitivity(surprise: np.ndarray, boundaries_score_floor: float, out_path: str):
    """(3) threshold를 min~99th percentile 구간 50단계로 바꾸며, threshold를
    넘는 (episode, step) 개수 변화를 본다. 실제 알고리즘엔 명시적 threshold가
    없으므로, 채택된 boundary들 중 최소 surprise 값을 "유효 컷오프"로 세로
    점선 표시한다 (모듈 docstring 참고)."""
    flat = surprise.flatten().astype(np.float64)
    flat = flat[np.isfinite(flat)]

    lo = flat.min()
    hi = np.percentile(flat, 99)
    thresholds = np.linspace(lo, hi, 50)
    n_above = np.array([(flat > t).sum() for t in thresholds])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, n_above, color="#4C72B0", marker="o", markersize=3)
    ax.axvline(
        boundaries_score_floor,
        color="#C44E52",
        linestyle="--",
        linewidth=1.5,
        label=f"effective cutoff (min score among chosen boundaries)={boundaries_score_floor:.2f}",
    )
    ax.set_xlabel("threshold (surprise score)")
    ax.set_ylabel("# (episode, step) pairs above threshold")
    ax.set_title("Threshold sensitivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {
        "threshold_min": float(lo),
        "threshold_p99": float(hi),
        "n_above_at_effective_cutoff": int((flat > boundaries_score_floor).sum()),
        "effective_cutoff": float(boundaries_score_floor),
    }


def interpret(interval_stats: dict, gaps: np.ndarray, min_seg: int, surprise_stats: dict) -> dict:
    """두 질문에 대한 해석을 실제 계산된 통계로부터 생성한다."""
    # Q1: 자연스러운 절단점(natural cutoff)이 있는가?
    n_minima = surprise_stats["kde_n_local_minima"]
    bc = surprise_stats["bimodality_coefficient"]
    if n_minima == 0:
        cutoff_verdict = (
            "KDE 곡선에 지역 극소값(valley)이 없음 -- surprise score 분포는 단봉(unimodal)이고, "
            "분포 자체에서 자연스럽게 갈라지는 절단점은 보이지 않는다. 다만 이는 곡선 형태(단봉 vs "
            "이봉)에 대한 판단이고, 알고리즘이 실제로 쓰는 '유효 컷오프'(채택된 boundary들 중 최소 "
            "score)가 이 단봉 분포의 어디쯤 위치하는지는 별개로 확인해야 한다 (아래 threshold_sensitivity 참고)."
        )
    else:
        cutoff_verdict = (
            f"KDE 곡선에 지역 극소값이 {n_minima}개 있음 (x={surprise_stats['kde_local_minima_x']}) -- "
            "분포가 다봉(multimodal)일 가능성이 있고, 이 극소점(들)이 '배경 노이즈'와 '진짜 surprise'를 "
            "구분하는 자연스러운 절단점의 후보가 된다."
        )
    bc_note = f"Bimodality coefficient={bc:.3f} ({'>' if bc > 0.555 else '<='} 0.555 rule-of-thumb)."

    # Q2: waypoint 간격이 예측 가능한 수준의 구조를 갖는가?
    cv = interval_stats["cv"]
    frac_at_floor = float((gaps == min_seg).mean())
    frac_1_2 = interval_stats["frac_gap_1_to_2"]
    structure_verdict = (
        f"간격의 변동계수(CV=std/mean)={cv:.3f}, min_seg={min_seg} 바로 그 값에 걸린 간격 비율="
        f"{frac_at_floor:.1%}. "
    )
    if frac_at_floor > 0.5:
        structure_verdict += (
            "간격의 절반 이상이 min_seg 하한에 그대로 걸려 있음 -- 이는 surprise 신호가 '어디에 waypoint를 "
            "둘지'를 실질적으로 결정하기보다, min_seg 최소-간격 제약이 배치를 지배하고 있다는 뜻이다. "
            "즉 간격 자체는 예측 가능하지만(대부분 min_seg), 그 예측 가능성의 원천이 신호가 아니라 "
            "알고리즘의 하드 제약이라는 점에 주의해야 한다."
        )
    elif cv < 0.3:
        structure_verdict += (
            "CV가 낮아 간격이 비교적 좁은 범위에 몰려 있음 -- 신호 기반 배치이지만 실질적으로는 균일 "
            "배치에 가까운 구조를 보인다."
        )
    else:
        structure_verdict += (
            "간격이 min_seg 하한에 쏠리지 않고 CV도 상당히 커서, surprise 신호가 실제로 배치 위치에 "
            "유의미한 변동을 만들고 있다는 근거로 볼 수 있다."
        )
    structure_verdict += f" (참고: 간격 1~2스텝 비율={frac_1_2:.1%} -- min_seg={min_seg}에서는 구조상 0%에 가까워야 정상.)"

    return {
        "q1_natural_cutoff_in_surprise_distribution": cutoff_verdict + " " + bc_note,
        "q2_waypoint_interval_structure": structure_verdict,
    }


def main():
    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(
            f"{IN_PATH} not found -- run kaggle/extract_embeddings.py on Kaggle first and copy "
            "outputs/embeddings.npz here."
        )
    os.makedirs(OUT_DIR, exist_ok=True)

    data = np.load(IN_PATH)
    surprise = data["surprise_scores"].astype(np.float32)
    boundaries = data["boundaries"]
    T = int(data["T"])
    min_seg = int(data["min_seg"])

    interval_stats, gaps = plot_waypoint_intervals(boundaries, T, os.path.join(OUT_DIR, "waypoint_intervals.png"))
    surprise_stats = plot_surprise_distribution(surprise, os.path.join(OUT_DIR, "surprise_distribution.png"))

    boundary_scores = np.array(
        [surprise[row, b] for row in range(surprise.shape[0]) for b in boundaries[row]]
    )
    effective_cutoff = float(boundary_scores.min())
    threshold_stats = plot_threshold_sensitivity(
        surprise, effective_cutoff, os.path.join(OUT_DIR, "threshold_sensitivity.png")
    )

    interpretation = interpret(interval_stats, gaps, min_seg, surprise_stats)

    diagnostics = {
        "input": {
            "n_episodes": int(surprise.shape[0]),
            "T": T,
            "min_seg": min_seg,
            "n_boundaries": int(data["n_boundaries"]),
            "seed": int(data["seed"]),
        },
        "waypoint_intervals": interval_stats,
        "surprise_distribution": surprise_stats,
        "threshold_sensitivity": threshold_stats,
        "interpretation": interpretation,
    }
    with open(os.path.join(OUT_DIR, "diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)

    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
