# Boundary-signal study, diagnostic (a) requested before Stage 2 design:
# does signal 1 (window-start-anchored open-loop error) or signal 1b
# (re-anchored one-step error) depend structurally on the window index t
# itself, independent of local dynamics? See PREREGISTRATION.md Amendment 1.
#
# Also reconstructs (deterministically -- the original file was never
# persisted locally, see PROGRESS.md) the boundary-position histogram that
# S5/S6b's adaptive_minseg8 changepoint cache would have produced, using the
# exact same signal-1 + pick_changepoints(min_seg=8) recipe.
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from signal1_common import get_model, iter_episodes  # noqa: E402
from signal1_common import compute_episode  # noqa: E402
from pldm_envs.diverse_maze.adaptive_waypoints.segmentation import pick_changepoints  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(__file__)


def main(limit_per_split=None):
    model = get_model()

    all_err1 = []  # list of (60,) arrays
    all_err1b = []
    all_boundaries = []  # flat list of boundary indices (5 per episode)

    n_done = 0
    for split in ("main", "probe"):
        for ep_idx, ep, images, ep_start in iter_episodes(split, limit=limit_per_split):
            err1, err1b, _ = compute_episode(model, ep, images, ep_start)
            all_err1.append(err1)
            all_err1b.append(err1b)

            boundaries = pick_changepoints(torch.from_numpy(err1), n_boundaries=5, min_seg=8)
            all_boundaries.extend(boundaries)

            n_done += 1
            if n_done % 200 == 0:
                print(f"{n_done} episodes done", flush=True)

    all_err1 = np.stack(all_err1)  # (N, 60)
    all_err1b = np.stack(all_err1b)
    N = all_err1.shape[0]
    print(f"total episodes processed: {N}")

    t_axis = np.arange(60)
    t_pooled = np.tile(t_axis, N)
    err1_pooled = all_err1.flatten()
    err1b_pooled = all_err1b.flatten()

    rho1, p1 = spearmanr(t_pooled, err1_pooled)
    rho1b, p1b = spearmanr(t_pooled, err1b_pooled)

    mean_err1_by_t = all_err1.mean(axis=0)
    mean_err1b_by_t = all_err1b.mean(axis=0)
    std_err1_by_t = all_err1.std(axis=0)
    std_err1b_by_t = all_err1b.std(axis=0)

    boundary_counts = np.bincount(all_boundaries, minlength=60)

    results = {
        "n_episodes": N,
        "spearman_err1_vs_t": {"rho": float(rho1), "p": float(p1)},
        "spearman_err1b_vs_t": {"rho": float(rho1b), "p": float(p1b)},
        "mean_err1_by_t": mean_err1_by_t.tolist(),
        "mean_err1b_by_t": mean_err1b_by_t.tolist(),
        "std_err1_by_t": std_err1_by_t.tolist(),
        "std_err1b_by_t": std_err1b_by_t.tolist(),
        "boundary_position_histogram_minseg8_signal1": boundary_counts.tolist(),
        "n_boundaries_total": len(all_boundaries),
    }

    out_path = os.path.join(HERE, "results_diagnostic_a.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")
    print(f"Spearman(err1, t) = {rho1:.4f} (p={p1:.2e})")
    print(f"Spearman(err1b, t) = {rho1b:.4f} (p={p1b:.2e})")

    # plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(t_axis, mean_err1_by_t, label="signal 1 (window-start-anchored)", color="crimson")
    axes[0].fill_between(t_axis, mean_err1_by_t - std_err1_by_t, mean_err1_by_t + std_err1_by_t, alpha=0.15, color="crimson")
    axes[0].plot(t_axis, mean_err1b_by_t, label="signal 1b (re-anchored)", color="steelblue")
    axes[0].fill_between(t_axis, mean_err1b_by_t - std_err1b_by_t, mean_err1b_by_t + std_err1b_by_t, alpha=0.15, color="steelblue")
    axes[0].set_xlabel("window index t")
    axes[0].set_ylabel("mean error (+/- 1 std)")
    axes[0].set_title(f"mean error vs t (N={N} episodes)\nSpearman rho: 1={rho1:.3f}, 1b={rho1b:.3f}")
    axes[0].legend()

    axes[1].bar(np.arange(60), boundary_counts, color="darkorange")
    axes[1].set_xlabel("window index (0-59)")
    axes[1].set_ylabel("count (out of 5 boundaries x N episodes)")
    axes[1].set_title("reconstructed S5/S6b changepoint boundary positions\n(signal 1, min_seg=8, deterministic reconstruction)")

    fig.tight_layout()
    plot_path = os.path.join(HERE, "plots", "diagnostic_a_position_dependency.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
