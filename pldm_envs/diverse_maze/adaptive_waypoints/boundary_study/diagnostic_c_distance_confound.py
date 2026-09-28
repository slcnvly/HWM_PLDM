# Boundary-signal study, diagnostic (c) requested before Stage 2 design:
# is signal 1 just measuring how far the latent trajectory has moved since
# the window start? See PREREGISTRATION.md Amendment 1.
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from signal1_common import get_model, iter_episodes, compute_episode  # noqa: E402

HERE = os.path.dirname(__file__)


def main(limit_per_split=None):
    model = get_model()

    all_err1 = []
    all_dist_from_start = []  # ||z_{t+1} - z_0||_2 for t=0..59

    n_done = 0
    for split in ("main", "probe"):
        for ep_idx, ep, images, ep_start in iter_episodes(split, limit=limit_per_split):
            err1, err1b, encodings = compute_episode(model, ep, images, ep_start)
            z0 = encodings[0]
            dist = np.linalg.norm(encodings[1:61] - z0[None, :], axis=1)  # (60,)
            all_err1.append(err1)
            all_dist_from_start.append(dist)

            n_done += 1
            if n_done % 200 == 0:
                print(f"{n_done} episodes done", flush=True)

    all_err1 = np.stack(all_err1)
    all_dist = np.stack(all_dist_from_start)
    N = all_err1.shape[0]

    err1_pooled = all_err1.flatten()
    dist_pooled = all_dist.flatten()

    pear_r, pear_p = pearsonr(err1_pooled, dist_pooled)
    spear_r, spear_p = spearmanr(err1_pooled, dist_pooled)

    results = {
        "n_episodes": N,
        "pearson_err1_vs_dist_from_start": {"r": float(pear_r), "p": float(pear_p)},
        "spearman_err1_vs_dist_from_start": {"rho": float(spear_r), "p": float(spear_p)},
    }
    out_path = os.path.join(HERE, "results_diagnostic_c.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")
    print(f"Pearson(err1, dist_from_window_start) = {pear_r:.4f} (p={pear_p:.2e})")
    print(f"Spearman(err1, dist_from_window_start) = {spear_r:.4f} (p={spear_p:.2e})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # scatter with a random subsample for readability (135000 points is too many to render)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(err1_pooled), size=min(20000, len(err1_pooled)), replace=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(dist_pooled[idx], err1_pooled[idx], s=2, alpha=0.15, color="steelblue")
    ax.set_xlabel("||z_{t+1} - z_0|| (latent distance from window start)")
    ax.set_ylabel("signal 1 (window-start-anchored open-loop error)")
    ax.set_title(f"signal 1 vs. distance-from-window-start\nPearson r={pear_r:.3f}, Spearman rho={spear_r:.3f} (N={N} eps, 20k pt subsample)")
    fig.tight_layout()
    plot_path = os.path.join(HERE, "plots", "diagnostic_c_distance_confound.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
