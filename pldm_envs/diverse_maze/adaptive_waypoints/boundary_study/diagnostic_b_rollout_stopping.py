# Boundary-signal study, diagnostic (b) requested before Stage 2 design:
# does the level1 open-loop rollout (the same one signal 1 is computed from)
# effectively stop evolving partway through the window, while the real
# trajectory keeps moving? See PREREGISTRATION.md Amendment 1.
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from signal1_common import get_model, iter_episodes, WINDOW  # noqa: E402

HERE = os.path.dirname(__file__)
N_EPISODES = 50
K_MAX = 20  # k = 0..20 -> 21 step-deltas each (k -> k+1)


from pldm_envs.diverse_maze.adaptive_waypoints.preprocess import normalize_images, normalize_proprio_vel, normalize_actions  # noqa: E402

def main():
    model = get_model()

    pred_deltas = []  # (N_EPISODES, K_MAX)
    real_deltas = []

    n_done = 0
    for split in ("main",):  # 50 episodes from main is enough per the request
        for ep_idx, ep, images, ep_start in iter_episodes(split, limit=N_EPISODES):
            obs = ep["observations"][:WINDOW]
            proprio_vel = normalize_proprio_vel(torch.from_numpy(obs[:, 2:4]).float()).unsqueeze(1)
            img_seq = normalize_images(torch.from_numpy(np.array(images[ep_start : ep_start + WINDOW])).float().permute(0, 3, 1, 2))
            states = img_seq.unsqueeze(1)
            actions = normalize_actions(torch.from_numpy(ep["actions"][: WINDOW - 1]).float()).unsqueeze(1)

            with torch.no_grad():
                result = model.level1.forward_posterior(states, actions, proprio_vel=proprio_vel, encode_only=False)

            predictions = result.pred_output.predictions.squeeze(1)  # (61, D...) rollout, predictions[0]==z_0
            real = result.backbone_output.encodings.squeeze(1)  # (61, D...) real per-frame encodings

            pd = [
                (predictions[k + 1] - predictions[k]).pow(2).mean().sqrt().item()
                for k in range(K_MAX)
            ]
            rd = [
                (real[k + 1] - real[k]).pow(2).mean().sqrt().item()
                for k in range(K_MAX)
            ]
            pred_deltas.append(pd)
            real_deltas.append(rd)

            n_done += 1
            if n_done % 10 == 0:
                print(f"{n_done} episodes done", flush=True)

    pred_deltas = np.array(pred_deltas)  # (50, 20)
    real_deltas = np.array(real_deltas)

    mean_pred = pred_deltas.mean(axis=0)
    mean_real = real_deltas.mean(axis=0)
    std_pred = pred_deltas.std(axis=0)
    std_real = real_deltas.std(axis=0)

    results = {
        "n_episodes": N_EPISODES,
        "k_max": K_MAX,
        "mean_predicted_step_delta": mean_pred.tolist(),
        "mean_real_step_delta": mean_real.tolist(),
        "std_predicted_step_delta": std_pred.tolist(),
        "std_real_step_delta": std_real.tolist(),
        "ratio_pred_to_real_at_k19": float(mean_pred[19] / mean_real[19]),
        "ratio_pred_to_real_at_k0": float(mean_pred[0] / mean_real[0]),
    }
    out_path = os.path.join(HERE, "results_diagnostic_b.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")
    print("mean predicted step delta (k=0..19):", np.round(mean_pred, 4))
    print("mean real step delta      (k=0..19):", np.round(mean_real, 4))
    print(f"pred/real ratio at k=0: {results['ratio_pred_to_real_at_k0']:.3f}, at k=19: {results['ratio_pred_to_real_at_k19']:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    k_axis = np.arange(K_MAX)
    ax.plot(k_axis, mean_pred, "-o", label="predicted (open-loop rollout) step delta", color="crimson")
    ax.fill_between(k_axis, mean_pred - std_pred, mean_pred + std_pred, alpha=0.15, color="crimson")
    ax.plot(k_axis, mean_real, "-o", label="real trajectory step delta", color="steelblue")
    ax.fill_between(k_axis, mean_real - std_real, mean_real + std_real, alpha=0.15, color="steelblue")
    ax.set_xlabel("rollout step k")
    ax.set_ylabel("||z_{k+1} - z_k|| (RMS)")
    ax.set_title(f"open-loop rollout vs real trajectory step size (N={N_EPISODES} episodes)")
    ax.legend()
    fig.tight_layout()
    plot_path = os.path.join(HERE, "plots", "diagnostic_b_rollout_stopping.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
