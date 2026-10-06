# Boundary-signal study, Amendment 3: curvature signals (11-13), action-
# sensitivity divergence (16, obs+proprio), and two direct-Metric-B
# algorithms (14 top-down, 15 bottom-up, each unconstrained + min_seg=8),
# scored via Metric B on the SAME 300-episode sample as the original
# Stage 2 predictor-free run, for a single combined comparison table.
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from signal1_common import get_model, WINDOW  # noqa: E402
from pldm_envs.diverse_maze.adaptive_waypoints.segmentation import pick_changepoints  # noqa: E402
from run_stage2_predictor_free import sample_episodes, get_episode_encodings_actions  # noqa: E402
from signals_curvature import (  # noqa: E402
    signal_11_curvature, signal_12_normalized_curvature, signal_13_chord_deviation,
    signal_16_action_sensitivity_divergence,
)
from metric_b import score_boundaries, top_down_split, bottom_up_merge, fixed_boundaries  # noqa: E402

HERE = os.path.dirname(__file__)
DATA_ROOT = os.path.join(HERE, "..", "..", "datasets", "r50_local", "r50_dataset")
MIN_SEG = 8
N_BOUNDARIES = 5
CKPT_PATH = os.path.join(HERE, "amendment3_checkpoint.npz")


from pldm_envs.diverse_maze.adaptive_waypoints.preprocess import normalize_images, normalize_proprio_vel, normalize_actions  # noqa: E402

def get_episode_torch_encodings(model, splits, ep_idx, images):
    """Like get_episode_encodings_actions but also returns the raw torch
    tensors needed by signal_16 (real per-frame encodings + proprio,
    unflattened)."""
    ep = splits[ep_idx]
    cum = sum(len(splits[i]["observations"]) for i in range(ep_idx))
    obs = ep["observations"][:WINDOW]
    proprio_vel = normalize_proprio_vel(torch.from_numpy(obs[:, 2:4]).float()).unsqueeze(1)
    img_seq = normalize_images(torch.from_numpy(np.array(images[cum : cum + WINDOW])).float().permute(0, 3, 1, 2))
    states = img_seq.unsqueeze(1)
    actions_t = normalize_actions(torch.from_numpy(ep["actions"][: WINDOW - 1]).float()).unsqueeze(1)
    with torch.no_grad():
        result = model.level1.forward_posterior(states, actions_t, proprio_vel=proprio_vel, encode_only=False)
    enc_torch = result.backbone_output.encodings  # (61,1,C,H,W)
    proprio_torch = result.backbone_output.proprio_component
    enc_np = enc_torch.squeeze(1).flatten(1).cpu().numpy()
    return enc_np, enc_torch, proprio_torch


def main():
    model = get_model()
    splits, chosen = sample_episodes()  # same default (300, seed=0) as original Stage 2 run
    images = np.load(os.path.join(DATA_ROOT, "main", "images.npy"), mmap_mode="r")
    print(f"using {len(chosen)} episodes (same sample as run_stage2_predictor_free.py)", flush=True)

    names = [
        "signal_11_curvature", "signal_12_norm_curvature", "signal_13_chord_dev",
        "signal_16_obs", "signal_16_proprio",
        "top_down_unconstrained", "top_down_minseg8",
        "bottom_up_unconstrained", "bottom_up_minseg8",
    ]
    scores = {name: [] for name in names}
    fixed_scores = []

    start = 0
    if os.path.exists(CKPT_PATH):
        ck = np.load(CKPT_PATH, allow_pickle=True)
        start = int(ck["next_idx"])
        scores = ck["scores"].item()
        fixed_scores = list(ck["fixed_scores"])
        print(f"resuming from episode index {start}/{len(chosen)}", flush=True)

    for pos in range(start, len(chosen)):
        ep_idx = int(chosen[pos])
        enc_np, enc_torch, proprio_torch = get_episode_torch_encodings(model, splits, ep_idx, images)

        sig11 = signal_11_curvature(enc_np)
        sig12 = signal_12_normalized_curvature(enc_np)
        sig13 = signal_13_chord_deviation(enc_np, w=5)
        sig16_obs, sig16_proprio = signal_16_action_sensitivity_divergence(model, enc_torch, proprio_torch, K=16, seed=pos)

        for name, sig in (
            ("signal_11_curvature", sig11), ("signal_12_norm_curvature", sig12),
            ("signal_13_chord_dev", sig13), ("signal_16_obs", sig16_obs), ("signal_16_proprio", sig16_proprio),
        ):
            bounds = pick_changepoints(torch.from_numpy(sig.astype(np.float64)), n_boundaries=N_BOUNDARIES, min_seg=MIN_SEG)
            scores[name].append(score_boundaries(enc_np, bounds))

        scores["top_down_unconstrained"].append(score_boundaries(enc_np, top_down_split(enc_np, N_BOUNDARIES, None)))
        scores["top_down_minseg8"].append(score_boundaries(enc_np, top_down_split(enc_np, N_BOUNDARIES, MIN_SEG)))
        scores["bottom_up_unconstrained"].append(score_boundaries(enc_np, bottom_up_merge(enc_np, N_BOUNDARIES, None)))
        scores["bottom_up_minseg8"].append(score_boundaries(enc_np, bottom_up_merge(enc_np, N_BOUNDARIES, MIN_SEG)))

        fixed_scores.append(score_boundaries(enc_np, fixed_boundaries()))

        if (pos + 1) % 20 == 0:
            np.savez(CKPT_PATH, next_idx=pos + 1, scores=scores, fixed_scores=np.array(fixed_scores))
            print(f"{pos + 1}/{len(chosen)} episodes done (checkpointed)", flush=True)

    fixed_mean = float(np.mean(fixed_scores))
    summary = {"fixed": {"mean_metric_b": fixed_mean, "n_episodes": len(fixed_scores)}}
    for name in names:
        vals = np.array(scores[name])
        mean_val = float(vals.mean())
        improvement = (fixed_mean - mean_val) / fixed_mean if fixed_mean > 0 else 0.0
        summary[name] = {"mean_metric_b": mean_val, "median_metric_b": float(np.median(vals)), "improvement_over_fixed": improvement, "n_episodes": len(vals)}
        print(f"{name}: mean={mean_val:.5f}, improvement_over_fixed={improvement*100:.1f}%")

    with open(os.path.join(HERE, "results_stage2_amendment3.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote results_stage2_amendment3.json")
    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)


if __name__ == "__main__":
    main()
