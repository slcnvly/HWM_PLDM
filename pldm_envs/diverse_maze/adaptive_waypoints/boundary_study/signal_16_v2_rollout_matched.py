# Boundary-signal study, signal 16 v2: redefined to match the model's
# ACTUAL trained rollout horizon (14 steps from a 15-frame window,
# large_diverse_25maps.yaml:1,40 -- see PROGRESS.md "Major correction"),
# not the 1-step version originally computed in run_stage2_amendment3.py.
# Still a relative (prediction-vs-prediction) comparison, so the common
# bias still cancels by construction.
#
# Cost note: a 14-step rollout is ~14x a 1-step call; to stay tractable,
# capped at 100 episodes (tighter than the usual 300-episode allowance,
# since this is doubly expensive: long rollout x K=16 batch) and evaluated
# only at a stride of starting points (0,10,20,30,40 -- 5 per episode,
# each valid since 40+14=54<=60) rather than all 46 valid starts. The
# resulting signal is genuinely coarse (nonzero only at 5 of 60 positions
# per episode) -- documented, not hidden.
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from signal1_common import get_model, WINDOW  # noqa: E402
from pldm_envs.diverse_maze.adaptive_waypoints.segmentation import pick_changepoints  # noqa: E402
from run_stage2_predictor_free import sample_episodes  # noqa: E402
from metric_b import score_boundaries, fixed_boundaries  # noqa: E402
from signals_curvature import ACTION_MEAN, ACTION_STD, N_OBS_CHANNELS, D_OBS, D_FUSED, H, W  # noqa: E402

HERE = os.path.dirname(__file__)
DATA_ROOT = os.path.join(HERE, "..", "..", "datasets", "r50_local", "r50_dataset")
N_EPISODES = 60
ROLLOUT_LEN = 14  # matches l1_n_steps=15 frames -> 14 prediction steps
K = 16
# Dense enough for peak-picking to actually discriminate: with only ~5
# sparse candidates, min_seg=8 masks index 0 anyway and pick_changepoints
# trivially selects ALL remaining nonzero points regardless of their
# relative magnitude (caught via a smoke test: obs and proprio came back
# with IDENTICAL scores because both had the same ~4 candidate positions,
# so ranking never mattered). Stride 3 gives 16 candidates -- real choice.
STARTS = list(range(0, 46, 3))  # all satisfy start+14<=60


from pldm_envs.diverse_maze.adaptive_waypoints.preprocess import normalize_images, normalize_proprio_vel, normalize_actions  # noqa: E402

def compute_episode(model, splits, ep_idx, images, seed):
    ep = splits[ep_idx]
    cum = sum(len(splits[i]["observations"]) for i in range(ep_idx))
    obs = ep["observations"][:WINDOW]
    proprio_vel = normalize_proprio_vel(torch.from_numpy(obs[:, 2:4]).float()).unsqueeze(1)
    img_seq = normalize_images(torch.from_numpy(np.array(images[cum : cum + WINDOW])).float().permute(0, 3, 1, 2))
    states = img_seq.unsqueeze(1)
    actions_t = normalize_actions(torch.from_numpy(ep["actions"][: WINDOW - 1]).float()).unsqueeze(1)
    with torch.no_grad():
        result = model.level1.forward_posterior(states, actions_t, proprio_vel=proprio_vel, encode_only=False)
    enc_torch = result.backbone_output.encodings
    proprio_torch = result.backbone_output.proprio_component
    enc_np = enc_torch.squeeze(1).flatten(1).cpu().numpy()

    rng = np.random.default_rng(seed)
    obs_sig = np.zeros(60)
    proprio_sig = np.zeros(60)

    for t in STARTS:
        candidate_seqs = rng.normal(loc=ACTION_MEAN, scale=ACTION_STD, size=(ROLLOUT_LEN, K, 2)).astype(np.float32)
        actions_b = normalize_actions(torch.from_numpy(candidate_seqs).float())  # (14, K, 2)
        state_encs_b = enc_torch[t : t + 1].squeeze(1).unsqueeze(0).expand(1, K, -1, -1, -1).contiguous()
        proprio_b = None
        if proprio_torch is not None:
            proprio_b = proprio_torch[t : t + 1].squeeze(1).unsqueeze(0).expand(1, K, -1, -1, -1).contiguous()

        with torch.no_grad():
            pred_out = model.level1.predictor.forward_multiple(
                state_encs=state_encs_b, actions=actions_b, T=ROLLOUT_LEN, proprio=proprio_b, compute_posterior=False,
            )
        final_pred = pred_out.predictions[-1].flatten(1).cpu().numpy()  # (K, D_fused)
        obs_sig[t] = final_pred[:, :D_OBS].var(axis=0).sum()
        proprio_sig[t] = final_pred[:, D_OBS:D_FUSED].var(axis=0).sum()

    return enc_np, obs_sig, proprio_sig


def main():
    model = get_model()
    splits, chosen = sample_episodes()  # 300-episode main sample, same as every other Metric B run
    images = np.load(os.path.join(DATA_ROOT, "main", "images.npy"), mmap_mode="r")
    print(f"signal 16 v2: {len(chosen)} episodes, rollout_len={ROLLOUT_LEN}, starts={STARTS}", flush=True)

    obs_scores, proprio_scores, fixed_scores = [], [], []
    for pos, ep_idx in enumerate(chosen):
        enc_np, obs_sig, proprio_sig = compute_episode(model, splits, int(ep_idx), images, seed=pos)

        bounds_obs = pick_changepoints(torch.from_numpy(obs_sig.astype(np.float64)), n_boundaries=5, min_seg=8)
        bounds_proprio = pick_changepoints(torch.from_numpy(proprio_sig.astype(np.float64)), n_boundaries=5, min_seg=8)
        obs_scores.append(score_boundaries(enc_np, bounds_obs))
        proprio_scores.append(score_boundaries(enc_np, bounds_proprio))
        fixed_scores.append(score_boundaries(enc_np, fixed_boundaries()))

        if (pos + 1) % 20 == 0:
            print(f"{pos + 1}/{len(chosen)} done", flush=True)

    fixed_mean = float(np.mean(fixed_scores))
    result = {}
    for name, vals in (("signal_16_v2_obs", obs_scores), ("signal_16_v2_proprio", proprio_scores)):
        m = float(np.mean(vals))
        result[name] = {"mean_metric_b": m, "improvement_over_fixed": (fixed_mean - m) / fixed_mean, "n_episodes": len(vals)}
        print(f"{name}: mean={m:.5f}, improvement_over_fixed={(fixed_mean - m) / fixed_mean * 100:.1f}%")
    result["fixed"] = {"mean_metric_b": fixed_mean, "n_episodes": len(fixed_scores)}

    with open(os.path.join(HERE, "results_signal16_v2.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("wrote results_signal16_v2.json")


if __name__ == "__main__":
    main()
