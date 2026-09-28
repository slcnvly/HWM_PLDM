# Boundary-signal study, Stage 2 driver for predictor-free signals only
# (6, 7, 8, 10 -- per the user's option-3 decision after the Amendment-2
# gate check; predictor-based signals 2,2b,3,3b,4,5,9,9b are on hold).
# Computes signals + fixed/random/oracle baselines, scores all via Metric B
# (PREREGISTRATION.md SS6), on up to 300 episodes from the selection set
# (main), stratified across its 25 maps -- matches the "expensive signal"
# cap in SS7 (BOCPD is the expensive one here; the others are cheap but
# scored on the same sample for a clean apples-to-apples comparison).
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from signal1_common import get_model, WINDOW  # noqa: E402
from pldm_envs.diverse_maze.adaptive_waypoints.segmentation import pick_changepoints  # noqa: E402
from signals_predictor_free import (  # noqa: E402
    signal_7_latent_speed,
    signal_8_direction_change,
    signal_10_action_delta,
    signal_6_bocpd,
)
from metric_b import score_boundaries, oracle_boundaries, random_boundaries, fixed_boundaries  # noqa: E402

HERE = os.path.dirname(__file__)
DATA_ROOT = os.path.join(HERE, "..", "..", "datasets", "r50_local", "r50_dataset")
N_EPISODES_TARGET = 300
N_PCA_COMPONENTS = 10
N_RANDOM_SEEDS = 20
MIN_SEG = 8
N_BOUNDARIES = 5


def sample_episodes(n_target=N_EPISODES_TARGET, seed=0):
    """Stratified sample from main (selection set), ~n_target/25 per map."""
    splits = torch.load(os.path.join(DATA_ROOT, "main", "data.p"), weights_only=False)
    by_map = {}
    for idx, ep in enumerate(splits):
        by_map.setdefault(int(ep["map_idx"]), []).append(idx)
    n_maps = len(by_map)
    per_map = max(1, n_target // n_maps)
    rng = np.random.default_rng(seed)
    chosen = []
    for m, idxs in by_map.items():
        k = min(per_map, len(idxs))
        chosen.extend(rng.choice(idxs, size=k, replace=False).tolist())
    chosen.sort()
    return splits, chosen


def get_episode_encodings_actions(model, splits, ep_idx, images):
    ep = splits[ep_idx]
    ep_start = 0
    # cumulative offset for image indexing (same convention as signal1_common)
    cum = 0
    for i in range(ep_idx):
        cum += len(splits[i]["observations"])
    ep_start = cum

    obs = ep["observations"][:WINDOW]
    proprio_vel = torch.from_numpy(obs[:, 2:4]).float().unsqueeze(1)
    img_seq = torch.from_numpy(np.array(images[ep_start : ep_start + WINDOW])).float().permute(0, 3, 1, 2)
    states = img_seq.unsqueeze(1)
    actions_t = torch.from_numpy(ep["actions"][: WINDOW - 1]).float().unsqueeze(1)

    with torch.no_grad():
        result = model.level1.forward_posterior(states, actions_t, proprio_vel=proprio_vel, encode_only=True)
    encodings = result.backbone_output.encodings.squeeze(1).flatten(1).cpu().numpy()  # (61, D)
    actions_np = ep["actions"][: WINDOW - 1]  # (60, A)
    return encodings, actions_np


def main():
    model = get_model()
    splits, chosen = sample_episodes()
    images = np.load(os.path.join(DATA_ROOT, "main", "images.npy"), mmap_mode="r")
    print(f"sampled {len(chosen)} episodes across {len(set(int(splits[i]['map_idx']) for i in chosen))} maps", flush=True)

    # --- fit PCA for signal 6 on a random subsample of encodings ---
    print("fitting PCA for signal 6...", flush=True)
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(0)
    pca_sample_eps = rng.choice(chosen, size=min(60, len(chosen)), replace=False)
    pca_rows = []
    for ep_idx in pca_sample_eps:
        enc, _ = get_episode_encodings_actions(model, splits, int(ep_idx), images)
        pca_rows.append(enc)
    pca_fit_data = np.concatenate(pca_rows, axis=0)
    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=0)
    pca.fit(pca_fit_data)
    print(f"PCA fit on {pca_fit_data.shape[0]} samples, explained variance ratio sum: {pca.explained_variance_ratio_.sum():.3f}", flush=True)
    del pca_rows, pca_fit_data

    signal_names = ["signal_6_bocpd", "signal_7_speed", "signal_8_direction", "signal_10_action_delta"]
    baseline_names = ["fixed", "random", "oracle"]
    metric_b_results = {name: [] for name in signal_names + baseline_names}

    rand_rng = np.random.default_rng(42)
    n_done = 0
    for ep_idx in chosen:
        ep_idx = int(ep_idx)
        enc, actions_np = get_episode_encodings_actions(model, splits, ep_idx, images)  # enc: (61, D)

        z_pca = pca.transform(enc)  # (61, K)

        sig7 = signal_7_latent_speed(enc)
        sig8 = signal_8_direction_change(enc)
        sig10 = signal_10_action_delta(actions_np)
        sig6 = signal_6_bocpd(z_pca)

        for name, sig in (
            ("signal_6_bocpd", sig6), ("signal_7_speed", sig7),
            ("signal_8_direction", sig8), ("signal_10_action_delta", sig10),
        ):
            bounds = pick_changepoints(torch.from_numpy(sig.astype(np.float64)), n_boundaries=N_BOUNDARIES, min_seg=MIN_SEG)
            metric_b_results[name].append(score_boundaries(enc, bounds))

        metric_b_results["fixed"].append(score_boundaries(enc, fixed_boundaries()))

        rand_scores = [
            score_boundaries(enc, random_boundaries(60, N_BOUNDARIES, MIN_SEG, rand_rng))
            for _ in range(N_RANDOM_SEEDS)
        ]
        metric_b_results["random"].append(float(np.mean(rand_scores)))

        _, oracle_score = oracle_boundaries(enc, N_BOUNDARIES, MIN_SEG)
        metric_b_results["oracle"].append(oracle_score)

        n_done += 1
        if n_done % 25 == 0:
            print(f"{n_done}/{len(chosen)} episodes done", flush=True)

    summary = {}
    fixed_mean = float(np.mean(metric_b_results["fixed"]))
    oracle_mean = float(np.mean(metric_b_results["oracle"]))
    for name in signal_names + baseline_names:
        vals = np.array(metric_b_results[name])
        mean_val = float(vals.mean())
        improvement_over_fixed = (fixed_mean - mean_val) / fixed_mean if fixed_mean > 0 else 0.0
        oracle_reachability = (
            (fixed_mean - mean_val) / (fixed_mean - oracle_mean) if (fixed_mean - oracle_mean) > 1e-12 else None
        )
        summary[name] = {
            "mean_metric_b": mean_val,
            "median_metric_b": float(np.median(vals)),
            "improvement_over_fixed": improvement_over_fixed,
            "oracle_reachability": oracle_reachability,
            "n_episodes": len(vals),
        }
        print(
            f"{name}: mean={mean_val:.5f}, improvement_over_fixed={improvement_over_fixed*100:.1f}%, "
            f"oracle_reachability={oracle_reachability*100:.1f}%" if oracle_reachability is not None else
            f"{name}: mean={mean_val:.5f}"
        )

    out_path = os.path.join(HERE, "results_stage2_predictor_free.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "n_episodes_sampled": len(chosen)}, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
