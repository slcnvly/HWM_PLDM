# Boundary-signal study, min_seg sweep: quantify exactly where the
# bottleneck is (per the user's corrected conclusion -- it's the min_seg=8
# constraint, not signal quality, per B6/B7 + Amendment 3's unconstrained
# vs constrained algorithm results). For min_seg in (8,6,5,4,3,2,1):
# oracle, bottom-up, top-down, fixed, random, and the 3 best-performing
# DENSE signals so far (13 chord-dev, 10 action-delta, 6 BOCPD -- excludes
# 16/16v2 since those are sparse/predictor-based and far more expensive to
# re-sweep 7x; noted explicitly). Encodings/signals computed ONCE per
# episode and reused across all 7 min_seg values (only the
# segmentation/extraction step depends on min_seg).
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from signal1_common import get_model  # noqa: E402
from pldm_envs.diverse_maze.adaptive_waypoints.segmentation import pick_changepoints  # noqa: E402
from run_stage2_predictor_free import sample_episodes, get_episode_encodings_actions  # noqa: E402
from signals_predictor_free import signal_6_bocpd, signal_10_action_delta  # noqa: E402
from signals_curvature import signal_13_chord_deviation  # noqa: E402
from metric_b import score_boundaries, oracle_boundaries, top_down_split, bottom_up_merge, fixed_boundaries, random_boundaries  # noqa: E402

HERE = os.path.dirname(__file__)
DATA_ROOT = os.path.join(HERE, "..", "..", "datasets", "r50_local", "r50_dataset")
N_EPISODES = 100
MIN_SEGS = [8, 6, 5, 4, 3, 2, 1]
N_BOUNDARIES = 5
N_RANDOM_SEEDS = 20
CKPT_PATH = os.path.join(HERE, "min_seg_sweep_checkpoint.npz")


def main():
    model = get_model()
    splits, chosen = sample_episodes(n_target=N_EPISODES, seed=3)
    images = np.load(os.path.join(DATA_ROOT, "main", "images.npy"), mmap_mode="r")
    print(f"min_seg sweep: {len(chosen)} episodes, min_segs={MIN_SEGS}", flush=True)

    from sklearn.decomposition import PCA
    rng0 = np.random.default_rng(0)
    pca_sample_eps = rng0.choice(chosen, size=min(60, len(chosen)), replace=False)
    pca_rows = []
    for ep_idx in pca_sample_eps:
        enc, _ = get_episode_encodings_actions(model, splits, int(ep_idx), images)
        pca_rows.append(enc)
    pca = PCA(n_components=10, random_state=0)
    pca.fit(np.concatenate(pca_rows, axis=0))
    del pca_rows
    print("PCA fit done", flush=True)

    names = ["oracle", "bottom_up", "top_down", "fixed", "random", "signal_13", "signal_10", "signal_6"]
    per_episode_scores = {ms: {name: [] for name in names} for ms in MIN_SEGS}

    rand_rng = np.random.default_rng(42)
    start = 0
    if os.path.exists(CKPT_PATH):
        ck = np.load(CKPT_PATH, allow_pickle=True)
        start = int(ck["next_idx"])
        per_episode_scores = ck["scores"].item()
        print(f"resuming from episode index {start}", flush=True)

    for pos in range(start, len(chosen)):
        ep_idx = int(chosen[pos])
        enc, actions_np = get_episode_encodings_actions(model, splits, ep_idx, images)
        z_pca = pca.transform(enc)

        sig13 = signal_13_chord_deviation(enc, w=5)
        sig10 = signal_10_action_delta(actions_np)
        sig6 = signal_6_bocpd(z_pca)
        fixed_score = score_boundaries(enc, fixed_boundaries())

        for ms in MIN_SEGS:
            per_episode_scores[ms]["fixed"].append(fixed_score)

            _, oscore = oracle_boundaries(enc, N_BOUNDARIES, ms)
            per_episode_scores[ms]["oracle"].append(oscore)

            per_episode_scores[ms]["bottom_up"].append(score_boundaries(enc, bottom_up_merge(enc, N_BOUNDARIES, ms)))
            per_episode_scores[ms]["top_down"].append(score_boundaries(enc, top_down_split(enc, N_BOUNDARIES, ms)))

            rand_scores = [score_boundaries(enc, random_boundaries(60, N_BOUNDARIES, ms, rand_rng)) for _ in range(N_RANDOM_SEEDS)]
            per_episode_scores[ms]["random"].append(float(np.mean(rand_scores)))

            for name, sig in (("signal_13", sig13), ("signal_10", sig10), ("signal_6", sig6)):
                bounds = pick_changepoints(torch.from_numpy(sig.astype(np.float64)), n_boundaries=N_BOUNDARIES, min_seg=ms)
                per_episode_scores[ms][name].append(score_boundaries(enc, bounds))

        if (pos + 1) % 10 == 0:
            np.savez(CKPT_PATH, next_idx=pos + 1, scores=per_episode_scores)
            print(f"{pos + 1}/{len(chosen)} episodes done (checkpointed)", flush=True)

    summary = {}
    for ms in MIN_SEGS:
        fixed_mean = float(np.mean(per_episode_scores[ms]["fixed"]))
        summary[ms] = {}
        for name in names:
            vals = np.array(per_episode_scores[ms][name])
            mean_val = float(vals.mean())
            improvement = (fixed_mean - mean_val) / fixed_mean if fixed_mean > 0 else 0.0
            summary[ms][name] = {"mean_metric_b": mean_val, "improvement_over_fixed": improvement}
        print(f"min_seg={ms}: " + ", ".join(f"{n}={summary[ms][n]['improvement_over_fixed']*100:.1f}%" for n in names))

    with open(os.path.join(HERE, "results_min_seg_sweep.json"), "w") as f:
        json.dump({"summary": summary, "per_episode_scores": {str(k): v for k, v in per_episode_scores.items()}, "n_episodes": len(chosen)}, f, indent=2)
    print("wrote results_min_seg_sweep.json")
    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)


if __name__ == "__main__":
    main()
