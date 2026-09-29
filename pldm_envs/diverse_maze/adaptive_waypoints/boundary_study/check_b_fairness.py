# Boundary-signal study, Part B fairness checks (post Amendment-2/3 gate).
# B5 (oracle respects min_seg=8): already confirmed via ad-hoc check (see
# PROGRESS.md) -- min gap == 8 across 20 episodes, no violation, nothing to
# recompute.
# B6: procedural diagnostic -- can pick_changepoints even recover an
# artificial signal that's 1 exactly at the oracle boundaries and 0
# elsewhere? If not, the bottleneck is the extraction procedure, not the
# signal.
# B7: min_seg sensitivity -- how does oracle's improvement over fixed
# change as min_seg varies (8, 5, 3, 1)?
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
from metric_b import oracle_boundaries, score_boundaries, fixed_boundaries  # noqa: E402

HERE = os.path.dirname(__file__)
DATA_ROOT = os.path.join(HERE, "..", "..", "datasets", "r50_local", "r50_dataset")
N_EPISODES = 50


def b6_recovery_check(model, splits, chosen, images):
    print("\n=== B6: procedural diagnostic (can peak-picking recover oracle boundaries?) ===")
    n_exact = 0
    n_within2 = 0
    for ep_idx in chosen:
        enc, _ = get_episode_encodings_actions(model, splits, int(ep_idx), images)
        oracle_b, _ = oracle_boundaries(enc, 5, 8)
        synthetic = np.zeros(60)
        for b in oracle_b:
            synthetic[b] = 1.0  # index b in the 60-length series IS frame b (0-indexed boundary position)
        recovered = pick_changepoints(torch.from_numpy(synthetic), n_boundaries=5, min_seg=8)
        exact = recovered == oracle_b
        within2 = all(min(abs(r - o) for o in oracle_b) <= 2 for r in recovered)
        n_exact += int(exact)
        n_within2 += int(within2)
    result = {
        "n_episodes": len(chosen),
        "exact_recovery_rate": n_exact / len(chosen),
        "within_2steps_recovery_rate": n_within2 / len(chosen),
    }
    print(f"exact recovery: {n_exact}/{len(chosen)} ({result['exact_recovery_rate']*100:.1f}%)")
    print(f"within +/-2 steps: {n_within2}/{len(chosen)} ({result['within_2steps_recovery_rate']*100:.1f}%)")
    return result


def b7_min_seg_sensitivity(model, splits, chosen, images):
    print("\n=== B7: min_seg sensitivity ===")
    results = {}
    for min_seg in (8, 5, 3, 1):
        fixed_scores, oracle_scores = [], []
        for ep_idx in chosen:
            enc, _ = get_episode_encodings_actions(model, splits, int(ep_idx), images)
            fixed_scores.append(score_boundaries(enc, fixed_boundaries()))
            _, oscore = oracle_boundaries(enc, 5, min_seg)
            oracle_scores.append(oscore)
        fixed_mean = float(np.mean(fixed_scores))
        oracle_mean = float(np.mean(oracle_scores))
        improvement = (fixed_mean - oracle_mean) / fixed_mean if fixed_mean > 0 else 0.0
        results[min_seg] = {"fixed_mean": fixed_mean, "oracle_mean": oracle_mean, "oracle_improvement_over_fixed": improvement}
        print(f"min_seg={min_seg}: fixed={fixed_mean:.5f}, oracle={oracle_mean:.5f}, improvement={improvement*100:.1f}%")
    return results


def main():
    model = get_model()
    splits, chosen = sample_episodes(n_target=N_EPISODES, seed=2)
    images = np.load(os.path.join(DATA_ROOT, "main", "images.npy"), mmap_mode="r")
    print(f"using {len(chosen)} episodes for B6/B7", flush=True)

    b6 = b6_recovery_check(model, splits, chosen, images)
    b7 = b7_min_seg_sensitivity(model, splits, chosen, images)

    with open(os.path.join(HERE, "results_b_fairness.json"), "w") as f:
        json.dump({"b6_recovery": b6, "b7_min_seg_sensitivity": b7, "n_episodes": len(chosen)}, f, indent=2)
    print("wrote results_b_fairness.json")


if __name__ == "__main__":
    main()
