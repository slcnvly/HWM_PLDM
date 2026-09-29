# Boundary-signal study, item 2 (compression-loss cost of relaxing
# min_seg): quantify the information loss from resampling a variable-length
# segment to a fixed 10 steps (DESIGN.md decision 2's actual mechanism --
# reusing segmentation.py's own resample_to_fixed_length for fidelity) and
# back, as a function of segment length L. This is presumably why min_seg
# existed in the first place -- quantifying its cost directly.
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from run_stage2_predictor_free import sample_episodes, get_episode_encodings_actions  # noqa: E402
from signal1_common import get_model  # noqa: E402
from pldm_envs.diverse_maze.adaptive_waypoints.segmentation import resample_to_fixed_length  # noqa: E402

HERE = os.path.dirname(__file__)
DATA_ROOT = os.path.join(HERE, "..", "..", "datasets", "r50_local", "r50_dataset")
N_EPISODES = 100
LENGTHS = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 58]
N_SAMPLES_PER_LENGTH_PER_EPISODE = 3  # random start positions tried per (episode, length)


def round_trip_error(z, a, length):
    """z: (61, D). Segment z[a:a+length+1] (length+1 points) -> compress to
    10 via linear interpolation -> decompress back to length+1 -> MSE
    against the original (mean over interior points and dims, matching
    Metric B's own convention)."""
    seg = torch.from_numpy(z[a : a + length + 1]).float()  # (length+1, D)
    compressed = resample_to_fixed_length(seg, 10)  # (10, D)
    reconstructed = resample_to_fixed_length(compressed, length + 1)  # (length+1, D)
    err = (seg - reconstructed).pow(2).mean().item()
    return err


def main():
    model = get_model()
    splits, chosen = sample_episodes(n_target=N_EPISODES, seed=5)
    images = np.load(os.path.join(DATA_ROOT, "main", "images.npy"), mmap_mode="r")
    print(f"compression loss: {len(chosen)} episodes, lengths={LENGTHS}", flush=True)

    rng = np.random.default_rng(0)
    errors_by_length = {L: [] for L in LENGTHS}

    for pos, ep_idx in enumerate(chosen):
        enc, _ = get_episode_encodings_actions(model, splits, int(ep_idx), images)
        for L in LENGTHS:
            max_start = 60 - L
            if max_start < 0:
                continue
            n_tries = min(N_SAMPLES_PER_LENGTH_PER_EPISODE, max_start + 1)
            starts = rng.choice(max_start + 1, size=n_tries, replace=False)
            for a in starts:
                errors_by_length[L].append(round_trip_error(enc, int(a), L))

        if (pos + 1) % 25 == 0:
            print(f"{pos + 1}/{len(chosen)} episodes done", flush=True)

    summary = {}
    for L in LENGTHS:
        vals = np.array(errors_by_length[L])
        if len(vals) == 0:
            continue
        summary[L] = {
            "mean": float(vals.mean()), "median": float(np.median(vals)),
            "p10": float(np.percentile(vals, 10)), "p90": float(np.percentile(vals, 90)),
            "n_samples": len(vals),
        }
        print(f"length={L}: mean_roundtrip_mse={summary[L]['mean']:.6f}, median={summary[L]['median']:.6f}, n={len(vals)}")

    with open(os.path.join(HERE, "results_compression_loss.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote results_compression_loss.json")


if __name__ == "__main__":
    main()
