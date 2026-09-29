# Boundary-signal study, item 2 (segment-length distribution) + item 3
# (combine Metric B gain and encoding loss to find the net-optimal
# min_seg). Uses bottom-up merging (the best-performing relaxation-
# sensitive algorithm from Amendment 3/min_seg_sweep) to get an actual
# segment-length distribution at each min_seg value, then applies the
# compression_loss.py length->loss lookup (interpolated) to estimate
# expected encoding loss per min_seg.
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from run_stage2_predictor_free import sample_episodes, get_episode_encodings_actions  # noqa: E402
from signal1_common import get_model  # noqa: E402
from metric_b import bottom_up_merge  # noqa: E402

HERE = os.path.dirname(__file__)
DATA_ROOT = os.path.join(HERE, "..", "..", "datasets", "r50_local", "r50_dataset")
N_EPISODES = 100
MIN_SEGS = [8, 6, 5, 4, 3, 2, 1]


def main():
    model = get_model()
    splits, chosen = sample_episodes(n_target=N_EPISODES, seed=3)  # SAME seed as min_seg_sweep.py -> same episodes
    images = np.load(os.path.join(DATA_ROOT, "main", "images.npy"), mmap_mode="r")
    print(f"segment length distribution: {len(chosen)} episodes, min_segs={MIN_SEGS}", flush=True)

    with open(os.path.join(HERE, "results_compression_loss.json")) as f:
        loss_table = json.load(f)
    loss_lengths = np.array(sorted(int(k) for k in loss_table.keys()))
    loss_means = np.array([loss_table[str(L)]["mean"] for L in loss_lengths])

    def interp_loss(L):
        return float(np.interp(L, loss_lengths, loss_means))

    summary = {}
    for ms in MIN_SEGS:
        all_lengths = []
        for ep_idx in chosen:
            enc, _ = get_episode_encodings_actions(model, splits, int(ep_idx), images)
            bounds = bottom_up_merge(enc, 5, ms)
            anchors = [0] + bounds + [60]
            lengths = [anchors[i + 1] - anchors[i] for i in range(len(anchors) - 1)]
            all_lengths.extend(lengths)
        all_lengths = np.array(all_lengths)
        expected_loss = float(np.mean([interp_loss(L) for L in all_lengths]))
        summary[ms] = {
            "min_length": int(all_lengths.min()), "max_length": int(all_lengths.max()),
            "p10": float(np.percentile(all_lengths, 10)), "median": float(np.percentile(all_lengths, 50)),
            "p90": float(np.percentile(all_lengths, 90)),
            "mean_expected_encoding_loss": expected_loss,
            "n_segments": len(all_lengths),
        }
        print(
            f"min_seg={ms}: length min={summary[ms]['min_length']} p10={summary[ms]['p10']:.1f} "
            f"median={summary[ms]['median']:.1f} p90={summary[ms]['p90']:.1f} max={summary[ms]['max_length']} "
            f"-> mean_expected_encoding_loss={expected_loss:.5f}",
            flush=True,
        )

    with open(os.path.join(HERE, "results_segment_length_vs_min_seg.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote results_segment_length_vs_min_seg.json")


if __name__ == "__main__":
    main()
