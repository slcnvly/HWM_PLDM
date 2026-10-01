# Boundary-signal study, Metric A for signal_16_v2 (obs+proprio), on its
# OWN existing 50-episode sample (seed=7) -- kept as-is (not expanded to
# 300) per the "conservative, keep existing settings" instruction while
# the user is away: re-running signal 16's expensive 14-step rollout at
# 300-episode scale would cost substantially more compute for a signal
# already confirmed to lose on Metric B, and this is exactly the kind of
# judgment call to resolve conservatively rather than escalate scope
# unilaterally. This means signal_16v2's Metric A numbers have a smaller,
# noisier sample than the other 12 candidates -- noted explicitly in
# RESULTS.md, not hidden.
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from signal1_common import get_model  # noqa: E402
from pldm_envs.diverse_maze.adaptive_waypoints.segmentation import pick_changepoints  # noqa: E402
from run_stage2_predictor_free import sample_episodes  # noqa: E402
from signal_16_v2_rollout_matched import compute_episode, N_EPISODES  # noqa: E402
from metric_a import step_level_auroc_ap, boundary_level_prf1, EVENT_TYPES  # noqa: E402
from event_labels import label_episode  # noqa: E402

HERE = os.path.dirname(__file__)
DATA_ROOT = os.path.join(HERE, "..", "..", "datasets", "r50_local", "r50_dataset")
MIN_SEG = 8
N_BOUNDARIES = 5


def main():
    model = get_model()
    splits, chosen = sample_episodes(n_target=N_EPISODES, seed=7)  # SAME as signal_16_v2_rollout_matched.py
    images = np.load(os.path.join(DATA_ROOT, "main", "images.npy"), mmap_mode="r")
    maps = torch.load(os.path.join(DATA_ROOT, "main", "train_maps.pt"), weights_only=False)
    print(f"Metric A for signal_16_v2: {len(chosen)} episodes", flush=True)

    auroc_ap = {"signal_16v2_obs": {"auroc": [], "ap": []}, "signal_16v2_proprio": {"auroc": [], "ap": []}}
    prf1 = {name: {"overall": {"precision": [], "recall": [], "f1": []}} for name in auroc_ap}
    for name in prf1:
        for etype in EVENT_TYPES:
            prf1[name][etype] = {"precision": [], "recall": [], "f1": []}

    for pos, ep_idx in enumerate(chosen):
        enc_np, obs_sig, proprio_sig = compute_episode(model, splits, int(ep_idx), images, seed=pos)
        ep = splits[int(ep_idx)]
        layout = maps[int(ep["map_idx"])].split("\\")
        events = label_episode(ep["observations"], layout)

        for name, sig in (("signal_16v2_obs", obs_sig), ("signal_16v2_proprio", proprio_sig)):
            bounds = pick_changepoints(torch.from_numpy(sig.astype(np.float64)), n_boundaries=N_BOUNDARIES, min_seg=MIN_SEG)
            auroc, ap = step_level_auroc_ap(sig, events)
            if auroc is not None:
                auroc_ap[name]["auroc"].append(auroc)
                auroc_ap[name]["ap"].append(ap)
            r = boundary_level_prf1(bounds, events)
            for key in r:
                for metric in ("precision", "recall", "f1"):
                    prf1[name][key][metric].append(r[key][metric])

        if (pos + 1) % 10 == 0:
            print(f"{pos + 1}/{len(chosen)} done", flush=True)

    summary = {"auroc_ap": {}, "prf1": {}}
    for name in auroc_ap:
        summary["auroc_ap"][name] = {
            "auroc_mean": float(np.mean(auroc_ap[name]["auroc"])) if auroc_ap[name]["auroc"] else None,
            "ap_mean": float(np.mean(auroc_ap[name]["ap"])) if auroc_ap[name]["ap"] else None,
            "n": len(auroc_ap[name]["auroc"]),
        }
        summary["prf1"][name] = {key: {m: float(np.mean(prf1[name][key][m])) for m in ("precision", "recall", "f1")} for key in prf1[name]}
        print(f"{name}: overall F1={summary['prf1'][name]['overall']['f1']:.4f}, AUROC={summary['auroc_ap'][name]['auroc_mean']:.4f}, AP={summary['auroc_ap'][name]['ap_mean']:.4f}")

    with open(os.path.join(HERE, "results_metric_a_signal16v2.json"), "w") as f:
        json.dump({"summary": summary, "n_episodes": len(chosen)}, f, indent=2)
    print("wrote results_metric_a_signal16v2.json")


if __name__ == "__main__":
    main()
