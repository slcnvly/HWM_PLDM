# Boundary-signal study, Metric A for all candidates at min_seg=8, on the
# same 300-episode sample as Stage 2 / Amendment 3 (signals
# 6,7,8,10,11,12,13, top_down, bottom_up, fixed, random, oracle). signal_16v2
# handled separately in run_metric_a_signal16v2.py (different, smaller
# sample -- kept as-is per "conservative, keep existing settings" while
# the user is away, see PROGRESS.md).
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
from signals_predictor_free import signal_6_bocpd, signal_7_latent_speed, signal_8_direction_change, signal_10_action_delta  # noqa: E402
from signals_curvature import signal_11_curvature, signal_12_normalized_curvature, signal_13_chord_deviation  # noqa: E402
from metric_b import oracle_boundaries, top_down_split, bottom_up_merge, fixed_boundaries, random_boundaries  # noqa: E402
from metric_a import step_level_auroc_ap, boundary_level_prf1, EVENT_TYPES  # noqa: E402
from event_labels import label_episode  # noqa: E402

HERE = os.path.dirname(__file__)
DATA_ROOT = os.path.join(HERE, "..", "..", "datasets", "r50_local", "r50_dataset")
MIN_SEG = 8
N_BOUNDARIES = 5
N_RANDOM_SEEDS = 20
CKPT_PATH = os.path.join(HERE, "metric_a_checkpoint.npz")

SIGNAL_NAMES = ["signal_6", "signal_7", "signal_8", "signal_10", "signal_11", "signal_12", "signal_13"]
ALGO_NAMES = ["top_down", "bottom_up", "fixed", "random", "oracle"]
ALL_NAMES = SIGNAL_NAMES + ALGO_NAMES


def main():
    model = get_model()
    splits, chosen = sample_episodes()  # default 300, seed=0 -- SAME as run_stage2_predictor_free.py
    images = np.load(os.path.join(DATA_ROOT, "main", "images.npy"), mmap_mode="r")
    maps = torch.load(os.path.join(DATA_ROOT, "main", "train_maps.pt"), weights_only=False)
    print(f"Metric A: {len(chosen)} episodes, min_seg={MIN_SEG}", flush=True)

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

    # per-episode storage: step-level (auroc,ap) lists for signals; boundary
    # P/R/F1 (overall + per type) lists for ALL candidates.
    auroc_ap = {name: {"auroc": [], "ap": []} for name in SIGNAL_NAMES}
    prf1 = {name: {"overall": {"precision": [], "recall": [], "f1": []}} for name in ALL_NAMES}
    for name in ALL_NAMES:
        for etype in EVENT_TYPES:
            prf1[name][etype] = {"precision": [], "recall": [], "f1": []}

    rand_rng = np.random.default_rng(42)
    start = 0
    if os.path.exists(CKPT_PATH):
        ck = np.load(CKPT_PATH, allow_pickle=True)
        start = int(ck["next_idx"])
        auroc_ap = ck["auroc_ap"].item()
        prf1 = ck["prf1"].item()
        print(f"resuming from episode index {start}", flush=True)

    for pos in range(start, len(chosen)):
        ep_idx = int(chosen[pos])
        enc, actions_np = get_episode_encodings_actions(model, splits, ep_idx, images)
        z_pca = pca.transform(enc)

        ep = splits[ep_idx]
        layout = maps[int(ep["map_idx"])].split("\\")
        events = label_episode(ep["observations"], layout)

        sig = {
            "signal_6": signal_6_bocpd(z_pca),
            "signal_7": signal_7_latent_speed(enc),
            "signal_8": signal_8_direction_change(enc),
            "signal_10": signal_10_action_delta(actions_np),
            "signal_11": signal_11_curvature(enc),
            "signal_12": signal_12_normalized_curvature(enc),
            "signal_13": signal_13_chord_deviation(enc, w=5),
        }

        bounds = {}
        for name, s in sig.items():
            bounds[name] = pick_changepoints(torch.from_numpy(s.astype(np.float64)), n_boundaries=N_BOUNDARIES, min_seg=MIN_SEG)
            auroc, ap = step_level_auroc_ap(s, events)
            if auroc is not None:
                auroc_ap[name]["auroc"].append(auroc)
                auroc_ap[name]["ap"].append(ap)

        bounds["top_down"] = top_down_split(enc, N_BOUNDARIES, MIN_SEG)
        bounds["bottom_up"] = bottom_up_merge(enc, N_BOUNDARIES, MIN_SEG)
        bounds["fixed"] = fixed_boundaries()
        _, _ = None, None
        bounds["oracle"], _ = oracle_boundaries(enc, N_BOUNDARIES, MIN_SEG)

        # random: average P/R/F1 across 20 seeds (not just Metric B score)
        random_prf1_accum = {"overall": {"precision": [], "recall": [], "f1": []}}
        for etype in EVENT_TYPES:
            random_prf1_accum[etype] = {"precision": [], "recall": [], "f1": []}
        for _ in range(N_RANDOM_SEEDS):
            rb = random_boundaries(60, N_BOUNDARIES, MIN_SEG, rand_rng)
            r = boundary_level_prf1(rb, events)
            for key in r:
                for metric in ("precision", "recall", "f1"):
                    random_prf1_accum[key][metric].append(r[key][metric])
        for key in random_prf1_accum:
            for metric in ("precision", "recall", "f1"):
                prf1["random"][key][metric].append(float(np.mean(random_prf1_accum[key][metric])))

        for name in SIGNAL_NAMES + ["top_down", "bottom_up", "fixed", "oracle"]:
            r = boundary_level_prf1(bounds[name], events)
            for key in r:
                for metric in ("precision", "recall", "f1"):
                    prf1[name][key][metric].append(r[key][metric])

        if (pos + 1) % 20 == 0:
            np.savez(CKPT_PATH, next_idx=pos + 1, auroc_ap=auroc_ap, prf1=prf1)
            print(f"{pos + 1}/{len(chosen)} episodes done (checkpointed)", flush=True)

    # summarize
    summary = {"auroc_ap": {}, "prf1": {}}
    for name in SIGNAL_NAMES:
        summary["auroc_ap"][name] = {
            "auroc_mean": float(np.mean(auroc_ap[name]["auroc"])) if auroc_ap[name]["auroc"] else None,
            "ap_mean": float(np.mean(auroc_ap[name]["ap"])) if auroc_ap[name]["ap"] else None,
            "n": len(auroc_ap[name]["auroc"]),
        }
    for name in ALL_NAMES:
        summary["prf1"][name] = {}
        for key in prf1[name]:
            summary["prf1"][name][key] = {m: float(np.mean(prf1[name][key][m])) for m in ("precision", "recall", "f1")}

    with open(os.path.join(HERE, "results_metric_a.json"), "w") as f:
        json.dump({"summary": summary, "auroc_ap_raw": auroc_ap, "prf1_raw": prf1, "n_episodes": len(chosen)}, f, indent=2)
    print("wrote results_metric_a.json")
    for name in ALL_NAMES:
        f1 = summary["prf1"][name]["overall"]["f1"]
        print(f"{name}: overall F1={f1:.4f}" + (f", AUROC={summary['auroc_ap'][name]['auroc_mean']:.4f}, AP={summary['auroc_ap'][name]['ap_mean']:.4f}" if name in SIGNAL_NAMES else ""))
    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)


if __name__ == "__main__":
    main()
