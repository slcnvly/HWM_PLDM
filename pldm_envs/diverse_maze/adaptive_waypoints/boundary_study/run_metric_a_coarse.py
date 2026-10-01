# Label-noise check for Metric A (see PROGRESS.md "Metric A label-noise check").
# Same 300-episode sample, same PCA, same candidates, same random-boundary RNG
# stream as run_metric_a.py. Per episode it scores every candidate against three
# label sets:
#   original          -- event_labels.py (must reproduce results_metric_a.json)
#   coarse_collapsed  -- event_labels_coarse.py, runs collapsed (PRIMARY)
#   coarse_per_frame  -- event_labels_coarse.py, no run collapsing
# and additionally re-runs dp_segmentation (oracle_boundaries) on a +/-2-step
# moving-average-smoothed latent trajectory to measure boundary stability.
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
from event_labels_coarse import label_episode_coarse  # noqa: E402

HERE = os.path.dirname(__file__)
DATA_ROOT = os.path.join(HERE, "..", "..", "datasets", "r50_local", "r50_dataset")
MIN_SEG = 8
N_BOUNDARIES = 5
N_RANDOM_SEEDS = 20
SMOOTH_RADIUS = 2
CKPT_PATH = os.path.join(HERE, "metric_a_coarse_checkpoint.npz")
BOUNDS_CACHE = os.path.join(HERE, "metric_a_coarse_boundaries.json")

SIGNAL_NAMES = ["signal_6", "signal_7", "signal_8", "signal_10", "signal_11", "signal_12", "signal_13"]
ALGO_NAMES = ["top_down", "bottom_up", "fixed", "random", "oracle"]
ALL_NAMES = SIGNAL_NAMES + ALGO_NAMES
LABEL_SETS = ["original", "coarse_collapsed", "coarse_per_frame"]
METRICS = ("precision", "recall", "f1")


def smooth(z, r=SMOOTH_RADIUS):
    c = np.concatenate([np.zeros((1, z.shape[1])), np.cumsum(z, axis=0)], axis=0)
    out = np.empty_like(z)
    for t in range(z.shape[0]):
        lo, hi = max(0, t - r), min(z.shape[0], t + r + 1)
        out[t] = (c[hi] - c[lo]) / (hi - lo)
    return out


def empty_prf1():
    d = {"overall": {m: [] for m in METRICS}}
    for e in EVENT_TYPES:
        d[e] = {m: [] for m in METRICS}
    return d


def main():
    model = get_model()
    splits, chosen = sample_episodes()
    images = np.load(os.path.join(DATA_ROOT, "main", "images.npy"), mmap_mode="r")
    maps = torch.load(os.path.join(DATA_ROOT, "main", "train_maps.pt"), weights_only=False)
    dp_cache = torch.load(os.path.join(DATA_ROOT, "main", "changepoints_minseg8.pt"), weights_only=False)
    print(f"Metric A coarse-label check: {len(chosen)} episodes", flush=True)

    from sklearn.decomposition import PCA
    rng0 = np.random.default_rng(0)
    pca_sample_eps = rng0.choice(chosen, size=min(60, len(chosen)), replace=False)
    pca_rows = [get_episode_encodings_actions(model, splits, int(e), images)[0] for e in pca_sample_eps]
    pca = PCA(n_components=10, random_state=0)
    pca.fit(np.concatenate(pca_rows, axis=0))
    del pca_rows
    print("PCA fit done", flush=True)

    state = {
        "auroc_ap": {ls: {n: {"auroc": [], "ap": []} for n in SIGNAL_NAMES} for ls in LABEL_SETS},
        "prf1": {ls: {n: empty_prf1() for n in ALL_NAMES} for ls in LABEL_SETS},
        "event_rate": {ls: {e: [] for e in EVENT_TYPES + ["pooled"]} for ls in LABEL_SETS},
        "events_per_ep": {ls: {e: [] for e in EVENT_TYPES + ["pooled"]} for ls in LABEL_SETS},
        "smooth": {"within2": [], "exact": [], "mean_abs_shift": [], "cache_match": []},
        "bounds": {},
    }
    rand_rng = np.random.default_rng(42)
    start = 0
    if os.path.exists(CKPT_PATH):
        ck = np.load(CKPT_PATH, allow_pickle=True)
        start = int(ck["next_idx"])
        state = ck["state"].item()
        rand_rng.bit_generator.state = ck["rng_state"].item()
        print(f"resuming from {start}", flush=True)

    for pos in range(start, len(chosen)):
        ep_idx = int(chosen[pos])
        enc, actions_np = get_episode_encodings_actions(model, splits, ep_idx, images)
        z_pca = pca.transform(enc)
        ep = splits[ep_idx]
        layout = maps[int(ep["map_idx"])].split("\\")
        lab = label_episode_coarse(ep["observations"], layout)
        labels = {"original": lab["original"], "coarse_collapsed": lab["collapsed"], "coarse_per_frame": lab["per_frame"]}

        for ls, ev in labels.items():
            pooled = np.any([ev[e] for e in EVENT_TYPES], axis=0)
            for e in EVENT_TYPES:
                state["event_rate"][ls][e].append(float(ev[e].mean()))
                state["events_per_ep"][ls][e].append(int(ev[e].sum()))
            state["event_rate"][ls]["pooled"].append(float(pooled.mean()))
            state["events_per_ep"][ls]["pooled"].append(int(pooled.sum()))

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
            for ls, ev in labels.items():
                auroc, ap = step_level_auroc_ap(s, ev)
                if auroc is not None:
                    state["auroc_ap"][ls][name]["auroc"].append(auroc)
                    state["auroc_ap"][ls][name]["ap"].append(ap)
        bounds["top_down"] = top_down_split(enc, N_BOUNDARIES, MIN_SEG)
        bounds["bottom_up"] = bottom_up_merge(enc, N_BOUNDARIES, MIN_SEG)
        bounds["fixed"] = fixed_boundaries()
        bounds["oracle"], _ = oracle_boundaries(enc, N_BOUNDARIES, MIN_SEG)

        rand_sets = [random_boundaries(60, N_BOUNDARIES, MIN_SEG, rand_rng) for _ in range(N_RANDOM_SEEDS)]
        for ls, ev in labels.items():
            acc = empty_prf1()
            for rb in rand_sets:
                r = boundary_level_prf1(rb, ev)
                for k in r:
                    for m in METRICS:
                        acc[k][m].append(r[k][m])
            for k in acc:
                for m in METRICS:
                    state["prf1"][ls]["random"][k][m].append(float(np.mean(acc[k][m])))
            for name in SIGNAL_NAMES + ["top_down", "bottom_up", "fixed", "oracle"]:
                r = boundary_level_prf1(bounds[name], ev)
                for k in r:
                    for m in METRICS:
                        state["prf1"][ls][name][k][m].append(r[k][m])

        orig_b = [int(b) for b in bounds["oracle"]]
        sm_b, _ = oracle_boundaries(smooth(enc), N_BOUNDARIES, MIN_SEG)
        sm_b = [int(b) for b in sm_b]
        state["smooth"]["within2"].append(float(np.mean([min(abs(b - s) for s in sm_b) <= 2 for b in orig_b])))
        state["smooth"]["exact"].append(float(np.mean([b in sm_b for b in orig_b])))
        state["smooth"]["mean_abs_shift"].append(float(np.mean([min(abs(b - s) for s in sm_b) for b in orig_b])))
        state["smooth"]["cache_match"].append(bool([int(x) for x in dp_cache[ep_idx]] == orig_b))
        state["bounds"][ep_idx] = {k: [int(x) for x in v] for k, v in bounds.items()}
        state["bounds"][ep_idx]["oracle_smoothed"] = sm_b

        if (pos + 1) % 20 == 0:
            np.savez(CKPT_PATH, next_idx=pos + 1, state=state, rng_state=rand_rng.bit_generator.state)
            print(f"{pos + 1}/{len(chosen)} done (checkpointed)", flush=True)

    summary = {"event_rate": {}, "events_per_ep": {}, "auroc_ap": {}, "prf1": {}}
    for ls in LABEL_SETS:
        summary["event_rate"][ls] = {e: float(np.mean(v)) for e, v in state["event_rate"][ls].items()}
        summary["events_per_ep"][ls] = {e: float(np.mean(v)) for e, v in state["events_per_ep"][ls].items()}
        summary["auroc_ap"][ls] = {
            n: {"auroc_mean": float(np.mean(d["auroc"])) if d["auroc"] else None,
                "ap_mean": float(np.mean(d["ap"])) if d["ap"] else None, "n": len(d["auroc"])}
            for n, d in state["auroc_ap"][ls].items()
        }
        summary["prf1"][ls] = {
            n: {k: {m: float(np.mean(state["prf1"][ls][n][k][m])) for m in METRICS} for k in state["prf1"][ls][n]}
            for n in ALL_NAMES
        }
    sm = state["smooth"]
    summary["smoothing"] = {
        "radius": SMOOTH_RADIUS,
        "frac_boundaries_within2": float(np.mean(sm["within2"])),
        "frac_boundaries_exact": float(np.mean(sm["exact"])),
        "mean_abs_shift": float(np.mean(sm["mean_abs_shift"])),
        "frac_episodes_all5_within2": float(np.mean([w == 1.0 for w in sm["within2"]])),
        "oracle_recompute_matches_cache": float(np.mean(sm["cache_match"])),
    }
    with open(os.path.join(HERE, "results_metric_a_coarse.json"), "w") as f:
        json.dump({"summary": summary, "prf1_raw": state["prf1"], "auroc_ap_raw": state["auroc_ap"],
                   "smooth_raw": sm, "n_episodes": len(chosen)}, f)
    with open(BOUNDS_CACHE, "w") as f:
        json.dump({str(k): v for k, v in state["bounds"].items()}, f)
    print(json.dumps({k: summary[k] for k in ("event_rate", "events_per_ep", "smoothing")}, indent=1))
    for ls in LABEL_SETS:
        print(f"\n[{ls}]")
        for n in ALL_NAMES:
            a = summary["auroc_ap"][ls].get(n)
            print(f"  {n:<10} F1={summary['prf1'][ls][n]['overall']['f1']:.4f}" + (f" AUROC={a['auroc_mean']:.4f}" if a else ""))
    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)


if __name__ == "__main__":
    main()
