# DISPLACEMENT_PAIRS M6: fixed, uniformly sampled evaluation frame set for future radius measurements.
# Same procedure as LATENT_LAW A2 (latent_law/ab_saturation.py:158-207), B1 (trained L1) only:
# 400 uniform frames per map (one rng over maps in sorted order), all within-map pairs, keep0,
# spike flags, curves(), radius(). Seed 0 must reproduce A2 (R_xy 1.35, R_maze 2, spike-excluded);
# seed 1 is the held-out evaluation set that gets committed.
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
AW = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(AW, "latent_law"))
import ab_saturation as ab  # noqa: E402


def run(seed, data, maps, images, starts, enc):
    rng = np.random.default_rng(seed)
    by_map = {}
    for e, ep in enumerate(data):
        by_map.setdefault(int(ep["map_idx"]), []).append(e)
    dists, mazes, xys, keeps, sps, frames = [], [], [], [], [], {}
    for m, eps in sorted(by_map.items()):
        L = maps[m].split("\\")
        dist = ab.bfs_all(L)
        cand = [(e, t) for e in eps for t in range(len(data[e]["observations"]))]
        pick = [cand[k] for k in np.sort(rng.choice(len(cand), ab.FRAMES_PER_MAP, replace=False))]
        frames[m] = [[int(e), int(t)] for e, t in pick]
        gidx = np.array([starts[e] + t for e, t in pick])
        raw = np.array(images[gidx])
        xy = np.array([data[e]["observations"][t, :2] for e, t in pick], dtype=np.float64)
        ij = [tuple(int(v) for v in c) for c in ab.obs_to_ij(xy)]
        ep_id = np.array([e for e, _ in pick]); t_id = np.array([t for _, t in pick])
        nb = []
        for e, t in pick:
            T = len(data[e]["observations"])
            nb += [starts[e] + max(t - 1, 0), starts[e] + min(t + 1, T - 1)]
        zc = ab.features("B1_trained", raw, enc)
        zn = ab.features("B1_trained", np.array(images[np.array(nb)]), enc).view(len(pick), 2, -1)
        dprev = (zc - zn[:, 0]).pow(2).mean(1).numpy()
        dnext = (zc - zn[:, 1]).pow(2).mean(1).numpy()
        has_prev = t_id > 0
        has_next = np.array([t < len(data[e]["observations"]) - 1 for e, t in pick])
        spike = (dprev > ab.SPIKE) & (dnext > ab.SPIKE) & has_prev & has_next
        iu = np.triu_indices(len(pick), 1)
        d_maze = np.array([dist[ij[a]].get(ij[b], -1) for a, b in zip(*iu)])
        d_xy = np.linalg.norm(xy[iu[0]] - xy[iu[1]], axis=1) / ab.GRID_SIZE
        keep0 = ~((d_maze == 0) & (ep_id[iu[0]] == ep_id[iu[1]]) & (np.abs(t_id[iu[0]] - t_id[iu[1]]) < 20))
        dists.append(ab.pair_mse(zc)[iu].numpy().astype(np.float32))
        mazes.append(d_maze); xys.append(d_xy); keeps.append(keep0 & (d_maze >= 0)); sps.append(spike[iu[0]] | spike[iu[1]])
    dist, d_maze, d_xy = np.concatenate(dists), np.concatenate(mazes), np.concatenate(xys)
    keep, sp = np.concatenate(keeps), np.concatenate(sps)
    out = {"seed": seed, "frames_per_map": ab.FRAMES_PER_MAP, "n_pairs": int(keep.sum()),
           "share_pairs_with_spike_frame": float(sp[keep].mean())}
    for subset, k in (("all_pairs", keep), ("no_spike_pairs", keep & ~sp)):
        c = ab.curves(dist, d_maze, d_xy, k)
        out[subset] = {"radius": ab.radius(c), "maze_curve_median": {d: v[1] for d, v in c["maze"].items()}}
    return out, frames


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 10)))
    data = torch.load(os.path.join(ab.DATA, "data.p"), weights_only=False)
    maps = torch.load(os.path.join(ab.DATA, "train_maps.pt"), weights_only=False)
    images = np.load(os.path.join(ab.DATA, "images.npy"), mmap_mode="r")
    starts = np.concatenate([[0], np.cumsum([len(e["observations"]) for e in data])])
    enc = ab.load_model().level1.backbone
    r0, _ = run(0, data, maps, images, starts, enc)
    a2 = json.load(open(os.path.join(AW, "latent_law", "results_ab_saturation.json")))["encoders"]["B1_trained"]
    r0["A2_reference_radius"] = {k: a2[k]["radius"] for k in ("all_pairs", "no_B1_spike_pairs")}
    print("seed0 reproduction:", r0["no_spike_pairs"]["radius"], "vs A2", r0["A2_reference_radius"]["no_B1_spike_pairs"], flush=True)
    r1, frames1 = run(1, data, maps, images, starts, enc)
    print("seed1:", r1["no_spike_pairs"]["radius"], "| all:", r1["all_pairs"]["radius"], flush=True)
    json.dump({"procedure": "LATENT_LAW A2 (ab_saturation.py), B1 trained L1, pretrained_baseline.ckpt, preprocess.py path",
               "seed": 1, "frames_per_map": ab.FRAMES_PER_MAP, "pairs": "all within-map pairs of these frames, keep0 rule of A2",
               "frames_episode_t_by_map": {int(m): v for m, v in frames1.items()}, "baseline_radius_current_checkpoint": r1},
              open(os.path.join(HERE, "eval_pairs_seed1.json"), "w"))
    json.dump({"seed0_reproduction": r0, "seed1": r1}, open(os.path.join(HERE, "results_dp_eval_set.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
