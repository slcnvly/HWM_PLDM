# COST_GEOMETRY M2: re-run the LATENT_LAW A2/B procedure on real r50 frames (seed 0, all
# representations) with the same helpers (latent_law/ab_saturation.py main loop, lines 158-207),
# writing to a new file (the original results_ab_saturation.json is not touched).
import json
import os

import numpy as np
import torch

import cg_common as cg

ab = cg.ab
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 10)))
    data = torch.load(os.path.join(ab.DATA, "data.p"), weights_only=False)
    maps = torch.load(os.path.join(ab.DATA, "train_maps.pt"), weights_only=False)
    images = np.load(os.path.join(ab.DATA, "images.npy"), mmap_mode="r")
    starts = np.concatenate([[0], np.cumsum([len(e["observations"]) for e in data])])
    encs = cg.load_encoders()
    rng = np.random.default_rng(0)
    by_map = {}
    for e, ep in enumerate(data):
        by_map.setdefault(int(ep["map_idx"]), []).append(e)
    pool = {n: [] for n in cg.REPS}
    pm, px, pk, ps = [], [], [], []
    for m, eps in sorted(by_map.items()):
        L = maps[m].split("\\")
        dist = ab.bfs_all(L)
        cand = [(e, t) for e in eps for t in range(len(data[e]["observations"]))]
        pick = [cand[k] for k in np.sort(rng.choice(len(cand), ab.FRAMES_PER_MAP, replace=False))]
        raw = np.array(images[np.array([starts[e] + t for e, t in pick])])
        xy = np.array([data[e]["observations"][t, :2] for e, t in pick], dtype=np.float64)
        ij = [tuple(int(v) for v in c) for c in ab.obs_to_ij(xy)]
        ep_id = np.array([e for e, _ in pick]); t_id = np.array([t for _, t in pick])
        nb = []
        for e, t in pick:
            T = len(data[e]["observations"])
            nb += [starts[e] + max(t - 1, 0), starts[e] + min(t + 1, T - 1)]
        zc = cg.feats("B1_trained", raw, encs)
        zn = cg.feats("B1_trained", np.array(images[np.array(nb)]), encs).view(len(pick), 2, -1)
        dprev = (zc - zn[:, 0]).pow(2).mean(1).numpy(); dnext = (zc - zn[:, 1]).pow(2).mean(1).numpy()
        has_next = np.array([t < len(data[e]["observations"]) - 1 for e, t in pick])
        spike = (dprev > ab.SPIKE) & (dnext > ab.SPIKE) & (t_id > 0) & has_next
        iu = np.triu_indices(len(pick), 1)
        d_maze = np.array([dist[ij[a]].get(ij[b], -1) for a, b in zip(*iu)])
        d_xy = np.linalg.norm(xy[iu[0]] - xy[iu[1]], axis=1) / ab.GRID_SIZE
        keep0 = ~((d_maze == 0) & (ep_id[iu[0]] == ep_id[iu[1]]) & (np.abs(t_id[iu[0]] - t_id[iu[1]]) < 20))
        for n in cg.REPS:
            Fm = zc if n == "B1_trained" else cg.feats(n, raw, encs)
            pool[n].append(ab.pair_mse(Fm)[iu].numpy().astype(np.float32))
        pm.append(d_maze); px.append(d_xy); pk.append(keep0 & (d_maze >= 0)); ps.append(spike[iu[0]] | spike[iu[1]])
        print(f"map {m}", flush=True)
    d_maze, d_xy, keep, sp = np.concatenate(pm), np.concatenate(px), np.concatenate(pk), np.concatenate(ps)
    ref = json.load(open(os.path.join(cg.AW, "latent_law", "results_ab_saturation.json")))["encoders"]
    res = {}
    for n in cg.REPS:
        dist = np.concatenate(pool[n])
        res[n] = {}
        for subset, k in (("all_pairs", keep), ("no_B1_spike_pairs", keep & ~sp)):
            c = ab.curves(dist, d_maze, d_xy, k)
            res[n][subset] = {"radius": ab.radius(c), "latent_law_reference": ref[n][subset]["radius"]}
        print(n, res[n]["no_B1_spike_pairs"], flush=True)
    json.dump(res, open(os.path.join(HERE, "results_cg_a2_repro.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
