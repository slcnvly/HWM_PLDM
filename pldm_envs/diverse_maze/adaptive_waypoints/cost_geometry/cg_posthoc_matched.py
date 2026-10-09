# POST-HOC (not preregistered): position-matched comparison. For maps 0-4, take the same 400 frames
# per map as LATENT_LAW A2 (rng seed 0 over maps in sorted order), and render a synthetic 4-px disc
# (agent colour) at each frame's true agent position on that map's agent-free background. Real and
# synthetic frames share positions and pairs, so any radius difference comes from rendering only.
import json
import os

import numpy as np
import torch

import cg_common as cg

ab = cg.ab
HERE = os.path.dirname(os.path.abspath(__file__))
A = np.array([[3.00092599e-03, 9.46012472e+00], [-9.45758650e+00, 1.11380388e-02], [8.91619347e+01, 7.78860918e+00]])
N_MAPS = 5


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 10)))
    data = torch.load(os.path.join(ab.DATA, "data.p"), weights_only=False)
    maps = torch.load(os.path.join(ab.DATA, "train_maps.pt"), weights_only=False)
    images = np.load(os.path.join(ab.DATA, "images.npy"), mmap_mode="r")
    starts = np.concatenate([[0], np.cumsum([len(e["observations"]) for e in data])])
    bgs = np.load(os.path.join(HERE, "cg_backgrounds.npz"))
    encs = cg.load_encoders()
    spike_all = np.load(os.path.join(cg.AW, "displacement_pairs", "dp_latents.npz"))["spike"]
    rng = np.random.default_rng(0)
    by_map = {}
    for e, ep in enumerate(data):
        by_map.setdefault(int(ep["map_idx"]), []).append(e)
    pool = {(src, n): [] for src in ("real", "synth") for n in cg.REPS}
    pm, px, pk, ps = [], [], [], []
    for m, eps in sorted(by_map.items()):
        cand = [(e, t) for e in eps for t in range(len(data[e]["observations"]))]
        pick = [cand[k] for k in np.sort(rng.choice(len(cand), ab.FRAMES_PER_MAP, replace=False))]  # A2 draw order
        if m >= N_MAPS:
            continue
        L = maps[m].split("\\")
        dist = ab.bfs_all(L)
        raw = np.array(images[np.array([starts[e] + t for e, t in pick])])
        xy = np.array([data[e]["observations"][t, :2] for e, t in pick], dtype=np.float64)
        rc = np.c_[xy, np.ones(len(xy))] @ A
        bg = bgs[str(m)].astype(np.float64)
        synth = np.stack([cg.compose(bg, cg.coverage("disc", 4, center=p)) for p in rc])
        ij = [tuple(int(v) for v in c) for c in ab.obs_to_ij(xy)]
        ep_id = np.array([e for e, _ in pick]); t_id = np.array([t for _, t in pick])
        spike = np.array([spike_all[e][t] for e, t in pick])
        iu = np.triu_indices(len(pick), 1)
        pm.append(np.array([dist[ij[a]].get(ij[b], -1) for a, b in zip(*iu)]))
        px.append(np.linalg.norm(xy[iu[0]] - xy[iu[1]], axis=1) / ab.GRID_SIZE)
        pk.append(~((pm[-1] == 0) & (ep_id[iu[0]] == ep_id[iu[1]]) & (np.abs(t_id[iu[0]] - t_id[iu[1]]) < 20)) & (pm[-1] >= 0))
        ps.append(spike[iu[0]] | spike[iu[1]])
        for n in cg.REPS:
            pool[("real", n)].append(ab.pair_mse(cg.feats(n, raw, encs))[iu].numpy())
            pool[("synth", n)].append(ab.pair_mse(cg.feats(n, synth, encs))[iu].numpy())
        print("map", m, flush=True)
    d_maze, d_xy, keep, sp = np.concatenate(pm), np.concatenate(px), np.concatenate(pk), np.concatenate(ps)
    res = {"note": "POST-HOC position-matched real vs synthetic 4-px disc, maps 0-4, A2 frames", "n_pairs": int(keep.sum())}
    for n in cg.REPS:
        res[n] = {}
        for src in ("real", "synth"):
            dv = np.concatenate(pool[(src, n)])
            for subset, k in (("all_pairs", keep), ("no_B1_spike_pairs", keep & ~sp)):
                c = ab.curves(dv, d_maze, d_xy, k)
                r = ab.radius(c)
                res[n][f"{src}|{subset}"] = {"R_xy": r.get("R_xy"), "R_maze": r.get("R_maze"), "xy_base": r.get("xy_base"), "xy_plateau": r.get("xy_plateau")}
        print(n, {k: (v["R_xy"], v["R_maze"]) for k, v in res[n].items()}, flush=True)
    json.dump(res, open(os.path.join(HERE, "results_cg_posthoc_matched.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
