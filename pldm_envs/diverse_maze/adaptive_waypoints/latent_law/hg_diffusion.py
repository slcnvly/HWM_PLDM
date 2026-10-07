# HIERARCHY_GEOMETRY step 3 (+ P2c): n-step displacement in r50 (n <= 100) and in the
# 200-step random-action simulation (same generator/settings, same mazes and starts).
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ab_saturation as ab  # noqa: E402

NS = [3, 5, 10, 14, 20, 30, 40, 60, 80, 100, 160]
SIM200 = os.path.abspath(os.path.join(ab.AW, "..", "..", "..", "..", "experiments", "kaggle_sim_random_r50_200", "output", "sim_random_r50_200.npz"))


def displacement(trajs, layouts, n):
    xy, bfs = [], []
    cache = {}
    for o, L in zip(trajs, layouts):
        if len(o) <= n:
            continue
        key = "\\".join(L)
        if key not in cache:
            cache[key] = ab.bfs_all(L)
        dist = cache[key]
        ij = [tuple(int(v) for v in c) for c in ab.obs_to_ij(o)]
        d = np.linalg.norm(o[n:] - o[:-n], axis=1) / ab.GRID_SIZE
        xy.append(d)
        bfs += [dist[ij[t]].get(ij[t + n], np.nan) for t in range(len(o) - n)]
    if not xy:
        return None
    xy = np.concatenate(xy)
    bfs = np.array(bfs, dtype=float)
    return {"xy_median": float(np.median(xy)), "xy_p90": float(np.percentile(xy, 90)), "xy_mean": float(xy.mean()),
            "bfs_median": float(np.nanmedian(bfs)), "bfs_p90": float(np.nanpercentile(bfs, 90)), "n_windows": int(len(xy))}


def fit(ns, vals):
    a, b = np.polyfit(np.log(ns), np.log(vals), 1)
    return float(a), float(np.exp(b))


def main():
    data = torch.load(os.path.join(ab.DATA, "data.p"), weights_only=False)
    maps = torch.load(os.path.join(ab.DATA, "train_maps.pt"), weights_only=False)
    layouts = [maps[int(e["map_idx"])].split("\\") for e in data]
    r50 = [e["observations"][:, :2].astype(np.float64) for e in data]
    res = {"r50": {n: displacement(r50, layouts, n) for n in NS}}
    if os.path.exists(SIM200):
        sim = np.load(SIM200)
        for k, name in (("a1", "sim200_data_init_vel"), ("a2", "sim200_zero_init_vel")):
            tr = [sim[k][i][:, :2].astype(np.float64) for i in range(len(sim["ep"]))]
            lay = [layouts[int(e)] for e in sim["ep"]]
            res[name] = {n: displacement(tr, lay, n) for n in NS}
    # power-law fits on n in {20, 30, 40, 80} (prereg), xy median
    fits = {}
    for src in res:
        pts = [(n, res[src][n]["xy_median"]) for n in (20, 30, 40, 80) if res[src].get(n)]
        if len(pts) >= 3:
            alpha, c = fit([p[0] for p in pts], [p[1] for p in pts])
            short = [(n, res[src][n]["xy_median"]) for n in (3, 5, 10) if res[src].get(n)]
            a_short, _ = fit([p[0] for p in short], [p[1] for p in short])
            need_disp = 15.0 / (1.35 / 0.722)
            n_needed = (need_disp / c) ** (1 / alpha) if alpha > 0 else None
            fits[src] = {"alpha_long_20_80": alpha, "c": c, "alpha_short_3_10": a_short,
                         "extrapolated_n_for_radius_15_cells": n_needed, "required_median_displacement_cells": need_disp,
                         "n_for_radius_15_if_sqrt": (need_disp / res[src][14]["xy_median"]) ** 2 * 14}
    res["fits"] = fits
    json.dump(res, open(os.path.join(HERE, "results_hg_diffusion.json"), "w"), indent=2)
    for src in [k for k in res if k != "fits"]:
        print(src)
        for n in NS:
            v = res[src][n]
            if v:
                print(f"  n={n:4d} xy med {v['xy_median']:.3f} p90 {v['xy_p90']:.3f} bfs med {v['bfs_median']:.1f} p90 {v['bfs_p90']:.1f} (windows {v['n_windows']})")
    print(json.dumps(fits, indent=1))


if __name__ == "__main__":
    main()
