# C1: does the planner's latent goal distance track real (maze) distance?
# For each r50 main trajectory (same 300-episode stratified sample as the boundary
# study), goal = the episode's last frame (its image equals a zero-velocity render
# of that position: maze rendering depends on position only, maze_draw.py:57-61).
# Latent distance = mean squared difference of L1 obs_component (the planner's cost,
# mppi_planner.py:260-270), with inputs normalized exactly as in evaluation.
# Real distance = BFS distance in grid cells (4-connected open cells).
import json
import os
import sys
from collections import deque

import numpy as np
import torch
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "boundary_study")))
from train_location_prober import load_model, DATA  # noqa: E402
from pldm_envs.diverse_maze.adaptive_waypoints.preprocess import normalize_images, normalize_proprio_vel  # noqa: E402
from run_stage2_predictor_free import sample_episodes  # noqa: E402
from grid_utils import obs_to_ij, open_neighbor_count  # noqa: E402

K = 5  # step lag for the "real decreases but latent increases" test


def bfs(layout, goal):
    rows, cols = len(layout), len(layout[0])
    dist = np.full((rows, cols), -1, dtype=int)
    dist[goal] = 0
    q = deque([goal])
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols and layout[ni][nj] != "#" and dist[ni, nj] < 0:
                dist[ni, nj] = dist[i, j] + 1
                q.append((ni, nj))
    return dist


def cell_type(layout, i, j):
    n = open_neighbor_count(layout, i, j)
    if n >= 3:
        return "junction"
    if n <= 1:
        return "dead_end"
    open_dirs = [(di, dj) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))
                 if layout[i + di][j + dj] != "#"]
    return "straight" if open_dirs[0][0] == -open_dirs[1][0] and open_dirs[0][1] == -open_dirs[1][1] else "corner"


def main():
    torch.set_num_threads(10)
    model = load_model()
    splits, chosen = sample_episodes()
    images = np.load(os.path.join(DATA, "main", "images.npy"), mmap_mode="r")
    maps = torch.load(os.path.join(DATA, "main", "train_maps.pt"), weights_only=False)
    starts = np.concatenate([[0], np.cumsum([len(e["observations"]) for e in splits])])

    rhos, pooled_bfs, pooled_lat = [], [], []
    n_pairs_dec = n_viol = 0
    type_occ, type_dec, type_viol = {}, {}, {}
    bad_bfs = 0
    for n, e in enumerate(chosen):
        ep = splits[e]
        T = len(ep["observations"])
        img = normalize_images(torch.from_numpy(np.array(images[starts[e]:starts[e] + T])).float().permute(0, 3, 1, 2))
        pv = normalize_proprio_vel(torch.from_numpy(ep["observations"][:, 2:4]).float())
        with torch.no_grad():
            obs = model.level1.backbone(img, proprio=pv).obs_component.flatten(1)
        lat = (obs - obs[-1]).pow(2).mean(dim=1).numpy()  # (T,)
        layout = maps[int(ep["map_idx"])].split("\\")
        ij = obs_to_ij(ep["observations"][:, :2].astype(np.float64))
        dist = bfs(layout, tuple(int(x) for x in ij[-1]))
        d = np.array([dist[int(a), int(b)] for a, b in ij])
        if (d < 0).any():
            bad_bfs += 1
            continue
        tt = slice(0, T - 1)  # exclude the goal frame itself
        if np.std(d[tt]) > 0:
            rhos.append(spearmanr(d[tt], lat[tt]).correlation)
        pooled_bfs.append(d[tt])
        pooled_lat.append(lat[tt])
        for t in range(T - 1 - K):
            ctype = cell_type(layout, int(ij[t + K, 0]), int(ij[t + K, 1]))
            type_occ[ctype] = type_occ.get(ctype, 0) + 1
            if d[t + K] < d[t]:
                n_pairs_dec += 1
                type_dec[ctype] = type_dec.get(ctype, 0) + 1
                if lat[t + K] > lat[t]:
                    n_viol += 1
                    type_viol[ctype] = type_viol.get(ctype, 0) + 1
        if (n + 1) % 50 == 0:
            print(f"{n + 1}/{len(chosen)}", flush=True)

    pb, pl = np.concatenate(pooled_bfs), np.concatenate(pooled_lat)
    res = {
        "n_episodes": len(rhos), "episodes_skipped_bfs_unreachable": bad_bfs, "lag_K": K,
        "spearman_per_episode_median": float(np.median(rhos)),
        "spearman_per_episode_iqr": [float(np.percentile(rhos, 25)), float(np.percentile(rhos, 75))],
        "spearman_per_episode_frac_negative": float(np.mean(np.array(rhos) < 0)),
        "spearman_pooled": float(spearmanr(pb, pl).correlation),
        "frac_real_decrease_but_latent_increase": n_viol / max(n_pairs_dec, 1),
        "n_real_decrease_pairs": n_pairs_dec,
        "by_cell_type": {
            c: {"occupancy_share": type_occ[c] / sum(type_occ.values()),
                "share_of_violations": type_viol.get(c, 0) / max(n_viol, 1),
                "violation_rate_given_real_decrease": type_viol.get(c, 0) / max(type_dec.get(c, 0), 1),
                "n_real_decrease": type_dec.get(c, 0)}
            for c in sorted(type_occ)
        },
        "latent_dist_by_bfs_cells": {int(b): float(np.median(pl[pb == b])) for b in np.unique(pb) if (pb == b).sum() >= 50},
    }
    json.dump(res, open(os.path.join(HERE, "results_c1_latent_vs_maze_distance.json"), "w"), indent=2)  # previous run kept as *_prev.json
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
