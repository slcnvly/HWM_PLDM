# Online-map feasibility, stage 2 (ONLINE_MAP_PREREG.md): can latent distance
# recognise the same place? r50 main, one map at a time (RAM), normalized inputs.
import json
import os
import sys
from collections import deque

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
AW = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(AW, "followup"))
from common import DATA, D_OBS_CH, encode, episode_starts  # noqa: E402
from grid_utils import obs_to_ij  # noqa: E402  (path via common)
from signal1_common import get_model  # noqa: E402
from pldm_envs.diverse_maze.adaptive_waypoints.preprocess import normalize_images, normalize_proprio_vel  # noqa: E402

PAIRS_PER_TYPE, PCA_FRAMES_PER_MAP, PCA_K = 300, 400, (4, 10, 50)
SPACES = ["obs_full", "obs_pca4", "obs_pca10", "obs_pca50", "fused_full"]
SPIKE = 1.0


def bfs_all(layout):
    cells = [(i, j) for i in range(len(layout)) for j in range(len(layout[0])) if layout[i][j] == "O"]
    dist = {}
    for c in cells:
        d = {c: 0}
        q = deque([c])
        while q:
            i, j = q.popleft()
            for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (i + a, j + b)
                if n in d or not (0 <= n[0] < len(layout) and 0 <= n[1] < len(layout[0])) or layout[n[0]][n[1]] != "O":
                    continue
                d[n] = d[(i, j)] + 1
                q.append(n)
        dist[c] = d
    return cells, dist


def encode_map(model, data, images, starts, eps, frames=None):
    obs_l, fus_l, meta = [], [], []
    for e in eps:
        ep = data[e]
        idx = np.arange(len(ep["observations"])) if frames is None else frames[e]
        img = normalize_images(torch.from_numpy(np.array(images[starts[e] + idx])).float().permute(0, 3, 1, 2))
        pv = normalize_proprio_vel(torch.from_numpy(ep["observations"][idx, 2:4]).float())
        enc, _ = encode(model, img, pv)
        obs_l.append(enc[:, :D_OBS_CH].flatten(1))
        fus_l.append(enc.flatten(1))
        for t in idx:
            meta.append((e, int(t), *ep["observations"][t]))
    return torch.cat(obs_l), torch.cat(fus_l), np.array(meta)


def fit_pca(model, data, images, starts, by_map, rng):
    rows = []
    for m, eps in by_map.items():
        frames = {e: np.sort(rng.choice(len(data[e]["observations"]), size=PCA_FRAMES_PER_MAP // len(eps) + 1, replace=False)) for e in eps}
        obs, _, _ = encode_map(model, data, images, starts, eps, frames)
        rows.append(obs)
    X = torch.cat(rows)
    mean = X.mean(0)
    U, S, V = torch.svd_lowrank(X - mean, q=max(PCA_K) + 10, niter=6)
    return mean, V[:, :max(PCA_K)], float((S[:max(PCA_K)] ** 2).sum() / ((X - mean) ** 2).sum())


def sample_pairs(cells_of, meta, layout, dist, rng):
    """Returns dict type -> list of (frame_idx1, frame_idx2)."""
    by_cell = {}
    for k, c in enumerate(cells_of):
        by_cell.setdefault(c, []).append(k)
    ep, t = meta[:, 0].astype(int), meta[:, 1].astype(int)
    out = {"a_same_cell": [], "b_open_neighbour": [], "c_across_wall": [], "d_far": []}
    same = [c for c, ks in by_cell.items() if len(ks) >= 2]
    tries = 0
    while len(out["a_same_cell"]) < PAIRS_PER_TYPE and tries < 50000 and same:
        tries += 1
        ks = by_cell[same[rng.integers(len(same))]]
        a, b = rng.choice(ks, 2, replace=False)
        if ep[a] != ep[b] or abs(t[a] - t[b]) >= 20:
            out["a_same_cell"].append((a, b))
    occ = [c for c in by_cell]
    nb_pairs, wall_pairs, far_pairs = [], [], []
    for c in occ:
        for d in occ:
            if c >= d:
                continue
            md = dist[c].get(d)
            if md is None:
                continue
            di, dj = d[0] - c[0], d[1] - c[1]
            if abs(di) + abs(dj) == 1:
                nb_pairs.append((c, d))
            elif (abs(di), abs(dj)) in ((2, 0), (0, 2)) and layout[(c[0] + d[0]) // 2][(c[1] + d[1]) // 2] == "#" and md >= 4:
                wall_pairs.append((c, d))
            if md >= 5:
                far_pairs.append((c, d))
    for key, cand in (("b_open_neighbour", nb_pairs), ("c_across_wall", wall_pairs), ("d_far", far_pairs)):
        for _ in range(PAIRS_PER_TYPE if cand else 0):
            c, d = cand[rng.integers(len(cand))]
            out[key].append((rng.choice(by_cell[c]), rng.choice(by_cell[d])))
    return out


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    rng = np.random.default_rng(0)
    model = get_model()
    data = torch.load(os.path.join(DATA, "main", "data.p"), weights_only=False)
    maps = torch.load(os.path.join(DATA, "main", "train_maps.pt"), weights_only=False)
    images = np.load(os.path.join(DATA, "main", "images.npy"), mmap_mode="r")
    starts = episode_starts(data)
    by_map = {}
    for e, ep in enumerate(data):
        by_map.setdefault(int(ep["map_idx"]), []).append(e)
    pca_mean, pca_V, pca_var = fit_pca(model, data, images, starts, by_map, rng)
    print(f"PCA fitted (top-50 explains {pca_var:.3f} of obs variance)", flush=True)
    rows, spike_stats = [], []
    for m, eps in sorted(by_map.items()):
        layout = maps[m].split("\\")
        cells, dist = bfs_all(layout)
        obs, fus, meta = encode_map(model, data, images, starts, eps)
        ij = obs_to_ij(meta[:, 2:4].astype(np.float64))
        cells_of = [tuple(int(v) for v in c) for c in ij]
        # post-hoc (stage 2b): obs-latent "spike" frames, far (>1.0) from both temporal neighbours
        same_ep_next = np.r_[meta[1:, 0] == meta[:-1, 0], False]
        step = torch.cat([(obs[1:] - obs[:-1]).pow(2).mean(1), torch.tensor([0.0])]).numpy()
        d_next = np.where(same_ep_next, step, 0.0)
        d_prev = np.r_[0.0, d_next[:-1]]
        spike = (d_prev > SPIKE) & (d_next > SPIKE)
        spike_online = d_prev > SPIKE
        spike_stats.append((int(spike.sum()), len(spike)))
        proj = (obs - pca_mean) @ pca_V
        pairs = sample_pairs(cells_of, meta, layout, dist, rng)
        for typ, plist in pairs.items():
            for a, b in plist:
                d_obs = float((obs[a] - obs[b]).pow(2).mean())
                d_fus = float((fus[a] - fus[b]).pow(2).mean())
                d_p = {k: float((proj[a, :k] - proj[b, :k]).pow(2).mean()) for k in PCA_K}
                rows.append({"map": m, "type": typ, "obs_full": d_obs, "fused_full": d_fus,
                             **{f"obs_pca{k}": v for k, v in d_p.items()},
                             "dv": float(np.linalg.norm(meta[a, 4:6] - meta[b, 4:6])),
                             "xy_a": meta[a, 2:4].tolist(), "xy_b": meta[b, 2:4].tolist(),
                             "cell_a": cells_of[a], "cell_b": cells_of[b],
                             "maze_dist": dist[cells_of[a]].get(cells_of[b], -1),
                             "ep_t": [int(meta[a, 0]), int(meta[a, 1]), int(meta[b, 0]), int(meta[b, 1])],
                             "spike": [bool(spike[a]), bool(spike[b])], "spike_online": [bool(spike_online[a]), bool(spike_online[b])]})
        print(f"map {m}: {[len(v) for v in pairs.values()]} pairs", flush=True)
        del obs, fus, proj
    json.dump({"pca_top50_explained": pca_var, "spike_frames": [sum(a for a, _ in spike_stats), sum(b for _, b in spike_stats)], "rows": rows}, open(os.path.join(HERE, "stage2_pairs.json"), "w"))
    print("wrote stage2_pairs.json", len(rows), flush=True)


if __name__ == "__main__":
    main()
