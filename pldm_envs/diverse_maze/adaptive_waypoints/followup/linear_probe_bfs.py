# Follow-up 4 (2026-10-06): can maze (BFS) distance be read linearly from the
# current and goal latents? Corrected (normalized) inputs, frozen L1, obs_component
# (the planner's cost space). Frames: every 10th frame of every r50 episode,
# main (25 maps) + probe (20 maps). Pairs: random (current, goal) frame pairs on the
# same map. Map-level 5-fold split (GroupKFold over the 45 maps).
#   probe_concat : ridge on [P z_cur, P z_goal]            (linear in z)
#   probe_sqdiff : ridge on (P z_cur - P z_goal)^2          (learned weighted latent distance)
#   base_latent  : ridge on the planner's latent distance mean((z_cur - z_goal)^2) alone
#   base_xy      : ridge on Euclidean xy distance alone
# P = fixed Gaussian random projection 29584 -> 1024 (seed 0; data independent, so no leakage).
import json
import os
from collections import deque

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import DATA, D_OBS_CH, encode, episode_starts
from grid_utils import obs_to_ij
from signal1_common import get_model
from pldm_envs.diverse_maze.adaptive_waypoints.preprocess import normalize_images, normalize_proprio_vel

HERE = os.path.dirname(os.path.abspath(__file__))
EVERY, PAIRS_PER_MAP, PROJ = 10, 1000, 1024


def bfs(layout, goal):
    rows, cols = len(layout), len(layout[0])
    d = np.full((rows, cols), -1, int)
    d[goal] = 0
    q = deque([goal])
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < rows and 0 <= b < cols and layout[a][b] != "#" and d[a, b] < 0:
                d[a, b] = d[i, j] + 1
                q.append((a, b))
    return d


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
    model = get_model()
    gen = torch.Generator().manual_seed(0)
    P = torch.randn(D_OBS_CH * 43 * 43, PROJ, generator=gen) / np.sqrt(PROJ)
    Z, PZ, XY, MAPKEY = [], [], [], []
    for split in ("main", "probe"):
        data = torch.load(os.path.join(DATA, split, "data.p"), weights_only=False)
        maps = torch.load(os.path.join(DATA, split, "train_maps.pt"), weights_only=False)
        images = np.load(os.path.join(DATA, split, "images.npy"), mmap_mode="r")
        starts = episode_starts(data)
        for e, ep in enumerate(data):
            idx = np.arange(0, len(ep["observations"]), EVERY)
            img = normalize_images(torch.from_numpy(np.array(images[starts[e] + idx])).float().permute(0, 3, 1, 2))
            pv = normalize_proprio_vel(torch.from_numpy(ep["observations"][idx, 2:4]).float())
            enc, _ = encode(model, img, pv)
            obs = enc[:, :D_OBS_CH].flatten(1)
            Z.append(obs.half())
            PZ.append(obs @ P)
            XY.append(ep["observations"][idx, :2])
            MAPKEY += [maps[int(ep["map_idx"])]] * len(idx)
        print(f"encoded {split}", flush=True)
    Z, PZ, XY = torch.cat(Z), torch.cat(PZ).numpy(), np.concatenate(XY)
    MAPKEY = np.array(MAPKEY)
    keys = sorted(set(MAPKEY.tolist()))
    rng = np.random.default_rng(0)
    rows = {"cur": [], "goal": [], "bfs": [], "map": []}
    for m, key in enumerate(keys):
        layout = key.split("\\")
        fr = np.where(MAPKEY == key)[0]
        ij = obs_to_ij(XY[fr].astype(np.float64))
        fields = {}
        for _ in range(PAIRS_PER_MAP):
            a, b = rng.choice(len(fr), size=2, replace=False)
            g = (int(ij[b, 0]), int(ij[b, 1]))
            if g not in fields:
                fields[g] = bfs(layout, g)
            d = fields[g][int(ij[a, 0]), int(ij[a, 1])]
            if d < 0:
                continue
            rows["cur"].append(fr[a]); rows["goal"].append(fr[b]); rows["bfs"].append(d); rows["map"].append(m)
    cur, goal = np.array(rows["cur"]), np.array(rows["goal"])
    y, groups = np.array(rows["bfs"], float), np.array(rows["map"])
    lat = np.array([float((Z[c].float() - Z[g].float()).pow(2).mean()) for c, g in zip(cur, goal)])
    feats = {
        "probe_concat": np.concatenate([PZ[cur], PZ[goal]], axis=1),
        "probe_sqdiff": (PZ[cur] - PZ[goal]) ** 2,
        "base_latent_mse": lat[:, None],
        "base_xy_euclid": np.linalg.norm(XY[cur] - XY[goal], axis=1)[:, None],
    }
    res = {"n_pairs": len(y), "n_maps": len(keys), "bfs_range": [float(y.min()), float(y.max())], "bfs_mean": float(y.mean()),
           "spearman_raw_feature_vs_bfs": {"latent_mse": float(spearmanr(lat, y).correlation),
                                           "xy_euclid": float(spearmanr(feats["base_xy_euclid"][:, 0], y).correlation)},
           "models": {}}
    for name, X in feats.items():
        pred = np.zeros_like(y)
        fold = []
        for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
            mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 15)))
            mdl.fit(X[tr], y[tr])
            pred[va] = mdl.predict(X[va])
            fold.append({"spearman": float(spearmanr(pred[va], y[va]).correlation), "mae": float(np.abs(pred[va] - y[va]).mean())})
        res["models"][name] = {
            "val_spearman_mean_over_folds": float(np.mean([f["spearman"] for f in fold])),
            "val_mae_mean_over_folds": float(np.mean([f["mae"] for f in fold])),
            "val_spearman_pooled": float(spearmanr(pred, y).correlation),
            "folds": fold,
        }
        print(name, res["models"][name]["val_spearman_mean_over_folds"], res["models"][name]["val_mae_mean_over_folds"], flush=True)
    res["mae_predict_mean_baseline"] = float(np.abs(y - y.mean()).mean())
    json.dump(res, open(os.path.join(HERE, "results_linear_probe_bfs.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "models"}, indent=1))


if __name__ == "__main__":
    main()
