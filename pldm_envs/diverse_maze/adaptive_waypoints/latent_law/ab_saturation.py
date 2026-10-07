# LATENT_LAW A + B (LATENT_LAW_PREREG.md): saturation radius of representation
# distance vs maze/xy distance, for the trained PLDM encoder and controls.
# r50 main, 400 frames per map (seed 0), all within-map pairs. CPU, one map at a time.
import json
import os
import sys
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
AW = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(AW, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(AW, "boundary_study"))
sys.path.insert(0, os.path.join(AW, "inference_study"))
from omegaconf import OmegaConf  # noqa: E402
from pldm.configs import DataclassArgParser  # noqa: E402
from pldm.models.hjepa import HJEPA, HJEPAConfig  # noqa: E402
from pldm_envs.diverse_maze.adaptive_waypoints.compute_changepoints import _strip_enum_prefixes  # noqa: E402
from pldm_envs.diverse_maze.adaptive_waypoints.preprocess import normalize_images  # noqa: E402
from grid_utils import GRID_SIZE, obs_to_ij  # noqa: E402
from train_location_prober import load_model  # noqa: E402

DATA = os.path.join(AW, "..", "datasets", "r50_local", "r50_dataset", "main")
CONFIG = os.path.join(ROOT, "pldm/configs/diverse_maze/icml/large_diverse_25maps_l2.yaml")
FRAMES_PER_MAP, SPIKE, D_MAX = 400, 1.0, 15
XY_BINS = np.round(np.arange(0, 4.0001, 0.1), 2)
ENCODERS = ["B1_trained", "B2_random_seed0", "B2_random_seed1", "B2_random_seed2", "B3_dinov2_cls", "B3_dinov2_patch8x8", "B4_pixels"]


def random_backbone(seed):
    full = OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=False)
    _strip_enum_prefixes(full)
    torch.manual_seed(seed)
    m = HJEPA(DataclassArgParser._populate_dataclass_from_dict(HJEPAConfig, dict(full["hjepa"])), input_dim=(3, 98, 98), ppos_dim=0, pvel_dim=2, loc_dim=2)
    return m.eval().level1.backbone


def bfs_all(layout):
    out = {}
    cells = [(i, j) for i in range(len(layout)) for j in range(len(layout[0])) if layout[i][j] == "O"]
    for c in cells:
        d = {c: 0}
        q = deque([c])
        while q:
            i, j = q.popleft()
            for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (i + a, j + b)
                if n not in d and 0 <= n[0] < len(layout) and 0 <= n[1] < len(layout[0]) and layout[n[0]][n[1]] == "O":
                    d[n] = d[(i, j)] + 1
                    q.append(n)
        out[c] = d
    return out


def a1(data, maps):
    xy, bfs = [], []
    for ep in data:
        L = maps[int(ep["map_idx"])].split("\\")
        dist = bfs_all(L)
        o = ep["observations"][:, :2].astype(np.float64)
        ij = [tuple(int(v) for v in c) for c in obs_to_ij(o)]
        for t in range(len(o) - 14):
            xy.append(np.linalg.norm(o[t + 14] - o[t]) / GRID_SIZE)
            bfs.append(dist[ij[t]].get(ij[t + 14], np.nan))
    xy, bfs = np.array(xy), np.array(bfs)
    return {"xy_cells": {"median": float(np.median(xy)), "p90": float(np.percentile(xy, 90)), "mean": float(xy.mean())},
            "bfs_steps": {"median": float(np.nanmedian(bfs)), "p90": float(np.nanpercentile(bfs, 90)), "mean": float(np.nanmean(bfs)),
                          "share_0": float(np.nanmean(bfs == 0)), "share_le1": float(np.nanmean(bfs <= 1)), "share_le2": float(np.nanmean(bfs <= 2))}}


@torch.no_grad()
def features(name, raw_uint8_hwc, enc):
    """raw: (N,98,98,3) uint8 -> (N, D) float32."""
    x = torch.from_numpy(raw_uint8_hwc).float().permute(0, 3, 1, 2)
    if name.startswith("B1") or name.startswith("B2"):
        z = enc(normalize_images(x), proprio=torch.zeros(len(x), 2)).obs_component
        return z.flatten(1)
    if name == "B4_pixels":
        return normalize_images(x).flatten(1)
    y = F.interpolate(x / 255.0, size=(224, 224), mode="bilinear", align_corners=False)
    y = (y - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    outs = []
    for s in range(0, len(y), 64):
        o = enc.forward_features(y[s:s + 64])
        if name == "B3_dinov2_cls":
            outs.append(o["x_norm_clstoken"])
        else:
            p = o["x_norm_patchtokens"].reshape(-1, 16, 16, 384).permute(0, 3, 1, 2)
            outs.append(F.avg_pool2d(p, 2).flatten(1))
    return torch.cat(outs)


def pair_mse(Fm):
    sq = (Fm.double() ** 2).sum(1)
    d = (sq[:, None] + sq[None, :] - 2 * Fm.double() @ Fm.double().T) / Fm.shape[1]
    return d.clamp_min(0).float()


def curves(dist, d_maze, d_xy, keep):
    out = {"maze": {}, "xy": {}}
    for d in range(0, D_MAX + 1):
        v = dist[keep & (d_maze == d)]
        if len(v) >= 20:
            out["maze"][d] = [float(np.percentile(v, 25)), float(np.median(v)), float(np.percentile(v, 75)), int(len(v))]
    for k, lo in enumerate(XY_BINS[:-1]):
        v = dist[keep & (d_xy >= lo) & (d_xy < XY_BINS[k + 1])]
        if len(v) >= 20:
            out["xy"][round(lo + 0.05, 2)] = [float(np.percentile(v, 25)), float(np.median(v)), float(np.percentile(v, 75)), int(len(v))]
    v = dist[keep & (d_xy >= 3.0)]
    out["xy_plateau_median_ge3"] = float(np.median(v)) if len(v) else None
    return out


def radius(c):
    res = {}
    m = {int(k): v[1] for k, v in c["maze"].items()}
    if 0 in m and all(d in m for d in range(11, 16)):
        B, P = m[0], float(np.mean([m[d] for d in range(11, 16)]))
        res["maze_base"], res["maze_plateau"] = B, P
        if P - B < 0.1 * B:
            res["R_maze"] = "no_rise"
        else:
            res["R_maze"] = next((d for d in range(1, D_MAX + 1) if d in m and m[d] >= B + 0.9 * (P - B)), None)
            res["R_maze_slope"] = next((d for d in range(0, D_MAX) if d in m and d + 1 in m and m[d + 1] - m[d] < 0.05 * (P - B)), None)
    xs = sorted(float(k) for k in c["xy"])
    if xs and c["xy_plateau_median_ge3"] is not None:
        B, P = c["xy"][xs[0]][1], c["xy_plateau_median_ge3"]
        res["xy_base"], res["xy_plateau"] = B, P
        if P - B < 0.1 * B:
            res["R_xy"] = "no_rise"
        else:
            res["R_xy"] = next((x for x in xs if c["xy"][x][1] >= B + 0.9 * (P - B)), None)
    return res


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 10)))
    data = torch.load(os.path.join(DATA, "data.p"), weights_only=False)
    maps = torch.load(os.path.join(DATA, "train_maps.pt"), weights_only=False)
    images = np.load(os.path.join(DATA, "images.npy"), mmap_mode="r")
    starts = np.concatenate([[0], np.cumsum([len(e["observations"]) for e in data])])
    res = {"A1_14step_displacement": a1(data, maps)}
    print("A1", res["A1_14step_displacement"], flush=True)
    encs = {"B1_trained": load_model().level1.backbone}
    for s in range(3):
        encs[f"B2_random_seed{s}"] = random_backbone(s)
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True).eval()
    encs["B3_dinov2_cls"] = dino
    encs["B3_dinov2_patch8x8"] = dino
    encs["B4_pixels"] = None
    rng = np.random.default_rng(0)
    by_map = {}
    for e, ep in enumerate(data):
        by_map.setdefault(int(ep["map_idx"]), []).append(e)
    pool = {n: [] for n in ENCODERS}
    pool_maze, pool_xy, pool_keep0, pool_spike = [], [], [], []
    for m, eps in sorted(by_map.items()):
        L = maps[m].split("\\")
        dist = bfs_all(L)
        cand = [(e, t) for e in eps for t in range(len(data[e]["observations"]))]
        pick = [cand[k] for k in np.sort(rng.choice(len(cand), FRAMES_PER_MAP, replace=False))]
        gidx = np.array([starts[e] + t for e, t in pick])
        raw = np.array(images[gidx])
        xy = np.array([data[e]["observations"][t, :2] for e, t in pick], dtype=np.float64)
        ij = [tuple(int(v) for v in c) for c in obs_to_ij(xy)]
        ep_id = np.array([e for e, _ in pick]); t_id = np.array([t for _, t in pick])
        # B1 spike flags from temporal neighbours (t-1, t+1 within the episode)
        nb = []
        for e, t in pick:
            T = len(data[e]["observations"])
            nb += [starts[e] + max(t - 1, 0), starts[e] + min(t + 1, T - 1)]
        zc = features("B1_trained", raw, encs["B1_trained"])
        zn = features("B1_trained", np.array(images[np.array(nb)]), encs["B1_trained"]).view(len(pick), 2, -1)
        dprev = (zc - zn[:, 0]).pow(2).mean(1).numpy()
        dnext = (zc - zn[:, 1]).pow(2).mean(1).numpy()
        has_prev = t_id > 0
        has_next = np.array([t < len(data[e]["observations"]) - 1 for e, t in pick])
        spike = (dprev > SPIKE) & (dnext > SPIKE) & has_prev & has_next
        iu = np.triu_indices(len(pick), 1)
        d_maze = np.array([dist[ij[a]].get(ij[b], -1) for a, b in zip(*iu)])
        d_xy = np.linalg.norm(xy[iu[0]] - xy[iu[1]], axis=1) / GRID_SIZE
        keep0 = ~((d_maze == 0) & (ep_id[iu[0]] == ep_id[iu[1]]) & (np.abs(t_id[iu[0]] - t_id[iu[1]]) < 20))
        sp = spike[iu[0]] | spike[iu[1]]
        for n in ENCODERS:
            Fm = zc if n == "B1_trained" else features(n, raw, encs[n])
            pool[n].append(pair_mse(Fm)[iu].numpy().astype(np.float32))
        pool_maze.append(d_maze); pool_xy.append(d_xy); pool_keep0.append(keep0 & (d_maze >= 0)); pool_spike.append(sp)
        print(f"map {m}: spikes {spike.mean():.3f}", flush=True)
    d_maze, d_xy = np.concatenate(pool_maze), np.concatenate(pool_xy)
    keep, sp = np.concatenate(pool_keep0), np.concatenate(pool_spike)
    res["n_pairs"] = int(keep.sum()); res["share_pairs_with_spike_frame"] = float(sp[keep].mean())
    res["encoders"] = {}
    for n in ENCODERS:
        dist = np.concatenate(pool[n])
        res["encoders"][n] = {}
        for subset, k in (("all_pairs", keep), ("no_B1_spike_pairs", keep & ~sp)):
            c = curves(dist, d_maze, d_xy, k)
            res["encoders"][n][subset] = {"curves": c, "radius": radius(c)}
        r = res["encoders"][n]
        print(n, "R all", r["all_pairs"]["radius"], "| R no-spike", r["no_B1_spike_pairs"]["radius"], flush=True)
    json.dump(res, open(os.path.join(HERE, "results_ab_saturation.json"), "w"), indent=2)
    # figures: normalised curves (m - B) / (P - B)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for n in ENCODERS:
        for ax, key, base_key in ((axes[0], "maze", "maze"), (axes[1], "xy", "xy")):
            c = res["encoders"][n]["no_B1_spike_pairs"]["curves"][key]
            rr = res["encoders"][n]["no_B1_spike_pairs"]["radius"]
            B, P = rr.get(f"{base_key}_base"), rr.get(f"{base_key}_plateau")
            if B is None or P is None or P == B:
                continue
            xs = sorted(float(k) for k in c)
            ax.plot(xs, [(c[k if key == "xy" else int(k)][1] - B) / (P - B) for k in (xs if key == "xy" else [int(x) for x in xs])],
                    "-o" if n == "B1_trained" else "-", ms=3, lw=2.2 if n == "B1_trained" else 1, label=n)
    a = res["A1_14step_displacement"]
    axes[1].axvspan(0.5 * a["xy_cells"]["median"], 2 * a["xy_cells"]["median"], color="0.9", label="A3 band [0.5x, 2x] 14-step median")
    axes[1].axvline(a["xy_cells"]["median"], c="k", ls=":", lw=0.8, label="14-step displacement median")
    axes[0].set_xlabel("maze distance (BFS cells)"); axes[1].set_xlabel("xy distance (cells)")
    for ax in axes:
        ax.axhline(0.9, c="r", ls=":", lw=0.7); ax.set_ylabel("(median - base) / (plateau - base)"); ax.legend(fontsize=6)
    fig.suptitle("Normalised representation distance (pairs without B1 spike frames)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "ab_saturation_curves.png"), dpi=110)


if __name__ == "__main__":
    main()
