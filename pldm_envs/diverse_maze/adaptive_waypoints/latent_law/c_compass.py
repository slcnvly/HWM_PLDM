# LATENT_LAW C (LATENT_LAW_PREREG.md): grid "compass" agents without MPPI.
# V0's 20 mazes / 120 instances; each open cell rendered at 5 positions (evaluator
# renderer, zero velocity); trained-encoder obs latents; goal = V0 goal render.
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
AW = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(AW, "online_map"))
sys.path.insert(0, os.path.join(AW, "inference_study"))
from f2_task1_spikes_in_eval import eval_path_encode, get_model  # noqa: E402
from f2_task3_latent_vs_maze_by_range import SWEEP  # noqa: E402
import stage1_failures as s1  # noqa: E402
import stage2_place_recognition as s2  # noqa: E402
import analyze_inference as ai  # noqa: E402
from grid_utils import GRID_SIZE, OBS_MIN_TOTAL, obs_to_ij  # noqa: E402

EXP = os.path.abspath(os.path.join(AW, "..", "..", "..", "..", "experiments"))
CELL_IMAGES = os.path.join(EXP, "kaggle_render_cells", "output", "cell_images.npz")
CELL_INPUTS = os.path.join(EXP, "hwm_cell_render_inputs", "cell_render_inputs.json")
MAX_MOVES, TAU, N_VIEWS = 54, 0.526, 5
NB = ((1, 0), (-1, 0), (0, 1), (0, -1))


def wilson(k, n, z=1.959964):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


class Memory:
    """Place identity: 'cell' (true cell id) or 'latent' (nearest stored view within TAU)."""

    def __init__(self, mode, D):
        self.mode, self.D = mode, D  # D: (n_views_total, n_views_total) pairwise MSE between all rendered views
        self.reps = []

    def ident(self, cell, vid):
        if self.mode == "cell":
            return cell
        if self.reps:
            d = self.D[vid, self.reps]
            j = int(np.argmin(d))
            if d[j] <= TAU:
                return ("node", j)
        self.reps.append(vid)
        return ("node", len(self.reps) - 1)


def run_agent(kind, heur, mem_mode, inst, rng):
    """kind in C1, C2, C3; heur in lat, xy. Returns dict with trajectory of cells."""
    open_cells, start, goal = inst["open"], inst["start"], inst["goal"]
    vid_of = inst["vid_of"]  # cell -> list of global view ids
    h_lat, h_xy = inst["h_lat"], inst["h_xy"]
    mem = Memory(mem_mode, inst["D"])
    counts, htab = {}, {}

    def view(c):
        return vid_of[c][rng.integers(N_VIEWS)]

    def h0(c, v):
        return h_lat[v] if heur == "lat" else h_xy[c]

    cur, path = start, [start]
    v = view(cur)
    cid = mem.ident(cur, v)
    counts[cid] = counts.get(cid, 0) + 1
    htab.setdefault(cid, h0(cur, v))
    for _ in range(MAX_MOVES):
        if cur == goal:
            break
        nbrs = [(cur[0] + a, cur[1] + b) for a, b in NB if (cur[0] + a, cur[1] + b) in open_cells]
        nv = [view(n) for n in nbrs]
        if kind == "C1":
            k = int(np.argmin([h0(n, x) for n, x in zip(nbrs, nv)]))
        elif kind == "C2":
            ids = [mem.ident(n, x) for n, x in zip(nbrs, nv)]
            key = [(counts.get(i, 0), h0(n, x)) for i, n, x in zip(ids, nbrs, nv)]
            k = min(range(len(nbrs)), key=lambda q: key[q])
        else:  # C3 LRTA* (max form, prereg)
            ids = [mem.ident(n, x) for n, x in zip(nbrs, nv)]
            hs = [htab.setdefault(i, h0(n, x)) for i, n, x in zip(ids, nbrs, nv)]
            k = int(np.argmin(hs))
            htab[cid] = max(htab[cid], 1 + hs[k])
        cur = nbrs[k]
        path.append(cur)
        v = view(cur)
        cid = mem.ident(cur, v)
        counts[cid] = counts.get(cid, 0) + 1
        htab.setdefault(cid, h0(cur, v))
    return path


def path_stats(path, inst):
    goal, dist, branches = inst["goal"], inst["dist"], inst["branches"]
    success = path[-1] == goal
    entries = [sum(1 for t in range(1, len(path)) if path[t] in b and path[t - 1] not in b) for b in branches]
    back = sum(1 for t in range(1, len(path)) if dist[goal][path[t]] > dist[goal][path[t - 1]])
    return {"success": success, "moves": len(path) - 1, "bfs": dist[goal][path[0]],
            "max_entries_one_branch": max(entries) if entries else 0, "total_branch_entries": sum(entries),
            "backward_move_share": back / max(len(path) - 1, 1)}


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    model = get_model()
    cin = json.load(open(CELL_INPUTS))
    cim = np.load(CELL_IMAGES)
    g = np.load(SWEEP)
    zg_all = eval_path_encode(model, g["goals"])
    goal_of = {int(t): k for k, t in enumerate(g["goal_trials"])}
    per_map = {}
    for mp in cin["maps"]:
        imgs = cim[f"map{mp['map_id']}"]  # (n_cells, 5, 3, 98, 98)
        z = eval_path_encode(model, imgs.reshape(-1, *imgs.shape[2:]))
        sq = (z.double() ** 2).sum(1)
        D = ((sq[:, None] + sq[None, :] - 2 * z.double() @ z.double().T) / z.shape[1]).clamp_min(0).numpy()
        cells = [tuple(c) for c in mp["cells"]]
        per_map[mp["map_key"]] = {"z": z, "D": D, "cells": cells, "vid_of": {c: list(range(k * N_VIEWS, (k + 1) * N_VIEWS)) for k, c in enumerate(cells)},
                                  "layout": mp["map_key"].split("\\")}
        print("encoded map", mp["map_id"], flush=True)
    trials = ai.load_variant("v0")
    configs = [("C1", "lat", "cell"), ("C1", "xy", "cell"),
               ("C2", "lat", "cell"), ("C2", "lat", "latent"), ("C2", "xy", "cell"), ("C2", "xy", "latent"),
               ("C3", "lat", "cell"), ("C3", "lat", "latent"), ("C3", "xy", "cell"), ("C3", "xy", "latent")]
    rows = {f"{a}_{h}_{m}": [] for a, h, m in configs}
    rows["C4_bfs"] = []
    v0_rows = []
    for tid, t in sorted(trials.items()):
        d = t["diag"]
        pm = per_map[d["map_key"]]
        layout = pm["layout"]
        start = tuple(int(v) for v in obs_to_ij(np.asarray(d["start_xy"])))
        goal = tuple(int(v) for v in obs_to_ij(np.asarray(d["target_xy"])))
        _, dist = s2.bfs_all(layout)
        h_lat = (pm["z"] - zg_all[goal_of[tid]]).pow(2).mean(1).numpy()
        h_xy = {c: float(np.linalg.norm(np.array([OBS_MIN_TOTAL + (c[0] + .5) * GRID_SIZE, OBS_MIN_TOTAL + (c[1] + .5) * GRID_SIZE]) - np.asarray(d["target_xy"])) / GRID_SIZE)
                for c in pm["cells"]}
        inst = {"open": set(pm["cells"]), "start": start, "goal": goal, "vid_of": pm["vid_of"], "h_lat": h_lat, "h_xy": h_xy,
                "D": pm["D"], "dist": dist, "branches": [b for b in s1.dead_end_branches(layout) if goal not in b]}
        for ci, (a, h, m) in enumerate(configs):
            rng = np.random.default_rng(tid * 1000 + ci)
            rows[f"{a}_{h}_{m}"].append(path_stats(run_agent(a, h, m, inst, rng), inst) | {"trial": tid})
        rows["C4_bfs"].append({"trial": tid, "success": dist[goal][start] <= MAX_MOVES, "moves": dist[goal][start], "bfs": dist[goal][start],
                               "max_entries_one_branch": 0, "total_branch_entries": 0, "backward_move_share": 0.0})
        v0_rows.append(int(t["success"]))
    summ = {}
    for name, rs in rows.items():
        k = sum(r["success"] for r in rs)
        succ = [r for r in rs if r["success"]]
        fail = [r for r in rs if not r["success"]]
        summ[name] = {
            "success": k, "n": len(rs), "rate": k / len(rs), "wilson95": wilson(k, len(rs)),
            "moves_over_bfs_success_median": float(np.median([r["moves"] / r["bfs"] for r in succ])) if succ else None,
            "moves_over_bfs_success_mean": float(np.mean([r["moves"] / r["bfs"] for r in succ])) if succ else None,
            "mean_max_entries_one_branch": float(np.mean([r["max_entries_one_branch"] for r in rs])),
            "failures_with_same_branch_reentry": float(np.mean([r["max_entries_one_branch"] >= 2 for r in fail])) if fail else None,
            "backward_move_share_failures": float(np.mean([r["backward_move_share"] for r in fail])) if fail else None,
            "backward_move_share_successes": float(np.mean([r["backward_move_share"] for r in succ])) if succ else None,
            "agreement_with_v0_success": float(np.mean([int(r["success"]) == v for r, v in zip(rs, v0_rows)])),
        }
    res = {"max_moves": MAX_MOVES, "tau": TAU, "summary": summ, "per_trial": rows}
    json.dump(res, open(os.path.join(HERE, "results_c_compass.json"), "w"), indent=2, default=lambda o: list(o) if isinstance(o, tuple) else str(o))
    for n, s in summ.items():
        print(f"{n:22s} {s['success']:3d}/120 [{s['wilson95'][0]:.2f},{s['wilson95'][1]:.2f}] moves/bfs med {s['moves_over_bfs_success_median']} | reentry-fail {s['failures_with_same_branch_reentry']} | back-fail {s['backward_move_share_failures']}")


if __name__ == "__main__":
    main()
