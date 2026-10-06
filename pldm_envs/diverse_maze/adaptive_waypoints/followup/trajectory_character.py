# What kind of behaviour is in r50? Position-only statistics computed identically for
#   r50 main (the training data), random-action simulations in the same mazes from the
#   same starts (kaggle_sim_random_r50: a1 = data's initial velocity, a2 = zero), and
#   evaluation trajectories (V0 successes up to their goal step; V0 failures for reference).
# One step = one recorded step = 4 simulator steps in every source.
# Velocity = per-step displacement (positions only), so all sources are treated alike.
#   straightness  : per 10-step chunk, |x_{s+10}-x_s| / sum |x_{t+1}-x_t|
#   revisit       : share of steps entering a cell already visited in the previous 20 steps
#                   (cell changed at t, and cell(t) in cells(t-20..t-2))
#   reversal      : share of steps with cos(d_t, d_{t-1}) < 0 (both displacements > 0.01 cells)
#   wall contact  : near a wall-adjacent edge and the toward-wall displacement dropped
#                   (strict / medium / loose thresholds, see THRESH)
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "boundary_study"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "inference_study"))
from grid_utils import GRID_SIZE, OBS_MIN_TOTAL, obs_to_ij  # noqa: E402

AW = os.path.dirname(HERE)
DATA = os.path.join(AW, "..", "datasets", "r50_local", "r50_dataset", "main")
SIM = os.path.join(AW, "..", "..", "..", "..", "experiments", "kaggle_sim_random_r50", "output", "sim_random_r50.npz")
CHUNK, REVISIT_WIN, MIN_DISP = 10, 20, 0.01
THRESH = {"strict": (0.07, 0.80), "medium": (0.12, 0.50), "loose": (0.20, 0.25)}  # (edge dist in cells, toward-wall drop)


def wall_normals(layout, i, j, x, y):
    """[(distance to edge in cells, unit normal pointing INTO the wall)] for wall neighbours of cell (i, j)."""
    out = []
    rows, cols = len(layout), len(layout[0])
    x0, y0 = OBS_MIN_TOTAL + i * GRID_SIZE, OBS_MIN_TOTAL + j * GRID_SIZE
    fx, fy = (x - x0) / GRID_SIZE, (y - y0) / GRID_SIZE

    def wall(a, b):
        return not (0 <= a < rows and 0 <= b < cols) or layout[a][b] == "#"

    if wall(i - 1, j):
        out.append((fx, np.array([-1.0, 0.0])))
    if wall(i + 1, j):
        out.append((1 - fx, np.array([1.0, 0.0])))
    if wall(i, j - 1):
        out.append((fy, np.array([0.0, -1.0])))
    if wall(i, j + 1):
        out.append((1 - fy, np.array([0.0, 1.0])))
    return out


def stats_for(trajs):
    """trajs: list of (xy (T,2) array, layout)."""
    straight, revisit_n, revisit_d, rev_n, rev_d, steps = [], 0, 0, 0, 0, 0
    wall_hits = {k: 0 for k in THRESH}
    for xy, layout in trajs:
        xy = np.asarray(xy, dtype=np.float64)
        T = len(xy)
        if T < 3:
            continue
        disp = np.diff(xy, axis=0) / GRID_SIZE  # cells per step
        dn = np.linalg.norm(disp, axis=1)
        for s in range(0, T - CHUNK, CHUNK):
            path = dn[s:s + CHUNK].sum()
            if path > 1e-6:
                straight.append(np.linalg.norm(xy[s + CHUNK] - xy[s]) / GRID_SIZE / path)
        ij = obs_to_ij(xy)
        cells = [tuple(c) for c in ij]
        for t in range(1, T):
            if cells[t] != cells[t - 1]:
                revisit_d += 1
                if cells[t] in set(cells[max(0, t - REVISIT_WIN):t - 1]):
                    revisit_n += 1
        for t in range(1, len(disp)):
            if dn[t] > MIN_DISP and dn[t - 1] > MIN_DISP:
                rev_d += 1
                rev_n += float(np.dot(disp[t], disp[t - 1])) < 0
        for t in range(1, len(disp)):
            x, y = xy[t + 1]
            normals = wall_normals(layout, int(ij[t + 1, 0]), int(ij[t + 1, 1]), x, y)
            for name, (dthr, drop) in THRESH.items():
                hit = False
                for d, n in normals:
                    before, after = float(np.dot(disp[t - 1], n)), float(np.dot(disp[t], n))
                    if d <= dthr and before > 0.005 and after <= (1 - drop) * before:
                        hit = True
                        break
                wall_hits[name] += hit
        steps += len(disp) - 1
    s = np.array(straight)
    hist, edges = np.histogram(s, bins=10, range=(0, 1))
    return {
        "n_trajectories": len(trajs), "n_steps": steps, "n_chunks": int(len(s)),
        "straightness": {"mean": float(s.mean()), "q10": float(np.percentile(s, 10)), "q25": float(np.percentile(s, 25)),
                         "median": float(np.median(s)), "q75": float(np.percentile(s, 75)), "q90": float(np.percentile(s, 90)),
                         "hist_bins_0_to_1_step_0.1": hist.tolist()},
        "revisit_within_20_share_of_steps": revisit_n / max(steps, 1),
        "revisit_within_20_share_of_cell_entries": revisit_n / max(revisit_d, 1),
        "cell_entries_per_step": revisit_d / max(steps, 1),
        "reversal_share_of_moving_steps": rev_n / max(rev_d, 1),
        "wall_contact_share_of_steps": {k: v / max(steps, 1) for k, v in wall_hits.items()},
        "mean_step_cells": float(np.mean([np.linalg.norm(np.diff(np.asarray(x), axis=0), axis=1).mean() / GRID_SIZE for x, _ in trajs])),
    }


def main():
    data = torch.load(os.path.join(DATA, "data.p"), weights_only=False)
    maps = torch.load(os.path.join(DATA, "train_maps.pt"), weights_only=False)
    layouts = [maps[int(e["map_idx"])].split("\\") for e in data]
    sources = {"r50_main": [(e["observations"][:, :2], L) for e, L in zip(data, layouts)]}
    if os.path.exists(SIM):
        sim = np.load(SIM)
        for k, name in (("a1", "random_sim_data_init_vel"), ("a2", "random_sim_zero_init_vel")):
            sources[name] = [(sim[k][n][:, :2], layouts[int(ep)]) for n, ep in enumerate(sim["ep"])]
    import analyze_inference as ai
    v0 = ai.load_variant("v0")
    succ, fail = [], []
    for t in v0.values():
        d = t["diag"]
        xy = np.array([d["start_xy"]] + [s[:2] for s in d["steps"]["s1"] + d["steps"]["s2"]])
        L = d["map_key"].split("\\")
        if t["success"]:
            succ.append((xy[:t["steps"] + 2], L))  # start .. the step where the goal was reached
        else:
            fail.append((xy, L))
    sources["eval_v0_success_until_goal"] = succ
    sources["eval_v0_failure_full"] = fail
    res = {"definitions": {"chunk": CHUNK, "revisit_window": REVISIT_WIN, "min_disp_cells_for_reversal": MIN_DISP,
                           "wall_thresholds_cells_drop": THRESH, "grid_size": GRID_SIZE},
           "sources": {k: stats_for(v) for k, v in sources.items()}}
    json.dump(res, open(os.path.join(HERE, "results_trajectory_character.json"), "w"), indent=2)
    for k, v in res["sources"].items():
        print(f"{k:32s} straight mean {v['straightness']['mean']:.3f} med {v['straightness']['median']:.3f} | revisit/step {v['revisit_within_20_share_of_steps']:.3f} "
              f"(/entry {v['revisit_within_20_share_of_cell_entries']:.3f}) | reversal {v['reversal_share_of_moving_steps']:.3f} | wall {v['wall_contact_share_of_steps']} | step {v['mean_step_cells']:.3f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, v in res["sources"].items():
        h = np.array(v["straightness"]["hist_bins_0_to_1_step_0.1"], float)
        ax.plot(np.arange(0.05, 1, 0.1), h / h.sum(), marker="o", label=k)
    ax.set_xlabel("straightness of 10-step chunk"); ax.set_ylabel("share of chunks"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "trajectory_straightness_hist.png"), dpi=120)


if __name__ == "__main__":
    main()
