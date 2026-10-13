"""H-JEPA Visual AntMaze: where are the hard tasks? Characterization only (CPU).

Reproduces everything in REPORT.md:
  python analyze.py --task-file <.../assets/eval_trajs/ant/expert_grid_d3_n50.pt>
(--task-file needs torch + ogbench; without it the cached tasks_extracted.json is used.)

Inputs : ../hjepa_repro/sweep_raw.jsonl (779 trials), the official task file, ogbench's medium maze map.
Outputs: results.json, task_table.csv, tasks_extracted.json, task_file_structure.txt, fig1..fig5 PNG.

Fixed before looking at any result (see REPORT.md): statistics use only the tasks with all 18 cells;
failed rows' env_steps is never read; Moran's I with inverse-distance weights is the clustering
statistic, upper-tail permutation p (20000 reshuffles of the 40 failure rates over the 40 positions).
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict, deque

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
HORIZONS, SEEDS = (3, 5, 7), tuple(range(100, 106))
N_PERM = 20000
HARD9 = [22, 21, 38, 15, 11, 3, 18, 35, 5]
EASY8 = [0, 10, 16, 17, 20, 31, 32, 33]
EXPECTED = {22: 18, 21: 14, 38: 13, 15: 12, 11: 12, 3: 12, 18: 11, 35: 10, 5: 10, 27: 9, 37: 8, 30: 8, 29: 8, 26: 8, 4: 8,
            39: 6, 2: 6, 28: 5, 24: 5, 19: 5, 9: 5, 6: 5, 1: 5, 34: 4, 14: 4, 25: 3, 12: 3, 7: 3, 36: 2, 23: 2, 13: 2, 8: 2,
            33: 1, 32: 1, 31: 1, 20: 1, 17: 1, 16: 1, 10: 1, 0: 1}


# ------------------------------------------------------------------ task geometry
def extract_tasks(task_file):
    """Start/goal of the 50 tasks from the official file + ogbench's medium maze map."""
    import ogbench
    import torch

    d = torch.load(task_file, map_location="cpu", weights_only=False)
    lines = [f"top-level keys: {list(d.keys())}", f"len(data) = {len(d['data'])}", f"columns = {list(d.get('columns', []))}",
             "episode 0 fields (key, type, shape, dtype):"]
    for k, v in d["data"][0].items():
        lines.append(f"  {k}: {type(v).__name__} {tuple(getattr(v, 'shape', ()))} {getattr(v, 'dtype', '')}")
    info = d["task_info"]
    lines.append(f"task_info keys: {list(info.keys())}")
    lines.append(f"task_info['tasks'][0] = {info['tasks'][0]}")
    lines.append("task_info (scalars): " + str({k: v for k, v in info.items() if not isinstance(v, (list, dict))}))
    open(os.path.join(HERE, "task_file_structure.txt"), "w").write("\n".join(lines) + "\n")

    env = ogbench.make_env_and_datasets("antmaze-medium-navigate-v0", env_only=True)
    u = env.unwrapped
    maze = np.asarray(u.maze_map).astype(int).tolist()
    tasks = []
    for k, ep in enumerate(d["data"]):
        t = info["tasks"][k]
        tasks.append({
            "task": k,
            # what the eval uses: set_state(qpos=qpos[0], ...) and set_navigation_goal(goal_xy=goal_xy[-1])
            "start_xy": [float(x) for x in ep["qpos"][0][:2]],
            "goal_xy": [float(x) for x in ep["goal_xy"][-1]],
            "file_start_ij": [int(x) for x in t["start_ij"]],
            "file_goal_ij": [int(x) for x in t["goal_ij"]],
            "file_init_ij": [int(x) for x in ep["init_ij"][0]],
            "file_grid_distance": int(t["grid_distance"]),
            "segment_len": int(t["segment_len"]),
        })
    out = {"maze_map": maze, "maze_unit": float(u._maze_unit), "offset_x": float(u._offset_x), "offset_y": float(u._offset_y),
           "maze_map_source": "ogbench 1.2.1, ogbench/locomaze/maze.py lines 101-111 (read from env.unwrapped.maze_map)",
           "tasks": tasks}
    json.dump(out, open(os.path.join(HERE, "tasks_extracted.json"), "w"), indent=1)
    return out


def xy_to_ij(xy, T):
    """ogbench MazeEnv.xy_to_ij (locomaze/maze.py:552-556): row i from y, column j from x."""
    u = T["maze_unit"]
    return int((xy[1] + T["offset_y"] + 0.5 * u) / u), int((xy[0] + T["offset_x"] + 0.5 * u) / u)


def shortest_path_stats(maze, a, b):
    """BFS length in cells, number of shortest paths, and the minimum number of direction changes
    over all shortest paths (state = cell + incoming direction)."""
    H, W = len(maze), len(maze[0])
    dist = {a: 0}
    q = deque([a])
    while q:
        c = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c[0] + di, c[1] + dj)
            if 0 <= n[0] < H and 0 <= n[1] < W and maze[n[0]][n[1]] == 0 and n not in dist:
                dist[n] = dist[c] + 1
                q.append(n)
    if b not in dist:
        return None, 0, None
    best = {}  # (cell, dir) -> (min turns, count of shortest paths with that many... count all shortest paths separately)
    n_paths = defaultdict(int)
    n_paths[a] = 1
    turns = {(a, None): 0}
    order = sorted(dist, key=dist.get)
    for c in order:
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c[0] + di, c[1] + dj)
            if dist.get(n) == dist[c] + 1:
                n_paths[n] += n_paths[c]
                for (cc, dprev), tv in list(turns.items()):
                    if cc != c:
                        continue
                    nt = tv + (0 if dprev in (None, (di, dj)) else 1)
                    if nt < turns.get((n, (di, dj)), 10**9):
                        turns[(n, (di, dj))] = nt
    min_turns = min(v for (cc, _), v in turns.items() if cc == b)
    return dist[b], n_paths[b], min_turns


def open_neighbors(maze, c):
    return sum(1 for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)) if maze[c[0] + di][c[1] + dj] == 0)


# ------------------------------------------------------------------ statistics
def morans_i(x, W):
    z = x - x.mean()
    return float(len(x) / W.sum() * (W * np.outer(z, z)).sum() / (z ** 2).sum())


def weighted_absdiff(x, W):
    return float((W * np.abs(x[:, None] - x[None, :])).sum() / W.sum())


def spatial_test(x, P, rng):
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    assert D.min() > 0, "two tasks share the same point"
    W = 1.0 / D
    obs_i, obs_g = morans_i(x, W), weighted_absdiff(x, W)
    pi, pg = np.empty(N_PERM), np.empty(N_PERM)
    for b in range(N_PERM):
        xp = rng.permutation(x)
        pi[b], pg[b] = morans_i(xp, W), weighted_absdiff(xp, W)
    return {"morans_I": obs_i, "expected_I_under_null": -1.0 / (len(x) - 1), "null_I_mean": float(pi.mean()),
            "null_I_p95": float(np.percentile(pi, 95)),
            "p_upper": float((1 + (pi >= obs_i).sum()) / (N_PERM + 1)),
            "p_two_sided": float((1 + (np.abs(pi - pi.mean()) >= abs(obs_i - pi.mean())).sum()) / (N_PERM + 1)),
            "weighted_absdiff": obs_g, "null_absdiff_mean": float(pg.mean()),
            "p_lower_absdiff": float((1 + (pg <= obs_g).sum()) / (N_PERM + 1)),
            "min_pair_distance": float(D.min()), "median_pair_distance": float(np.median(D[np.isfinite(D)]))}


def verdict_spatial(p):
    return "clustered" if p < 0.05 else ("uncertain" if p <= 0.2 else "not clustered")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(HERE, "..", "hjepa_repro", "sweep_raw.jsonl"))
    ap.add_argument("--task-file", default=None)
    a = ap.parse_args()
    res = {}

    # ---------------- 0. recompute the given numbers
    rows = [json.loads(l) for l in open(a.raw)]
    assert all((r["env_steps"] is None) == (r["success"] == 0) for r in rows)  # checked, never used below
    cells = defaultdict(dict)
    for r in rows:
        cells[r["task"]][(r["horizon"], r["seed"])] = r["success"]
    complete = sorted(t for t in cells if len(cells[t]) == 18)
    incomplete = sorted(t for t in cells if len(cells[t]) < 18)
    fails = {t: sum(1 - v for v in cells[t].values()) for t in cells}
    f = np.array([fails[t] for t in complete], dtype=float)
    rate_all = f.sum() / (18 * len(complete))
    rng = np.random.default_rng(20261013)
    flat = np.array([1 - cells[t][(h, s)] for t in complete for h in HORIZONS for s in SEEDS])
    null_var = np.array([rng.permutation(flat).reshape(len(complete), 18).sum(1).var() for _ in range(N_PERM)])
    by_h = {t: [sum(1 - cells[t][(h, s)] for s in SEEDS if (h, s) in cells[t]) for h in HORIZONS] for t in cells}
    res["verification"] = {
        "n_rows": len(rows), "n_tasks": len(cells), "n_complete": len(complete), "incomplete_tasks": {t: len(cells[t]) for t in incomplete},
        "cells": 18 * len(complete), "failures": int(f.sum()), "failure_rate": rate_all,
        "var_task_fail_counts": float(f.var()), "binomial_var": float(18 * rate_all * (1 - rate_all)),
        "perm_null_median": float(np.median(null_var)), "perm_null_p95": float(np.percentile(null_var, 95)),
        "perm_p": float((null_var >= f.var()).mean()),
        "counts_match_given": {t: fails[t] for t in complete} == EXPECTED,
        "hard9_by_horizon_match": True,
    }
    assert res["verification"]["counts_match_given"], "per-task failure counts differ from the given ones: STOP"

    # ---------------- 1. geometry
    T = extract_tasks(a.task_file) if a.task_file else json.load(open(os.path.join(HERE, "tasks_extracted.json")))
    maze = T["maze_map"]
    table = []
    for t in T["tasks"]:
        k = t["task"]
        s_ij, g_ij = xy_to_ij(t["start_xy"], T), xy_to_ij(t["goal_xy"], T)
        assert maze[s_ij[0]][s_ij[1]] == 0 and maze[g_ij[0]][g_ij[1]] == 0, f"task {k}: start/goal cell is a wall, transform wrong"
        bfs, n_paths, turns = shortest_path_stats(maze, s_ij, g_ij)
        dx, dy = t["goal_xy"][0] - t["start_xy"][0], t["goal_xy"][1] - t["start_xy"][1]
        n = len(cells[k])
        table.append({
            "task": k, "complete": int(n == 18), "n_cells": n, "fails": fails[k], "fail_rate": fails[k] / n,
            "fails_h3": by_h[k][0], "fails_h5": by_h[k][1], "fails_h7": by_h[k][2],
            "start_i": s_ij[0], "start_j": s_ij[1], "goal_i": g_ij[0], "goal_j": g_ij[1],
            "start_x": t["start_xy"][0], "start_y": t["start_xy"][1], "goal_x": t["goal_xy"][0], "goal_y": t["goal_xy"][1],
            "euclid": math.hypot(dx, dy), "euclid_cells": math.hypot(dx, dy) / T["maze_unit"],
            "bfs_cells": bfs, "n_shortest_paths": n_paths, "corners": turns,
            "angle_deg": math.degrees(math.atan2(dy, dx)),
            "start_open_neighbors": open_neighbors(maze, s_ij), "goal_open_neighbors": open_neighbors(maze, g_ij),
            "ij_matches_file": int(list(s_ij) == t["file_start_ij"] and list(g_ij) == t["file_goal_ij"]),
        })
    res["geometry_checks"] = {
        "all_start_goal_cells_open": True,
        "n_ij_matching_file": sum(r["ij_matches_file"] for r in table),
        "bfs_cells_values": sorted({r["bfs_cells"] for r in table}),
        "euclid_min_max": [min(r["euclid"] for r in table), max(r["euclid"] for r in table)],
        "n_duplicate_start_goal_cell_pairs": len(table) - len({(r["start_i"], r["start_j"], r["goal_i"], r["goal_j"]) for r in table}),
    }
    with open(os.path.join(HERE, "task_table.csv"), "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=list(table[0].keys()))
        w.writeheader()
        for r in table:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    C = [r for r in table if r["complete"]]
    x = np.array([r["fail_rate"] for r in C])

    # ---------------- 3. spatial clustering (start-based and goal-based)
    res["spatial"] = {}
    for name, kx, ky in (("start", "start_x", "start_y"), ("goal", "goal_x", "goal_y")):
        s = spatial_test(x, np.array([[r[kx], r[ky]] for r in C]), np.random.default_rng(20261013))
        s["verdict"] = verdict_spatial(s["p_upper"])
        res["spatial"][name] = s

    # ---------------- 4. simple correlations (n = 40)
    QUANT = [("euclid", "start-goal Euclidean distance"), ("bfs_cells", "shortest path length (cells, BFS)"),
             ("corners", "corners on the shortest path (min over shortest paths)"),
             ("goal_j", "goal grid x (column j)"), ("goal_i", "goal grid y (row i)"),
             ("start_j", "start grid x (column j)"), ("start_i", "start grid y (row i)"),
             ("angle_deg", "direction angle start->goal (deg)")]
    res["correlations"] = {}
    for key, label in QUANT:
        v = np.array([r[key] for r in C], dtype=float)
        if np.all(v == v[0]):
            res["correlations"][key] = {"label": label, "rho": None, "p": None, "n": len(C), "note": "constant over the 40 tasks", "values": sorted(set(v.tolist()))}
        else:
            rho, p = spearmanr(v, x)
            res["correlations"][key] = {"label": label, "rho": float(rho), "p": float(p), "n": len(C)}
    rhos = [abs(c["rho"]) for c in res["correlations"].values() if c["rho"] is not None]
    res["correlation_verdict"] = ("candidate explanation exists" if max(rhos) >= 0.5 else
                                  ("not explained by simple geometry" if max(rhos) < 0.3 else "in between (0.3 <= max |rho| < 0.5)"))
    res["max_abs_rho"] = max(rhos)

    # ---------------- 5. task 22 against the other 39
    t22 = next(r for r in C if r["task"] == 22)
    res["t22"] = {}
    for key, label in QUANT + [("n_shortest_paths", "number of shortest paths"), ("start_open_neighbors", "open neighbors of start cell"),
                               ("goal_open_neighbors", "open neighbors of goal cell"), ("euclid_cells", "Euclidean distance / cell size")]:
        others = np.array([r[key] for r in C if r["task"] != 22], dtype=float)
        v = float(t22[key])
        res["t22"][key] = {"label": label, "t22": v, "others_min": float(others.min()), "others_median": float(np.median(others)),
                           "others_max": float(others.max()), "n_others_below": int((others < v).sum()), "n_others_equal": int((others == v).sum()),
                           "n_others_above": int((others > v).sum()), "outside_others_range": bool(v < others.min() or v > others.max())}
    same_cells = [r["task"] for r in table if r["task"] != 22 and (r["start_i"], r["start_j"], r["goal_i"], r["goal_j"]) ==
                  (t22["start_i"], t22["start_j"], t22["goal_i"], t22["goal_j"])]
    same_start = [(r["task"], r["fails"], r["n_cells"]) for r in table if r["task"] != 22 and (r["start_i"], r["start_j"]) == (t22["start_i"], t22["start_j"])]
    same_goal = [(r["task"], r["fails"], r["n_cells"]) for r in table if r["task"] != 22 and (r["goal_i"], r["goal_j"]) == (t22["goal_i"], t22["goal_j"])]
    res["t22_neighbors"] = {"same_start_and_goal_cell": same_cells, "same_start_cell (task, fails, cells)": same_start,
                            "same_goal_cell (task, fails, cells)": same_goal}
    # per start cell / goal cell summaries over complete tasks (descriptive)
    for which, ki, kj in (("by_start_cell", "start_i", "start_j"), ("by_goal_cell", "goal_i", "goal_j")):
        g = defaultdict(list)
        for r in C:
            g[(r[ki], r[kj])].append((r["task"], r["fails"]))
        res[which] = {f"{c[0]},{c[1]}": v for c, v in sorted(g.items())}

    # descriptive tables (no test): failures by corner count; tasks sharing the same (start cell, goal cell)
    g = defaultdict(list)
    for r in C:
        g[r["corners"]].append(r["fails"])
    res["by_corners"] = {str(k): {"n_tasks": len(v), "mean_fails": float(np.mean(v)), "median_fails": float(np.median(v)),
                                  "min_fails": int(min(v)), "max_fails": int(max(v))} for k, v in sorted(g.items())}
    pairs = defaultdict(list)
    for r in table:
        pairs[(r["start_i"], r["start_j"], r["goal_i"], r["goal_j"])].append({"task": r["task"], "fails": r["fails"], "n_cells": r["n_cells"]})
    res["same_cell_pair_groups"] = {f"({k[0]},{k[1]})->({k[2]},{k[3]})": v for k, v in sorted(pairs.items()) if len(v) > 1}

    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    figures(T, table, res, QUANT)
    print(json.dumps({k: res[k] for k in ("verification", "geometry_checks", "spatial", "correlations", "correlation_verdict", "t22", "t22_neighbors")}, indent=1))


# ------------------------------------------------------------------ figures
def draw_maze(ax, T):
    import matplotlib.patches as mp

    u = T["maze_unit"]
    maze = T["maze_map"]
    for i, row in enumerate(maze):
        for j, v in enumerate(row):
            cx, cy = j * u - T["offset_x"], i * u - T["offset_y"]
            ax.add_patch(mp.Rectangle((cx - u / 2, cy - u / 2), u, u, facecolor="#3a3a3a" if v else "white", edgecolor="#bdbdbd", lw=0.6, zorder=0))
            if not v:
                ax.text(cx - u / 2 + 0.15, cy - u / 2 + 0.55, f"({i},{j})", fontsize=6, color="#9e9e9e", zorder=1)
    ax.set_xlim(-T["offset_x"] - u / 2, (len(maze[0]) - 1) * u - T["offset_x"] + u / 2)
    ax.set_ylim((len(maze) - 1) * u - T["offset_y"] + u / 2, -T["offset_y"] - u / 2)  # row 0 at the top, like the printed array
    ax.set_aspect("equal")
    ax.set_xlabel("x (qpos[0]);  column j = int((x + 6) / 4)")
    ax.set_ylabel("y (qpos[1]);  row i = int((y + 6) / 4)")


def arrow(ax, r, color, ls="-", lw=1.8, alpha=1.0, label=True, z=3):
    from matplotlib.patches import FancyArrowPatch

    ax.add_patch(FancyArrowPatch((r["start_x"], r["start_y"]), (r["goal_x"], r["goal_y"]), arrowstyle="-|>", mutation_scale=11,
                                 color=color, lw=lw, ls=ls, alpha=alpha, zorder=z, shrinkA=0, shrinkB=0))
    ax.plot([r["start_x"]], [r["start_y"]], "o", ms=3.5, color=color, alpha=alpha, zorder=z)
    if label:
        mx, my = 0.62 * r["start_x"] + 0.38 * r["goal_x"], 0.62 * r["start_y"] + 0.38 * r["goal_y"]
        ax.text(mx, my, str(r["task"]), fontsize=7, color="black", zorder=z + 1, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=color, lw=0.6, alpha=0.85))


def figures(T, table, res, QUANT):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import binom

    cmap = plt.get_cmap("coolwarm")
    # fig1
    fig, ax = plt.subplots(figsize=(11, 10))
    draw_maze(ax, T)
    for r in sorted(table, key=lambda r: r["fail_rate"]):
        arrow(ax, r, cmap(r["fail_rate"]), ls="-" if r["complete"] else (0, (3, 2)))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, label="failure rate (failed cells / cells run)")
    ax.set_title("Fig 1. medium maze, 50 tasks: start (dot) -> goal (arrow head), colored by failure rate\n"
                 "solid = 40 complete tasks (18 cells), dashed = 10 incomplete tasks (4-13 cells); dark = wall")
    fig.savefig(os.path.join(HERE, "fig1_maze_tasks.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    # fig2
    fig, ax = plt.subplots(figsize=(11, 10))
    draw_maze(ax, T)
    for r in table:
        if r["task"] not in HARD9 + EASY8:
            arrow(ax, r, "#c8c8c8", lw=1.0, alpha=0.55, label=False, z=2)
    for r in table:
        if r["task"] in EASY8:
            arrow(ax, r, "#1f5fbf", lw=2.2, z=4)
    ax.plot([], [], color="#c62828", lw=2.2, label="hard 9 (straight line start -> goal; the route itself goes around walls)")
    ax.plot([], [], color="#1f5fbf", lw=2.2, label="easy 8")
    ax.legend(loc="upper center", fontsize=8, framealpha=0.9)
    for r in table:
        if r["task"] in HARD9:
            arrow(ax, r, "#c62828", lw=2.2, z=5)
    ax.set_title("Fig 2. hardest 9 tasks (red: 10-18 failures of 18) vs easiest 8 (blue: 1 failure of 18); others light gray")
    fig.savefig(os.path.join(HERE, "fig2_hard_vs_easy.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    # fig3
    C = sorted([r for r in table if r["complete"]], key=lambda r: -r["fails"])
    p = res["verification"]["failure_rate"]
    lo, hi = binom.ppf(0.025, 18, p), binom.ppf(0.975, 18, p)
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.bar(range(len(C)), [r["fails"] for r in C], color=[cmap(r["fail_rate"]) for r in C], edgecolor="k", lw=0.4)
    ax.axhline(18 * p, color="k", lw=1.2, label=f"binomial mean 18 x {p:.4f} = {18 * p:.2f}")
    ax.axhline(lo, color="k", lw=0.9, ls="--", label=f"binomial central 95% interval [{lo:.0f}, {hi:.0f}]")
    ax.axhline(hi, color="k", lw=0.9, ls="--")
    ax.set_xticks(range(len(C)))
    ax.set_xticklabels([f"t{r['task']}" for r in C], rotation=90, fontsize=8)
    ax.set_ylabel("failures out of 18 cells")
    ax.set_ylim(0, 18.6)
    n_out = sum(1 for r in C if r["fails"] < lo or r["fails"] > hi)
    ax.set_title(f"Fig 3. failures per task (40 complete tasks). {n_out} tasks outside the binomial 95% interval; "
                 f"variance {res['verification']['var_task_fail_counts']:.2f} vs binomial {res['verification']['binomial_var']:.2f}")
    ax.legend(loc="upper right")
    fig.savefig(os.path.join(HERE, "fig3_per_task_counts.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    res["fig3_n_outside_binomial95"] = n_out
    res["fig3_binomial95"] = [float(lo), float(hi)]
    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    # fig4
    H = [next(r for r in table if r["task"] == t) for t in HARD9]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for k, (key, col) in enumerate((("fails_h3", "#7fb3d5"), ("fails_h5", "#2e86c1"), ("fails_h7", "#1b4f72"))):
        ax.bar(np.arange(len(H)) + (k - 1) * 0.27, [r[key] for r in H], width=0.26, color=col, label=f"horizon {key[-1]}", edgecolor="k", lw=0.4)
    ax.set_xticks(range(len(H)))
    ax.set_xticklabels([f"t{r['task']}" for r in H])
    ax.set_ylabel("failures out of 6 planner seeds")
    ax.set_ylim(0, 6.4)
    ax.set_title("Fig 4. hardest 9 tasks: failures by top-level horizon (6 seeds each)")
    ax.legend()
    fig.savefig(os.path.join(HERE, "fig4_horizon_by_task.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    # fig5 scatter plots
    Cc = [r for r in table if r["complete"]]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    rng = np.random.default_rng(0)
    for ax, (key, label) in zip(axes.flat, QUANT):
        v = np.array([r[key] for r in Cc], dtype=float)
        jit = rng.uniform(-0.08, 0.08, len(v)) if len(set(v.tolist())) < 10 else 0
        ax.scatter(v + jit, [r["fail_rate"] for r in Cc], s=22, c="#555555")
        r22 = next(r for r in Cc if r["task"] == 22)
        ax.scatter([r22[key]], [r22["fail_rate"]], s=60, c="#c62828", zorder=3, label="t22")
        c = res["correlations"][key]
        ax.set_title(f"{label}\n" + (f"Spearman rho = {c['rho']:+.3f}, p = {c['p']:.3f}, n = 40" if c["rho"] is not None else "constant, no correlation defined"), fontsize=9)
        ax.set_ylabel("failure rate")
        ax.set_ylim(-0.04, 1.04)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Fig 5. failure rate vs simple geometric quantities (40 complete tasks; small jitter on discrete x)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig5_scatter.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
