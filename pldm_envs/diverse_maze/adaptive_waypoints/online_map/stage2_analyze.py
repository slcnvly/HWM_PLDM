# Stage 2 analysis: distance distributions per pair type, threshold sweeps,
# stop condition 2, velocity split, and the 20 closest across-wall pairs on the maze.
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
AW = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(AW, "boundary_study"))
from grid_utils import GRID_SIZE, OBS_MIN_TOTAL  # noqa: E402

SPACES = ["obs_full", "obs_pca4", "obs_pca10", "obs_pca50", "fused_full"]
TYPES = ["a_same_cell", "b_open_neighbour", "c_across_wall", "d_far"]
COLORS = {"a_same_cell": "tab:green", "b_open_neighbour": "tab:blue", "c_across_wall": "tab:red", "d_far": "0.5"}
DV_BINS = [(0, 0.5), (0.5, 1.5), (1.5, 3.0), (3.0, 99)]


def sweep(d):
    a, b, c, f = (np.array(d[t]) for t in TYPES)
    taus = np.unique(np.quantile(np.concatenate([a, b, c, f]), np.linspace(0, 1, 2001)))
    recall = np.searchsorted(np.sort(a), taus, side="right") / len(a)
    wall = np.searchsorted(np.sort(c), taus, side="right") / len(c)
    merge = np.searchsorted(np.sort(b), taus, side="right") / len(b)
    far = np.searchsorted(np.sort(f), taus, side="right") / len(f)
    return taus, recall, wall, merge, far


def best_at(taus, recall, wall, merge, far, max_wall):
    ok = wall <= max_wall
    if not ok.any():
        return None
    k = np.argmax(np.where(ok, recall, -1))
    return {"tau": float(taus[k]), "recall_same_cell": float(recall[k]), "false_across_wall": float(wall[k]),
            "merge_open_neighbour": float(merge[k]), "merge_far": float(far[k])}


def main():
    blob = json.load(open(os.path.join(HERE, "stage2_pairs.json")))
    rows = blob["rows"]
    res = {"n_pairs": {t: sum(r["type"] == t for r in rows) for t in TYPES}, "pca_top50_explained": blob["pca_top50_explained"], "spaces": {}}
    fig_h, axes_h = plt.subplots(1, len(SPACES), figsize=(4 * len(SPACES), 3.4))
    fig_c, axes_c = plt.subplots(1, len(SPACES), figsize=(4 * len(SPACES), 3.4))
    for s_i, sp in enumerate(SPACES):
        d = {t: [r[sp] for r in rows if r["type"] == t] for t in TYPES}
        taus, recall, wall, merge, far = sweep(d)
        a, c = np.array(d["a_same_cell"]), np.array(d["c_across_wall"])
        auroc_ac = roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(c))], -np.r_[a, c])
        entry = {
            "median": {t: float(np.median(v)) for t, v in d.items()},
            "auroc_same_cell_vs_across_wall": float(auroc_ac),
            "auroc_same_cell_vs_open_neighbour": float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(d["b_open_neighbour"]))], -np.r_[a, d["b_open_neighbour"]])),
            "best_at_wall_le": {f"{p:.2f}": best_at(taus, recall, wall, merge, far, p) for p in (0.02, 0.05, 0.10)},
        }
        entry["passes_stop_condition_2"] = bool(entry["best_at_wall_le"]["0.05"] and entry["best_at_wall_le"]["0.05"]["recall_same_cell"] >= 0.5)
        t5 = entry["best_at_wall_le"]["0.05"]["tau"] if entry["best_at_wall_le"]["0.05"] else None
        entry["same_cell_by_velocity_difference"] = {}
        for lo, hi in DV_BINS:
            v = np.array([r[sp] for r in rows if r["type"] == "a_same_cell" and lo <= r["dv"] < hi])
            entry["same_cell_by_velocity_difference"][f"{lo}-{hi}"] = {
                "n": int(len(v)), "median_dist": float(np.median(v)) if len(v) else None,
                "recall_at_tau_wall5": float((v <= t5).mean()) if len(v) and t5 is not None else None}
        res["spaces"][sp] = entry
        ax = axes_h[s_i]
        allv = np.concatenate([np.array(v) for v in d.values()])
        bins = np.logspace(np.log10(max(allv.min(), 1e-6)), np.log10(allv.max()), 60)
        for t in TYPES:
            ax.hist(d[t], bins=bins, histtype="step", density=True, color=COLORS[t], label=t)
        ax.set_xscale("log"); ax.set_title(f"{sp} (AUROC a-vs-c {auroc_ac:.2f})", fontsize=8); ax.legend(fontsize=6)
        ax = axes_c[s_i]
        ax.plot(taus, recall, color=COLORS["a_same_cell"], label="same-cell recognition")
        ax.plot(taus, wall, color=COLORS["c_across_wall"], label="across-wall false merge")
        ax.plot(taus, merge, color=COLORS["b_open_neighbour"], label="open-neighbour merge")
        ax.plot(taus, far, color=COLORS["d_far"], label="far (>=5) merge")
        ax.axhline(0.5, ls=":", c="k", lw=0.6); ax.axhline(0.05, ls=":", c="r", lw=0.6)
        ax.set_xscale("log"); ax.set_title(sp, fontsize=8); ax.legend(fontsize=6); ax.set_xlabel("threshold tau")
    fig_h.tight_layout(); fig_h.savefig(os.path.join(HERE, "stage2_distance_distributions.png"), dpi=110)
    fig_c.tight_layout(); fig_c.savefig(os.path.join(HERE, "stage2_threshold_curves.png"), dpi=110)
    best_space = max(SPACES, key=lambda s: (res["spaces"][s]["best_at_wall_le"]["0.05"] or {"recall_same_cell": -1})["recall_same_cell"])
    res["best_space_at_wall5"] = best_space
    res["stop_condition_2"] = {"rule": "stop unless some space/threshold gives same-cell recognition >= 50% with across-wall false merge <= 5%",
                               "passes_by_space": {s: res["spaces"][s]["passes_stop_condition_2"] for s in SPACES},
                               "stop": not any(res["spaces"][s]["passes_stop_condition_2"] for s in SPACES)}
    # 20 closest across-wall pairs (best space) on the maze
    maps = torch.load(os.path.join(AW, "..", "datasets", "r50_local", "r50_dataset", "main", "train_maps.pt"), weights_only=False)
    cw = sorted([r for r in rows if r["type"] == "c_across_wall"], key=lambda r: r[best_space])[:20]
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    for ax, r in zip(axes.flat, cw):
        L = maps[r["map"]].split("\\")
        for i in range(len(L)):
            for j in range(len(L[0])):
                if L[i][j] == "#":
                    ax.add_patch(plt.Rectangle((OBS_MIN_TOTAL + i * GRID_SIZE, OBS_MIN_TOTAL + j * GRID_SIZE), GRID_SIZE, GRID_SIZE, color="0.4"))
        (x1, y1), (x2, y2) = r["xy_a"], r["xy_b"]
        ax.plot([x1, x2], [y1, y2], "r--", lw=1)
        ax.plot(x1, y1, "o", c="tab:orange"); ax.plot(x2, y2, "o", c="tab:purple")
        lo, hi = OBS_MIN_TOTAL, OBS_MIN_TOTAL + len(L) * GRID_SIZE
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"map {r['map']} d={r[best_space]:.3g} maze {r['maze_dist']} dv {r['dv']:.1f}", fontsize=7)
    fig.suptitle(f"20 closest across-wall pairs in {best_space}", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "stage2_closest_across_wall.png"), dpi=100)
    res["closest_across_wall_pairs"] = [{k: r[k] for k in ("map", "cell_a", "cell_b", "maze_dist", "dv", best_space)} for r in cw]
    json.dump(res, open(os.path.join(HERE, "results_stage2.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "closest_across_wall_pairs"}, indent=1))


if __name__ == "__main__":
    main()
