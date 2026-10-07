# Follow-up 2, task 3: at every V0 stage-1 replan, (a) obs-latent distance from the
# current observation to the goal observation (planner-style mean squared difference,
# eval-path encoding of evaluator-identical renders), (b) maze BFS steps agent->goal.
# Spearman(a, b) within maze-distance bins 1-2, 3-5, 6-10, 11+, for successful
# (until the goal step) and failed trials.
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from f2_task1_spikes_in_eval import RENDER, eval_path_encode, get_model  # noqa: E402
import analyze_inference as ai  # noqa: E402
from grid_utils import obs_to_ij  # noqa: E402

SWEEP = os.path.abspath(os.path.join(os.path.dirname(RENDER), "..", "..", "kaggle_render_sweeps", "output", "sweep_and_goal_images.npz"))
BINS = [(1, 2), (3, 5), (6, 10), (11, 99)]


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 6)))
    model = get_model()
    r = np.load(RENDER)
    imgs, tids, ridx = r["images"], r["trial_id"], r["replan_index"]
    g = np.load(SWEEP)
    zg = eval_path_encode(model, g["goals"])
    goal_of = {int(t): k for k, t in enumerate(g["goal_trials"])}
    trials = ai.load_variant("v0")
    rows = []
    for tid, t in sorted(trials.items()):
        sel = np.where(tids == tid)[0]
        sel = sel[np.argsort(ridx[sel])]
        z = eval_path_encode(model, imgs[sel])
        ts = ai.trial_series(t)
        f = ts["field"]
        reps = ts["reps"]
        lat = (z - zg[goal_of[tid]]).pow(2).mean(1).numpy()
        for k, q in enumerate(sel):
            rep = reps[ridx[q]]
            if t["success"] and rep["step"] > t["steps"]:
                break
            i, j = obs_to_ij(np.asarray(rep["agent_xy"]))
            md = int(f[i, j]) if 0 <= i < f.shape[0] and 0 <= j < f.shape[1] else -1
            if md < 0:
                continue
            rows.append((int(t["success"]), md, float(lat[k]), tid))
    rows = np.array(rows, dtype=float)
    res = {"n_replans": int(len(rows)), "groups": {}}
    for gname, m in (("all", np.ones(len(rows), bool)), ("success", rows[:, 0] == 1), ("failure", rows[:, 0] == 0)):
        R = rows[m]
        entry = {"overall_spearman": float(spearmanr(R[:, 1], R[:, 2]).correlation), "n": int(len(R)), "bins": {}}
        for lo, hi in BINS:
            b = R[(R[:, 1] >= lo) & (R[:, 1] <= hi)]
            entry["bins"][f"{lo}-{hi if hi < 99 else '+'}"] = {
                "n": int(len(b)),
                "spearman": float(spearmanr(b[:, 1], b[:, 2]).correlation) if len(b) > 10 and len(set(b[:, 1])) > 1 else None,
                "latent_dist_median": float(np.median(b[:, 2])) if len(b) else None,
            }
        entry["latent_dist_median_by_maze_steps"] = {int(s): float(np.median(R[R[:, 1] == s, 2])) for s in sorted(set(R[:, 1].astype(int))) if (R[:, 1] == s).sum() >= 20}
        res["groups"][gname] = entry
    json.dump(res, open(os.path.join(HERE, "results_f2_task3_latent_vs_maze_by_range.json"), "w"), indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
