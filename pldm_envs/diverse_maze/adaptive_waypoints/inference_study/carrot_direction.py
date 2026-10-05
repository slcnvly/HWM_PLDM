# Post-hoc (not pre-registered): does waypoint 1 point "forward" in maze distance?
# Per stage-1 replan: progress = BFS(agent cell) - BFS(wp1 cell); >0 forward, <0 backward, 0 same distance.
import json
import sys

import numpy as np

import analyze_inference as a
from grid_utils import obs_to_ij


def carrot_stats(trials):
    out = {"success": [], "fail": []}
    for t in trials.values():
        if t["diag"] is None:
            continue
        ts = a.trial_series(t)
        f, L = ts["field"], ts["layout"]
        fw = bw = same = wall = 0
        for r in ts["reps"]:
            if r["wp1_xy"] is None:
                continue
            ai, aj = obs_to_ij(np.asarray(r["agent_xy"]))
            wi, wj = obs_to_ij(np.asarray(r["wp1_xy"]))
            if not (0 <= wi < len(L) and 0 <= wj < len(L[0])) or f[wi, wj] < 0:
                wall += 1
                continue
            p = f[ai, aj] - f[wi, wj]
            fw += p > 0
            bw += p < 0
            same += p == 0
        n = fw + bw + same + wall
        out["success" if t["success"] else "fail"].append({"forward": fw / n, "backward": bw / n, "same_cell_dist": same / n, "wall": wall / n})
    summ = {}
    for g, rows in out.items():
        summ[g] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else None
        summ[g]["n_trials"] = len(rows)
    return summ


if __name__ == "__main__":
    res = {}
    for v in sys.argv[1:]:
        tr = a.load_variant(v)
        if tr:
            res[v] = carrot_stats(tr)
    json.dump(res, open("results_carrot_direction.json", "w"), indent=2)
    print(json.dumps(res, indent=1))
