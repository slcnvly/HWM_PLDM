# Online-map feasibility, stage 1 (ONLINE_MAP_PREREG.md): are there failures a
# place memory could fix? V0 diagnostic logs only (positions every MPC step).
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AW = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(AW, "boundary_study"))
sys.path.insert(0, os.path.join(AW, "inference_study"))
from grid_utils import obs_to_ij, open_neighbor_count  # noqa: E402
import analyze_inference as ai  # noqa: E402

NB = ((1, 0), (-1, 0), (0, 1), (0, -1))


def open_nbrs(layout, i, j):
    return [(i + a, j + b) for a, b in NB
            if 0 <= i + a < len(layout) and 0 <= j + b < len(layout[0]) and layout[i + a][j + b] == "O"]


def dead_end_branches(layout):
    """Chains from a degree-1 cell through degree-2 cells, stopping before the first junction (degree>=3).
    Chains that end at another dead end without a junction are dropped."""
    branches = []
    seen = set()
    for i in range(len(layout)):
        for j in range(len(layout[0])):
            if layout[i][j] != "O" or open_neighbor_count(layout, i, j) != 1 or (i, j) in seen:
                continue
            chain, prev, cur, ok = [], None, (i, j), False
            while True:
                deg = open_neighbor_count(layout, *cur)
                if deg >= 3:
                    ok = True
                    break
                chain.append(cur)
                nxt = [c for c in open_nbrs(layout, *cur) if c != prev]
                if not nxt:  # reached another dead end without a junction
                    break
                prev, cur = cur, nxt[0]
            seen.update(chain)
            if ok and chain:
                branches.append(set(chain))
    return branches


def trial_cells(t):
    d = t["diag"]
    xy = np.array([d["start_xy"]] + [s[:2] for s in d["steps"]["s1"] + d["steps"]["s2"]])
    if t["success"]:
        xy = xy[:t["steps"] + 2]
    return [tuple(int(v) for v in c) for c in obs_to_ij(xy)]


def analyse(t, fail_types):
    d = t["diag"]
    layout = d["map_key"].split("\\")
    goal = tuple(int(v) for v in obs_to_ij(np.asarray(d["target_xy"])))
    branches = [b for b in dead_end_branches(layout) if goal not in b]
    cells = trial_cells(t)
    entries = [0] * len(branches)
    for k, b in enumerate(branches):
        for t_ in range(1, len(cells)):
            if cells[t_] in b and cells[t_ - 1] not in b:
                entries[k] += 1
    # revisits: entering a cell that was visited before and then left
    last_left, visited, revisit_steps, gaps = {}, set(), 0, []
    for t_ in range(1, len(cells)):
        if cells[t_] != cells[t_ - 1]:
            last_left[cells[t_ - 1]] = t_
            if cells[t_] in visited:
                revisit_steps += 1
                gaps.append(t_ - last_left.get(cells[t_], t_))
        visited.add(cells[t_ - 1])
        visited.add(cells[t_])
    n = len(cells) - 1
    out = {"n_branches": len(branches), "branch_entries": entries, "max_entries_same_branch": max(entries) if entries else 0,
           "n_steps": n, "revisit_step_share": revisit_steps / n, "revisit_gaps": gaps}
    if not t["success"] and len(cells) > 101:
        before = set(cells[:len(cells) - 100])
        out["last100_in_previously_visited_share"] = float(np.mean([c in before for c in cells[-100:]]))
    return out


def summarise(rows):
    if not rows:
        return None
    gaps = np.concatenate([r["revisit_gaps"] for r in rows if r["revisit_gaps"]] or [np.array([])])
    total_entries = [sum(r["branch_entries"]) for r in rows]
    out = {
        "n_trials": len(rows),
        "share_with_same_branch_reentry": float(np.mean([r["max_entries_same_branch"] >= 2 for r in rows])),
        "n_with_same_branch_reentry": int(sum(r["max_entries_same_branch"] >= 2 for r in rows)),
        "branch_entries_per_trial": {"mean": float(np.mean(total_entries)), "median": float(np.median(total_entries)), "max": int(max(total_entries))},
        "max_entries_into_one_branch": {"mean": float(np.mean([r["max_entries_same_branch"] for r in rows])),
                                        "dist": np.bincount([min(r["max_entries_same_branch"], 6) for r in rows]).tolist()},
        "revisit_step_share_mean": float(np.mean([r["revisit_step_share"] for r in rows])),
        "revisit_gap_steps": {q: float(np.percentile(gaps, p)) for q, p in (("q10", 10), ("q25", 25), ("median", 50), ("q75", 75), ("q90", 90))} if len(gaps) else None,
        "n_revisits": int(len(gaps)),
    }
    l100 = [r["last100_in_previously_visited_share"] for r in rows if "last100_in_previously_visited_share" in r]
    if l100:
        out["last100_in_previously_visited"] = {"mean": float(np.mean(l100)), "share_trials_ge_0.9": float(np.mean(np.array(l100) >= 0.9)),
                                                 "share_trials_eq_1": float(np.mean(np.array(l100) >= 0.999))}
    return out


def main():
    trials = ai.load_variant("v0")
    l1_p90 = json.load(open(os.path.join(AW, "inference_study", "results_inference.json")))["l1_final_cost_p90_v0_success"]
    groups = {"wrong_path_failures": [], "all_failures": [], "successes": []}
    per_trial = {}
    for tid, t in sorted(trials.items()):
        ft = ai.classify(ai.trial_series(t), l1_p90) if not t["success"] else []
        r = analyse(t, ft)
        per_trial[tid] = {k: v for k, v in r.items() if k != "revisit_gaps"} | {"success": t["success"], "fail_types": ft}
        if t["success"]:
            groups["successes"].append(r)
        else:
            groups["all_failures"].append(r)
            if "d_wrong_path" in ft:
                groups["wrong_path_failures"].append(r)
    res = {g: summarise(rows) for g, rows in groups.items()}
    wp = res["wrong_path_failures"]
    res["stop_condition_1"] = {"rule": "stop if < 25% of wrong-path failures re-enter the same dead-end branch",
                               "value": wp["share_with_same_branch_reentry"], "stop": wp["share_with_same_branch_reentry"] < 0.25}
    res["per_trial"] = per_trial
    json.dump(res, open(os.path.join(HERE, "results_stage1.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "per_trial"}, indent=1))


if __name__ == "__main__":
    main()
