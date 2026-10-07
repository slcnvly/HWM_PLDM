# Follow-up 2, task 5: a map grown from ONE evaluation trial. For every V0 failure,
# grow the latent map from that trial's own stage-1 replan observations only
# (rendered with the evaluator's renderer, encoded on the eval path; one frame per
# replan = every 4 MPC steps). At each re-entry into a dead-end branch already
# entered earlier in the same trial, does the first replan frame inside the branch
# join a node founded inside that branch (= "recognised as already visited")?
# False alarms: on FIRST entries, joining a node founded >= 2 maze steps away.
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stage1_failures as s1  # noqa: E402
import stage2_place_recognition as s2  # noqa: E402
from f2_task1_spikes_in_eval import RENDER, eval_path_encode, get_model  # noqa: E402
import analyze_inference as ai  # noqa: E402
from grid_utils import obs_to_ij  # noqa: E402

KS = (1, 3)


def causal_median(z, k):
    if k == 1:
        return z
    return torch.stack([z[max(0, i - k + 1):i + 1].median(0).values for i in range(len(z))])


def grow_assign(z, tau):
    """Returns per-frame (node id, joined_existing bool) with first-frame representatives."""
    R = torch.empty_like(z)
    n, nodes, joined = 0, [], []
    for k in range(len(z)):
        j = None
        if n:
            d = (R[:n] - z[k]).pow(2).mean(1)
            jj = int(torch.argmin(d))
            if float(d[jj]) <= tau:
                j = jj
        if j is None:
            R[n] = z[k]
            j = n
            n += 1
            joined.append(False)
        else:
            joined.append(True)
        nodes.append(j)
    return nodes, joined, n


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
    st2 = json.load(open(os.path.join(HERE, "results_stage2.json")))
    taus = {n: st2["spaces"]["obs_full"]["best_at_wall_le"][k]["tau"] for n, k in (("conservative", "0.02"), ("medium", "0.05"))}
    model = get_model()
    r = np.load(RENDER)
    imgs, tids, ridx = r["images"], r["trial_id"], r["replan_index"]
    trials = ai.load_variant("v0")
    l1_p90 = json.load(open(os.path.join(os.path.dirname(HERE), "inference_study", "results_inference.json")))["l1_final_cost_p90_v0_success"]
    agg = {f"k{k}_{n}": {"reentries": 0, "recognised_first_frame": 0, "recognised_any_frame_in_visit": 0, "reentry_wrong_place": 0,
                         "first_entries": 0, "first_entry_false_alarm": 0, "nodes": [], "trials": 0, "trials_with_reentry_recognised": 0,
                         "trials_with_reentry": 0}
           for k in KS for n in taus}
    per_trial = {}
    for tid, t in sorted(trials.items()):
        if t["success"]:
            continue
        ft = ai.classify(ai.trial_series(t), l1_p90)
        d = t["diag"]
        layout = d["map_key"].split("\\")
        _, dist_maze = s2.bfs_all(layout)
        goal = tuple(int(v) for v in obs_to_ij(np.asarray(d["target_xy"])))
        branches = [b for b in s1.dead_end_branches(layout) if goal not in b]
        reps = [x for x in d["replans"] if x["stage"] == "s1"]
        sel = np.where(tids == tid)[0]
        sel = sel[np.argsort(ridx[sel])]
        z = eval_path_encode(model, imgs[sel])
        rep_cells = [tuple(int(v) for v in obs_to_ij(np.asarray(reps[ridx[q]]["agent_xy"]))) for q in sel]
        # visits into each branch, in replan-frame indices
        visits = []  # (branch idx, visit number, [frame indices inside the branch during this visit])
        for b_i, b in enumerate(branches):
            inside = [c in b for c in rep_cells]
            vnum, k = 0, 0
            while k < len(inside):
                if inside[k] and (k == 0 or not inside[k - 1]):
                    run = []
                    while k < len(inside) and inside[k]:
                        run.append(k)
                        k += 1
                    if not (vnum == 0 and run[0] == 0):  # starting inside is not an entry
                        visits.append((b_i, vnum, run))
                    vnum += 1
                else:
                    k += 1
        per_trial[tid] = {"fail_types": ft, "n_frames": len(sel), "visits": [(b, v, len(run)) for b, v, run in visits]}
        for kk in KS:
            zk = causal_median(z, kk)
            for name, tau in taus.items():
                nodes, joined, n = grow_assign(zk, tau)
                found_cell = {}
                for f_i, nd in enumerate(nodes):
                    found_cell.setdefault(nd, rep_cells[f_i])
                a = agg[f"k{kk}_{name}"]
                a["nodes"].append(n)
                a["trials"] += 1
                had_re, rec_re = False, False
                for b_i, vnum, run in visits:
                    f0 = run[0]
                    b = branches[b_i]
                    if vnum == 0:
                        a["first_entries"] += 1
                        if joined[f0] and dist_maze[found_cell[nodes[f0]]].get(rep_cells[f0], 99) >= 2:
                            a["first_entry_false_alarm"] += 1
                        continue
                    had_re = True
                    a["reentries"] += 1
                    rec0 = joined[f0] and found_cell[nodes[f0]] in b
                    rec_any = any(joined[f] and found_cell[nodes[f]] in b and nodes[f] not in {nodes[g] for g in run if g < f and not joined[g]} for f in run)
                    a["recognised_first_frame"] += int(rec0)
                    a["recognised_any_frame_in_visit"] += int(rec_any)
                    if joined[f0] and not rec0 and dist_maze[found_cell[nodes[f0]]].get(rep_cells[f0], 99) >= 2:
                        a["reentry_wrong_place"] += 1
                    rec_re = rec_re or rec_any
                a["trials_with_reentry"] += int(had_re)
                a["trials_with_reentry_recognised"] += int(rec_re)
    summ = {}
    for key, a in agg.items():
        summ[key] = {
            "reentries": a["reentries"],
            "recognised_at_first_frame": a["recognised_first_frame"] / max(a["reentries"], 1),
            "recognised_any_frame_in_visit": a["recognised_any_frame_in_visit"] / max(a["reentries"], 1),
            "reentry_joined_wrong_place": a["reentry_wrong_place"] / max(a["reentries"], 1),
            "first_entries": a["first_entries"],
            "first_entry_false_alarm": a["first_entry_false_alarm"] / max(a["first_entries"], 1),
            "trials_with_reentry": a["trials_with_reentry"],
            "trials_with_reentry_recognised": a["trials_with_reentry_recognised"],
            "nodes_per_trial_mean": float(np.mean(a["nodes"])), "nodes_per_trial_max": int(max(a["nodes"])),
        }
    res = {"taus": taus, "n_failure_trials": len(per_trial), "summary": summ, "per_trial": per_trial}
    json.dump(res, open(os.path.join(HERE, "results_f2_task5_single_trial_maps.json"), "w"), indent=2)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
