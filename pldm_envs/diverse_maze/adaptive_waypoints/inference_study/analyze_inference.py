# Analysis for the inference-design study (INFERENCE_PREREG.md). Reads the kernel
# outputs (<variant>_progress.json + diag/<label>.json) downloaded under
# experiments/kaggle_inference_<variant>/output/ and writes results_inference.json.
#
# usage: analyze_inference.py [--plots]   (variants without output are reported as not run)
import argparse
import glob
import json
import os
import sys
from collections import deque

import numpy as np
from scipy.stats import binomtest, spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "boundary_study")))
from grid_utils import GRID_SIZE, OBS_MIN_TOTAL, obs_to_ij, open_neighbor_count  # noqa: E402

EXP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "..", "experiments"))
VARIANTS = ["v0", "v1", "v2", "v3", "v4"]
FAIL_TYPES = ["a_stagnation", "b_oscillation", "c_unreachable_carrot", "d_wrong_path", "e_slow_progress"]
SIGNALS = ["l1_final_cost", "wp1_move_latent", "wp1_move_xy", "ess_l2", "ess_l1", "l2_clamp_frac"]


# ----------------------------------------------------------------------------- loading
def load_variant(name):
    out = os.path.join(EXP, f"kaggle_inference_{name}", "output")
    prog = glob.glob(os.path.join(out, "**", f"{name}_progress.json"), recursive=True)
    if not prog:
        return None
    progress = json.load(open(prog[0]))
    diag = {}
    for f in glob.glob(os.path.join(os.path.dirname(prog[0]), "diag", "*.json")):
        label = os.path.splitext(os.path.basename(f))[0]
        diag[label] = json.load(open(f))
    trials = {}
    for label, p in progress.items():
        if p["per_trial"] is None:
            continue
        for j, (succ, steps) in enumerate(zip(p["per_trial"]["success"], p["per_trial"]["steps"])):
            tid = p["start"] + j
            d = diag.get(label, {}).get(str(tid))
            trials[tid] = {"label": label, "success": int(succ), "steps": int(steps), "diag": d}
    return trials


def layout_of(d):
    return d["map_key"].split("\\")


def bfs_field(layout, goal_ij):
    rows, cols = len(layout), len(layout[0])
    dist = np.full((rows, cols), -1, dtype=int)
    dist[goal_ij] = 0
    q = deque([goal_ij])
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols and layout[ni][nj] != "#" and dist[ni, nj] < 0:
                dist[ni, nj] = dist[i, j] + 1
                q.append((ni, nj))
    return dist


def in_wall(layout, xy):
    i, j = (int(v) for v in obs_to_ij(np.asarray(xy, dtype=np.float64)))
    rows, cols = len(layout), len(layout[0])
    return not (0 <= i < rows and 0 <= j < cols) or layout[i][j] == "#"


def trial_series(t):
    """Per-step BFS distance d(t) (positions after each MPC step), stage-1 replans, flags."""
    d = t["diag"]
    layout = layout_of(d)
    goal = tuple(int(v) for v in obs_to_ij(np.asarray(d["target_xy"])))
    field = bfs_field(layout, goal)
    xy = np.array([s[:2] for s in d["steps"]["s1"] + d["steps"]["s2"]])
    ij = obs_to_ij(xy)
    dist = np.array([field[a, b] if 0 <= a < len(layout) and 0 <= b < len(layout[0]) else -1 for a, b in ij])
    reps = [r for r in d["replans"] if r["stage"] == "s1"]
    prev = None
    for r in reps:
        r["wp1_in_wall"] = None if r["wp1_xy"] is None else in_wall(layout, r["wp1_xy"])
        r["wp1_move_xy"] = None if (prev is None or r["wp1_xy"] is None) else float(np.linalg.norm(np.subtract(r["wp1_xy"], prev)))
        prev = r["wp1_xy"]
    start_d = field[tuple(int(v) for v in obs_to_ij(np.asarray(d["start_xy"])))]
    return {"layout": layout, "field": field, "dist": dist, "xy": xy, "reps": reps, "start_d": int(start_d),
            "agent_in_wall_steps": int((dist < 0).sum())}


# ----------------------------------------------------------------------------- failure types
def classify(ts, l1_cost_p90):
    d = ts["dist"].astype(float)
    d[d < 0] = np.nan
    T = len(d)
    types = []
    # (a) stagnation: last 100 steps never get 1 cell closer than d(T-101)
    if T > 101 and np.nanmin(d[T - 100:]) > d[T - 101] - 1:
        types.append("a_stagnation")
    # (b) oscillation
    reps = ts["reps"]
    moves = []
    for r0, r1 in zip(reps[:-1], reps[1:]):
        if r0["wp1_xy"] is not None and r1["wp1_xy"] is not None:
            moves.append(np.subtract(r1["wp1_xy"], r0["wp1_xy"]))
    big = [m for m in moves if np.linalg.norm(m) > GRID_SIZE]
    flips = sum(1 for m0, m1 in zip(big[:-1], big[1:])
                if np.dot(m0, m1) / (np.linalg.norm(m0) * np.linalg.norm(m1)) < -0.5)
    if reps and flips >= 5 and flips >= 0.1 * len(reps):
        types.append("b_oscillation")
    # (c) unreachable carrot
    bad = [((r["wp1_in_wall"] is True) or (r["l1_final_cost"] > l1_cost_p90)) for r in reps]
    if reps and np.mean(bad) >= 0.3:
        types.append("c_unreachable_carrot")
    # (d) wrong path
    run_min = np.fmin.accumulate(np.nan_to_num(d, nan=np.inf))
    if np.nanmax(d - run_min) >= 3:
        types.append("d_wrong_path")
    # (e) slow progress
    ok = ~np.isnan(d)
    rho = spearmanr(np.arange(T)[ok], d[ok]).correlation if ok.sum() > 2 and np.nanstd(d) > 0 else 0
    if d[-1] <= ts["start_d"] - 1 and rho < -0.5 and "a_stagnation" not in types and "d_wrong_path" not in types:
        types.append("e_slow_progress")
    return types or ["unclassified"]


# ----------------------------------------------------------------------------- stats
def wilson(k, n, z=1.959964):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


def holm(pvals):
    """pvals: dict name -> p or None (not run -> treated as p=1, m fixed)."""
    names = list(pvals)
    ps = np.array([1.0 if pvals[n] is None else pvals[n] for n in names])
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[idx]))
        adj[idx] = running
    return {n: (None if pvals[n] is None else float(adj[i])) for i, n in enumerate(names)}


def paired(v0, vx):
    ids = sorted(set(v0) & set(vx))
    a = np.array([v0[i]["success"] for i in ids])
    b = np.array([vx[i]["success"] for i in ids])
    v0_fail_vx_succ = int(((a == 0) & (b == 1)).sum())
    v0_succ_vx_fail = int(((a == 1) & (b == 0)).sum())
    disc = v0_fail_vx_succ + v0_succ_vx_fail
    p_mc = float(binomtest(v0_fail_vx_succ, disc, 0.5).pvalue) if disc else 1.0
    both = [i for i in ids if v0[i]["success"] and vx[i]["success"]]
    diff = np.array([vx[i]["steps"] - v0[i]["steps"] for i in both], dtype=float)
    if len(diff) and np.any(diff != 0):
        p_w = float(wilcoxon(diff, zero_method="wilcox").pvalue)
    else:
        p_w = 1.0
    return {"n_paired": len(ids), "v0_fail_vx_success": v0_fail_vx_succ, "v0_success_vx_fail": v0_succ_vx_fail,
            "mcnemar_p": p_mc, "n_both_success": len(both),
            "steps_mean_diff_vx_minus_v0": float(diff.mean()) if len(diff) else None,
            "steps_median_diff": float(np.median(diff)) if len(diff) else None,
            "n_vx_faster": int((diff < 0).sum()), "n_vx_slower": int((diff > 0).sum()), "wilcoxon_p": p_w}


def junction_fraction(t):
    """Share of junction cells (>=3 open neighbours) on a BFS shortest path start->goal."""
    ts = t["_ts"]
    layout, field = ts["layout"], ts["field"]
    cur = tuple(int(v) for v in obs_to_ij(np.asarray(t["diag"]["start_xy"])))
    path = [cur]
    while field[cur] > 0:
        i, j = cur
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (i + di, j + dj)
            if 0 <= n[0] < len(layout) and 0 <= n[1] < len(layout[0]) and field[n] == field[cur] - 1:
                cur = n
                break
        path.append(cur)
    return float(np.mean([open_neighbor_count(layout, *c) >= 3 for c in path]))


# ----------------------------------------------------------------------------- per-variant summary
def summarize(name, trials, l1_cost_p90):
    ids = sorted(trials)
    n = len(ids)
    k = sum(trials[i]["success"] for i in ids)
    succ_steps = [trials[i]["steps"] for i in ids if trials[i]["success"]]
    counts = {f: 0 for f in FAIL_TYPES + ["unclassified"]}
    for i in ids:
        t = trials[i]
        if t["diag"] is None:
            continue
        t["_ts"] = trial_series(t)
        if not t["success"]:
            t["fail_types"] = classify(t["_ts"], l1_cost_p90)
            for f in t["fail_types"]:
                counts[f] += 1
    n_fail = n - k
    sig = {}
    for s in ["l1_final_cost", "wp1_move_latent", "wp1_move_xy", "ess_l2", "ess_l1", "l2_clamp_frac", "l2_best_cost"]:
        vals = {"success": [], "fail": []}
        for i in ids:
            t = trials[i]
            if t.get("_ts") is None:
                continue
            v = [r[s] for r in t["_ts"]["reps"] if r.get(s) is not None]
            if v:
                vals["success" if t["success"] else "fail"].append(float(np.mean(v)))
        sig[s] = {g: {"median": float(np.median(x)) if x else None, "iqr": [float(np.percentile(x, 25)), float(np.percentile(x, 75))] if x else None, "n": len(x)}
                  for g, x in vals.items()}
    wp_wall = [r["wp1_in_wall"] for i in ids if trials[i].get("_ts") for r in trials[i]["_ts"]["reps"] if r["wp1_in_wall"] is not None]
    return {
        "n": n, "k": k, "rate": k / n if n else None, "wilson95": wilson(k, n) if n else None,
        "avg_steps_success": float(np.mean(succ_steps)) if succ_steps else None,
        "median_steps_success": float(np.median(succ_steps)) if succ_steps else None,
        "n_fail": n_fail, "fail_type_counts": counts,
        "fail_type_rates": {f: (c / n_fail if n_fail else None) for f, c in counts.items()},
        "signal_trial_means_by_outcome": sig,
        "wp1_in_wall_rate_all_replans": float(np.mean(wp_wall)) if wp_wall else None,
        "agent_in_wall_steps_total": int(sum(trials[i]["_ts"]["agent_in_wall_steps"] for i in ids if trials[i].get("_ts"))),
    }


def early_auroc(trials, t_max=100):
    out = {}
    ids = [i for i in sorted(trials) if trials[i].get("_ts")]
    y = np.array([1 - trials[i]["success"] for i in ids])  # 1 = failure
    for s in SIGNALS + ["l2_best_cost"]:
        x, yy = [], []
        for i, lab in zip(ids, y):
            v = [r[s] for r in trials[i]["_ts"]["reps"] if r["step"] < t_max and r.get(s) is not None]
            if v:
                x.append(float(np.mean(v)))
                yy.append(lab)
        out[s] = float(roc_auc_score(yy, x)) if len(set(yy)) == 2 else None
    return out


# ----------------------------------------------------------------------------- validation
def validation():
    out = os.path.join(EXP, "kaggle_inference_validate", "output")
    prog = glob.glob(os.path.join(out, "**", "validate_progress.json"), recursive=True)
    if not prog:
        return None
    p = json.load(open(prog[0]))
    diag_dir = os.path.join(os.path.dirname(prog[0]), "diag")
    a = p["cs20_0"]["per_trial"]
    b10 = p["cs10_0"]["per_trial"], p["cs10_1"]["per_trial"]
    b = {"success": b10[0]["success"] + b10[1]["success"], "steps": b10[0]["steps"] + b10[1]["steps"]}
    same_s = [x == y for x, y in zip(a["success"], b["success"])]
    same_t = [x == y for x, y in zip(a["steps"], b["steps"])]
    traj = {}
    d20 = json.load(open(os.path.join(diag_dir, "cs20_0.json")))
    d10 = {**json.load(open(os.path.join(diag_dir, "cs10_0.json"))), **json.load(open(os.path.join(diag_dir, "cs10_1.json")))}
    first_div = []
    for tid in range(20):
        s20 = np.array([s[:2] for s in d20[str(tid)]["steps"]["s1"] + d20[str(tid)]["steps"]["s2"]])
        s10 = np.array([s[:2] for s in d10[str(tid)]["steps"]["s1"] + d10[str(tid)]["steps"]["s2"]])
        diff = np.abs(s20 - s10).max(axis=1)
        nz = np.nonzero(diff > 0)[0]
        first_div.append(int(nz[0]) if len(nz) else None)
        traj[tid] = float(diff.max())
    return {
        "success_identical": int(sum(same_s)), "steps_identical": int(sum(same_t)), "n": 20,
        "pass": all(same_s) and all(same_t),
        "trajectories_bit_identical": int(sum(v == 0 for v in traj.values())),
        "max_xy_diff_per_trial": traj, "first_divergent_step_per_trial": first_div,
        "cs20": a, "cs10": b,
    }


# ----------------------------------------------------------------------------- plots
def plot_trials(name, trials, max_success=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs = []
    fails = [i for i in sorted(trials) if not trials[i]["success"] and trials[i].get("_ts")]
    succs = [i for i in sorted(trials) if trials[i]["success"] and trials[i].get("_ts")][:max_success]
    for i in fails + succs:
        t = trials[i]
        ts = t["_ts"]
        layout = ts["layout"]
        fig, ax = plt.subplots(figsize=(5, 5))
        for a in range(len(layout)):
            for b in range(len(layout[0])):
                if layout[a][b] == "#":
                    ax.add_patch(plt.Rectangle((OBS_MIN_TOTAL + a * GRID_SIZE, OBS_MIN_TOTAL + b * GRID_SIZE), GRID_SIZE, GRID_SIZE, color="0.35"))
        xy = ts["xy"]
        ax.plot(xy[:, 0], xy[:, 1], "-", color="tab:blue", lw=1.2, label="agent")
        wp = np.array([r["wp1_xy"] for r in ts["reps"] if r["wp1_xy"] is not None])
        if len(wp):
            ax.scatter(wp[:, 0], wp[:, 1], c=np.arange(len(wp)), cmap="autumn", s=8, label="waypoint 1 (decoded)", zorder=3)
        ax.plot(*t["diag"]["start_xy"], "go", ms=8, label="start")
        ax.plot(*t["diag"]["target_xy"], "r*", ms=14, label="goal")
        lo, hi = OBS_MIN_TOTAL, OBS_MIN_TOTAL + len(layout) * GRID_SIZE
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        status = "SUCCESS" if t["success"] else "FAIL " + ",".join(t.get("fail_types", []))
        ax.set_title(f"{name} trial {i}: {status} (steps {t['steps']})", fontsize=8)
        ax.legend(fontsize=6, loc="upper right")
        figs.append((f"{name}/trial_{i:03d}_{'S' if t['success'] else 'F'}", fig))
    return figs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()
    res = {"validation": validation(), "variants": {}, "comparisons": {}, "not_run": []}
    data = {v: load_variant(v) for v in VARIANTS}
    # p90 of L1 final cost over all stage-1 replans of V0 successes (prereg SS6c)
    l1_p90 = None
    if data["v0"]:
        vals = []
        for t in data["v0"].values():
            if t["success"] and t["diag"]:
                vals += [r["l1_final_cost"] for r in t["diag"]["replans"] if r["stage"] == "s1"]
        l1_p90 = float(np.percentile(vals, 90))
    res["l1_final_cost_p90_v0_success"] = l1_p90
    for v in VARIANTS:
        if data[v] is None:
            res["not_run"].append(v)
            continue
        res["variants"][v] = summarize(v, data[v], l1_p90)
        res["variants"][v]["early_auroc_first100"] = early_auroc(data[v])
        if v != "v0" and data["v0"]:
            res["comparisons"][v] = paired(data["v0"], data[v])
    for metric, key in (("success", "mcnemar_p"), ("steps", "wilcoxon_p")):
        adj = holm({v: (res["comparisons"][v][key] if v in res["comparisons"] else None) for v in VARIANTS[1:]})
        for v, a in adj.items():
            if v in res["comparisons"]:
                res["comparisons"][v][f"{key}_holm"] = a
    # V4 straight-vs-junction split
    if data["v0"] and data["v4"]:
        jf = {i: junction_fraction(data["v0"][i]) for i in data["v0"] if data["v0"][i].get("_ts")}
        med = float(np.median(list(jf.values())))
        split = {}
        for g, sel in (("straight_heavy", lambda f: f <= med), ("junction_heavy", lambda f: f > med)):
            ids = [i for i, f in jf.items() if sel(f) and i in data["v4"]]
            split[g] = {"n": len(ids),
                        "v0_success": sum(data["v0"][i]["success"] for i in ids),
                        "v4_success": sum(data["v4"][i]["success"] for i in ids),
                        **{k: paired({i: data["v0"][i] for i in ids}, {i: data["v4"][i] for i in ids})[k]
                           for k in ("steps_mean_diff_vx_minus_v0", "n_both_success", "wilcoxon_p", "v0_fail_vx_success", "v0_success_vx_fail")}}
        res["v4_split_by_junction_fraction"] = {"median_junction_fraction": med, **split}
    out_path = os.path.join(HERE, "results_inference.json")
    json.dump(res, open(out_path, "w"), indent=2, default=lambda o: None)
    print(json.dumps({k: res[k] for k in ("not_run", "comparisons")}, indent=1, default=str))
    if args.plots:
        import wandb
        run = wandb.init(project="hwm-boundary-study", name="inference_study_trajectories", job_type="analysis")
        for v in VARIANTS:
            if data[v] is None:
                continue
            for key, fig in plot_trials(v, data[v]):
                run.log({key: wandb.Image(fig)})
                import matplotlib.pyplot as plt
                plt.close(fig)
        for v, s in res["variants"].items():
            for k2 in ("rate", "avg_steps_success"):
                run.summary[f"{v}/{k2}"] = s[k2]
        print("wandb:", run.url)
        run.finish()


if __name__ == "__main__":
    main()
