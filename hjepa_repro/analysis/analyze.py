"""Stage 6 analysis of the top-level horizon sweep (PREREG.md). Reads sweep_raw.jsonl and the
frozen prereg.json only; separate from the run scripts.

  python analyze.py --raw ../sweep_raw.jsonl --prereg ../prereg.json --out ../sweep_summary.json
"""
import argparse
import json

import numpy as np

NULL_REPS, BOOT_REPS = 1000, 2000
NULL_SEED, BOOT_SEED = 20261011, 20261012


def load(raw_path, pre):
    """Y[task, horizon, seed] success (nan = not run), W = wall time per task, E = env steps."""
    hs, seeds = pre["horizons_priority"], pre["seeds_A"] + pre["seeds_B"]
    shape = (pre["n_tasks"], len(hs), len(seeds))
    Y, W, E = (np.full(shape, np.nan) for _ in range(3))
    n = 0
    for line in open(raw_path):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["horizon"] not in hs or r["seed"] not in seeds or r["task"] >= pre["n_tasks"]:
            continue
        i = (r["task"], hs.index(r["horizon"]), seeds.index(r["seed"]))
        assert r["split"] == ("A" if r["seed"] in pre["seeds_A"] else "B"), r
        Y[i], W[i] = r["success"], r["wall_time"]
        E[i] = r["env_steps"] if r["env_steps"] is not None else np.nan
        n += 1
    return Y, W, E, n


def masks(Y, pre):
    """Seeds usable for a task = seeds where every candidate horizon finished. A task is analyzed
    when it has >= min_seeds_per_half such seeds in A and in B (so every (task, candidate) pair
    used has at least that many seeds in both halves)."""
    nA = len(pre["seeds_A"])
    complete = ~np.isnan(Y).any(axis=1)
    mA, mB = complete.copy(), complete.copy()
    mA[:, nA:] = False
    mB[:, :nA] = False
    keep = (mA.sum(1) >= pre["min_seeds_per_half"]) & (mB.sum(1) >= pre["min_seeds_per_half"])
    return mA, mB, np.where(keep)[0]


def rates(X, mask):
    w = mask[:, None, :].astype(float)
    return np.where(w > 0, np.nan_to_num(X), 0).sum(2) / np.maximum(w.sum(2), 1)


def pick_fixed(rA):
    m = rA.mean(0)
    return int(np.flatnonzero(np.isclose(m, m.max()))[0]), m  # ties -> higher priority (lower index)


def pick_per_task(rA, g):
    """Best A rate per task; ties -> the global fixed best g if tied, else the highest-priority tied one."""
    choice = np.zeros(len(rA), dtype=int)
    for i, r in enumerate(rA):
        tied = np.flatnonzero(np.isclose(r, r.max()))
        choice[i] = g if g in tied else tied[0]
    return choice


def gap_stats(rA, rB):
    g, _ = pick_fixed(rA)
    choice = pick_per_task(rA, g)
    fixed = float(rB[:, g].mean())
    oracle = float(rB[np.arange(len(rB)), choice].mean())
    return {"fixed": fixed, "oracle": oracle, "gap": oracle - fixed, "g": g, "choice": choice}


def analyze(Y, W, E, pre, n_lines):
    hs = pre["horizons_priority"]
    mA, mB, keep = masks(Y, pre)
    out = {"n_trial_lines": n_lines, "n_tasks_analyzed": int(len(keep)), "tasks_analyzed": keep.tolist(),
           "usable_seeds_A_per_task": mA.sum(1)[keep].tolist(), "usable_seeds_B_per_task": mB.sum(1)[keep].tolist(),
           "trials_finished_per_horizon": {str(h): int((~np.isnan(Y[:, k, :])).sum()) for k, h in enumerate(hs)}}
    if len(keep) == 0:
        out["verdict"] = "undetermined"
        out["verdict_reason"] = "no task has enough finished seeds in both halves"
        return out
    Yk, Wk, mAk, mBk = Y[keep], W[keep], mA[keep], mB[keep]
    allm = mAk | mBk
    rA, rB, rAll = rates(Yk, mAk), rates(Yk, mBk), rates(Yk, allm)
    obs = gap_stats(rA, rB)
    n = len(keep)
    out.update({"fixed_horizon": hs[obs["g"]], "fixed_B": obs["fixed"], "oracle_B": obs["oracle"], "gap": obs["gap"],
                "per_task_choice": [hs[c] for c in obs["choice"]],
                "choice_counts": {str(h): int((obs["choice"] == k).sum()) for k, h in enumerate(hs)}})
    # how many tasks have a strict single best horizon on A, and on the pooled data
    out["tasks_with_unique_best_on_A"] = int(sum(np.isclose(r, r.max()).sum() == 1 for r in rA))
    best_all = [[hs[k] for k in np.flatnonzero(np.isclose(r, r.max()))] for r in rAll]
    out["pooled_best_counts_ties_split"] = {str(h): float(sum(1 / len(b) for b in best_all if h in b)) for h in hs}

    naive_fixed, naive_oracle = float(rAll.mean(0).max()), float(rAll.max(1).mean())
    out["naive"] = {"fixed": naive_fixed, "oracle": naive_oracle, "gap": naive_oracle - naive_fixed,
                    "selection_bias": (naive_oracle - naive_fixed) - obs["gap"]}

    rng = np.random.default_rng(NULL_SEED)
    null = np.empty(NULL_REPS)
    for b in range(NULL_REPS):
        perm = np.argsort(rng.random(Yk.shape), axis=1)  # independent shuffle of candidate labels per (task, seed)
        Yp = np.take_along_axis(Yk, perm, axis=1)
        null[b] = gap_stats(rates(Yp, mAk), rates(Yp, mBk))["gap"]
    out["null"] = {"reps": NULL_REPS, "seed": NULL_SEED, "mean": float(null.mean()),
                   "p5": float(np.percentile(null, 5)), "p50": float(np.percentile(null, 50)), "p95": float(np.percentile(null, 95)),
                   "observed_percentile": float((null <= obs["gap"]).mean() * 100),
                   "observed_percentile_strict": float((null < obs["gap"]).mean() * 100)}

    rng = np.random.default_rng(BOOT_SEED)
    boot = np.empty((BOOT_REPS, 3))
    for b in range(BOOT_REPS):
        idx = rng.integers(0, n, n)
        s = gap_stats(rA[idx], rB[idx])
        boot[b] = (s["fixed"], s["oracle"], s["gap"])
    ci = lambda c: [float(np.percentile(boot[:, c], 2.5)), float(np.percentile(boot[:, c], 97.5))]
    out["ci95"] = {"reps": BOOT_REPS, "seed": BOOT_SEED, "fixed": ci(0), "oracle": ci(1), "gap": ci(2)}

    tA, tB, tAll = rates(Wk, mAk), rates(Wk, mBk), rates(Wk, allm)
    out["per_horizon"] = {str(h): {"success_A": float(rA[:, k].mean()), "success_B": float(rB[:, k].mean()),
                                   "success_all": float(rAll[:, k].mean()), "seconds_per_task": float(tAll[:, k].mean()),
                                   "env_steps_mean_successes": float(np.nanmean(np.where(allm[:, None, :], E[keep], np.nan)[:, k, :]))
                                   if np.isfinite(np.where(allm[:, None, :], E[keep], np.nan)[:, k, :]).any() else None}
                          for k, h in enumerate(hs)}
    chosen_t = float(tB[np.arange(n), obs["choice"]].mean())
    fixed_t = float(tB[:, obs["g"]].mean())
    out["compute"] = {"chosen_seconds_per_task": chosen_t, "fixed_seconds_per_task": fixed_t, "ratio": chosen_t / fixed_t}

    above = obs["gap"] > out["null"]["p95"]
    ci_ok = out["ci95"]["gap"][0] > 0.03
    comp_ok = out["compute"]["ratio"] < 1.5
    out["verdict_inputs"] = {"gap_above_null_p95": bool(above), "ci_lower_above_3pp": bool(ci_ok), "time_ratio_below_1.5": bool(comp_ok)}
    out["verdict"] = "no gap" if not above else ("gap" if (ci_ok and comp_ok) else "undetermined")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pre = json.load(open(a.prereg))
    Y, W, E, n = load(a.raw, pre)
    out = analyze(Y, W, E, pre, n)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("tasks_analyzed", "per_task_choice", "usable_seeds_A_per_task",
                                                                 "usable_seeds_B_per_task")}, indent=1))


if __name__ == "__main__":
    main()
