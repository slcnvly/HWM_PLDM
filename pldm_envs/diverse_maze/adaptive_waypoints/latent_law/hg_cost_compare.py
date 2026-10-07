# HIERARCHY_GEOMETRY step 4: offline comparison of L2 candidate selection by
#   (a) the V0 cost: sum_t MSE(pred_obs_t, goal latent)      (sum_all_diffs)
#   (b) first-hit time: first t with MSE <= theta (= A2 d=2 median, 0.587), else T+1 (ties -> (a))
# Candidates: K = 200 latent sequences z ~ N(0, 10 I) clamped to the [p2+0.1, p98-0.1] bounds
# (V0 sampling, nominal U = 0), horizon = that replan's L2 plan_size, rolled out on CPU by the
# trained L2 predictor from the V0 replan state (evaluator-identical render, eval-path encoding).
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import binomtest, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
AW = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(AW, "followup"))
sys.path.insert(0, os.path.join(AW, "online_map"))
import ab_saturation as ab  # noqa: E402
from common import decode_xy, load_prober  # noqa: E402
from decision_points import l2_latent_stats  # noqa: E402
from f2_task1_spikes_in_eval import RENDER  # noqa: E402
from f2_task3_latent_vs_maze_by_range import SWEEP  # noqa: E402
import analyze_inference as ai  # noqa: E402
from pldm_envs.diverse_maze.adaptive_waypoints.preprocess import normalize_images, normalize_proprio_vel  # noqa: E402

K, N_REPLANS, THETA, NOISE_VAR, VEL_SCALE = 200, 300, 0.587, 10.0, 25.647


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 10)))
    model = ab.load_model()
    prober = load_prober()
    stats = l2_latent_stats(model)
    lo, hi = torch.from_numpy(stats["lo"]).float(), torch.from_numpy(stats["hi"]).float()
    r = np.load(RENDER)
    imgs, tids, ridx = r["images"], r["trial_id"], r["replan_index"]
    g = np.load(SWEEP)
    goal_imgs = {int(t): g["goals"][k] for k, t in enumerate(g["goal_trials"])}
    trials = ai.load_variant("v0")
    # candidate replans: stage-1 replans with a logged plan, proportional over success/failure (uniform over replans)
    pool = []
    for q in range(len(tids)):
        t = trials[int(tids[q])]
        rep = [x for x in t["diag"]["replans"] if x["stage"] == "s1"][ridx[q]]
        if rep.get("l2_T") and (not t["success"] or rep["step"] <= t["steps"]):
            pool.append(q)
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(pool, size=N_REPLANS, replace=False))
    rows = []
    gen = torch.Generator().manual_seed(0)
    for n_done, q in enumerate(pick):
        tid = int(tids[q])
        t = trials[tid]
        ts = ai.trial_series(t)
        reps = ts["reps"]
        rep = reps[ridx[q]]
        steps = np.array([s[:2] for s in t["diag"]["steps"]["s1"]])
        i = rep["step"]
        v = (steps[i - 1] - steps[i - 2]) * VEL_SCALE if i >= 2 else ((steps[0] - np.asarray(t["diag"]["start_xy"])) * VEL_SCALE if i == 1 else np.zeros(2))
        with torch.no_grad():
            x = torch.stack([normalize_images(torch.from_numpy(imgs[q]).float())])
            pv = normalize_proprio_vel(torch.from_numpy(np.asarray(v, dtype=np.float32))[None])
            o = model.level1.backbone(x, proprio=pv)
            og = model.level1.backbone(torch.stack([normalize_images(torch.from_numpy(goal_imgs[tid]).float())]), proprio=torch.zeros(1, 2)).obs_component.flatten(1)
            T = int(rep["l2_T"])
            lat = torch.randn((T, K, 8), generator=gen) * np.sqrt(NOISE_VAR)
            lat = torch.max(torch.min(lat, hi), lo)
            roll = model.level2.predictor.forward_multiple(o.encodings.expand(K, -1, -1, -1).unsqueeze(0), actions=None, T=T,
                                                           proprio=o.proprio_component.expand(K, -1, -1, -1).unsqueeze(0), latents=lat)
            pobs = roll.obs_component  # (T+1, K, 16, 43, 43)
            mse = (pobs[1:].flatten(2) - og).pow(2).mean(-1)  # (T, K)
            cost_a = mse.sum(0)
            hit = (mse <= THETA)
            first = torch.where(hit.any(0), hit.float().argmax(0) + 1, torch.full((K,), T + 1))
            ka = int(torch.argmin(cost_a))
            order = sorted(range(K), key=lambda k: (int(first[k]), float(cost_a[k])))
            kb = order[0]
            wps = decode_xy(prober, pobs[1, [ka, kb]]).numpy()
            all_wp = decode_xy(prober, pobs[1]).numpy()
        f = ts["field"]
        ai_, aj = ab.obs_to_ij(np.asarray(rep["agent_xy"]))
        d_agent = int(f[ai_, aj])

        def delta(xy):
            wi, wj = ab.obs_to_ij(np.asarray(xy, dtype=np.float64))
            if 0 <= wi < f.shape[0] and 0 <= wj < f.shape[1] and f[wi, wj] >= 0:
                return int(d_agent - f[wi, wj])
            return None
        da, db = delta(wps[0]), delta(wps[1])
        dall = [delta(w) for w in all_wp]
        dall = [x for x in dall if x is not None]
        rows.append({"trial": tid, "success_trial": int(t["success"]), "step": i, "T": T, "d_agent": d_agent,
                     "delta_a": da, "delta_b": db, "same_choice": ka == kb, "any_hit": bool(hit.any()),
                     "n_hit_candidates": int(hit.any(0).sum()), "first_hit_b": int(first[kb]),
                     "delta_random_candidate_mean": float(np.mean(dall)) if dall else None,
                     "backward_rate_random_candidate": float(np.mean([x < 0 for x in dall])) if dall else None,
                     "delta_v0_logged": (lambda w: delta(w) if w else None)(rep["wp1_xy"])})
        if (n_done + 1) % 25 == 0:
            print(f"{n_done + 1}/{N_REPLANS}", flush=True)
            json.dump({"rows": rows}, open(os.path.join(HERE, "results_hg_cost_compare.json"), "w"))

    def summarise(rs):
        ok = [x for x in rs if x["delta_a"] is not None and x["delta_b"] is not None]
        ba = np.array([x["delta_a"] < 0 for x in ok]); bb = np.array([x["delta_b"] < 0 for x in ok])
        a_only, b_only = int((ba & ~bb).sum()), int((~ba & bb).sum())
        diff = np.array([x["delta_b"] - x["delta_a"] for x in ok], float)
        return {
            "n": len(rs), "n_both_decoded_in_maze": len(ok),
            "backward_rate_a": float(ba.mean()) if len(ok) else None, "backward_rate_b": float(bb.mean()) if len(ok) else None,
            "mcnemar_backward_a_only_vs_b_only": [a_only, b_only],
            "mcnemar_p": float(binomtest(a_only, a_only + b_only, 0.5).pvalue) if a_only + b_only else 1.0,
            "delta_mean_a": float(np.mean([x["delta_a"] for x in ok])) if ok else None,
            "delta_mean_b": float(np.mean([x["delta_b"] for x in ok])) if ok else None,
            "wilcoxon_p_delta": float(wilcoxon(diff).pvalue) if len(ok) and np.any(diff != 0) else 1.0,
            "same_choice_rate": float(np.mean([x["same_choice"] for x in rs])),
            "any_hit_rate": float(np.mean([x["any_hit"] for x in rs])),
            "random_candidate_backward_rate": float(np.nanmean([x["backward_rate_random_candidate"] for x in rs if x["backward_rate_random_candidate"] is not None])),
            "random_candidate_delta_mean": float(np.nanmean([x["delta_random_candidate_mean"] for x in rs if x["delta_random_candidate_mean"] is not None])),
            "v0_logged_backward_rate": float(np.mean([x["delta_v0_logged"] < 0 for x in rs if x["delta_v0_logged"] is not None])),
            "v0_logged_delta_mean": float(np.mean([x["delta_v0_logged"] for x in rs if x["delta_v0_logged"] is not None])),
        }
    res = {"settings": {"K": K, "n_replans": N_REPLANS, "theta": THETA, "noise_var": NOISE_VAR, "l2_bounds_lo": stats["lo"].tolist(), "l2_bounds_hi": stats["hi"].tolist()},
           "all": summarise(rows), "replans_with_any_hit": summarise([x for x in rows if x["any_hit"]]),
           "replans_without_hit": summarise([x for x in rows if not x["any_hit"]]),
           "success_trials": summarise([x for x in rows if x["success_trial"]]), "failure_trials": summarise([x for x in rows if not x["success_trial"]]),
           "rows": rows}
    json.dump(res, open(os.path.join(HERE, "results_hg_cost_compare.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
