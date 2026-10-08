# PROBE_SUBSPACE analysis: aggregate ps_main/ps_mlp outputs, curves + radius (ab_saturation),
# Spearman by distance band, embedding share, PCA radii, preregistered verdict.
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
AW = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(AW, "latent_law"))
import ab_saturation as ab  # noqa: E402

REPS = ["B1_trained", "B2_random_seed0", "B2_random_seed1", "B2_random_seed2", "B4_pixels"]
BANDS = [(1, 2), (3, 5), (6, 9), (10, 15)]


def main():
    R = json.load(open(os.path.join(HERE, "results_ps_main.json")))["maps"]
    P = np.load(os.path.join(HERE, "ps_pairs.npz"))
    maps = sorted(R, key=int)
    out = {"n_maps": len(maps)}
    # P1 / P3
    p1 = {}
    for rep in REPS:
        p1[rep] = {}
        for ver in ("excl", "incl"):
            r2s = np.array([R[m]["reps"][rep][ver]["r2_xy_mean"][2] for m in maps])
            errs = np.array([R[m]["reps"][rep][ver]["err_cells_median"] for m in maps])
            s = [R[m]["reps"][rep][ver]["s_best"] for m in maps]
            p1[rep][ver] = {"r2_median": float(np.median(r2s)), "r2_min": float(r2s.min()), "r2_max": float(r2s.max()),
                            "r2_x_median": float(np.median([R[m]["reps"][rep][ver]["r2_xy_mean"][0] for m in maps])),
                            "r2_y_median": float(np.median([R[m]["reps"][rep][ver]["r2_xy_mean"][1] for m in maps])),
                            "err_cells_median_of_maps": float(np.median(errs)), "err_cells_max_map": float(errs.max()),
                            "s_best_counts": {str(k): int(s.count(k)) for k in sorted(set(s))},
                            "r2_fold0_median": float(np.median([R[m]["reps"][rep][ver]["r2_fold0"][2] for m in maps]))}
    out["P1_P3_ridge"] = p1
    b1 = p1["B1_trained"]["excl"]["r2_median"]
    b2 = float(np.mean([p1[f"B2_random_seed{s}"]["excl"]["r2_median"] for s in range(3)]))
    b4 = p1["B4_pixels"]["excl"]["r2_median"]
    out["P3_ratios_excl"] = {"B2_mean_over_B1": b2 / b1, "B4_over_B1": b4 / b1}
    # pairs
    def cat(key):
        return np.concatenate([P[f"{m}|{key}"] for m in maps])
    d_maze, d_xy, keep0, sp = cat("pairs_d_maze"), cat("pairs_d_xy"), cat("pairs_keep0").astype(bool), cat("pairs_spike").astype(bool)
    p4 = {}
    for ver in ("excl", "incl"):
        keep = keep0 & (~sp if ver == "excl" else True)
        dists = {"a_full_latent": cat(f"B1|{ver}|full"), "b_probe_coords_crossfit": cat(f"B1_trained|{ver}|pred_dist_crossfit"),
                 "b_probe_coords_fullfit": cat(f"B1|{ver}|pred_dist_fullfit"), "c_probe_subspace": cat(f"B1|{ver}|subspace")}
        for k in (1, 2, 4, 8, 16, 64):
            dists[f"pca{k}"] = cat(f"B1|{ver}|pca{k}")
        for rep in REPS[1:]:
            dists[f"control_b_coords_crossfit_{rep}"] = cat(f"{rep}|{ver}|pred_dist_crossfit")
        p4[ver] = {}
        for name, dv in dists.items():
            c = ab.curves(dv, d_maze, d_xy, keep)
            sp_bands = {}
            for lo, hi in BANDS:
                msk = keep & (d_maze >= lo) & (d_maze <= hi)
                sp_bands[f"{lo}-{hi}"] = float(spearmanr(dv[msk], d_maze[msk]).correlation) if msk.sum() > 10 else None
            p4[ver][name] = {"radius": ab.radius(c), "maze_curve_q25_q50_q75_n": c["maze"], "spearman_by_band": sp_bands,
                             "spearman_all_d_le15": float(spearmanr(dv[keep & (d_maze <= 15)], d_maze[keep & (d_maze <= 15)]).correlation)}
        share = cat(f"B1|{ver}|subspace_share")[keep]
        p4[ver]["P5_subspace_share_of_sq_distance"] = {"median_pct": float(100 * np.median(share)), "mean_pct": float(100 * share.mean()),
                                                       "p10_pct": float(100 * np.percentile(share, 10)), "p90_pct": float(100 * np.percentile(share, 90)),
                                                       "random_2d_expectation_pct": 100 * 2 / 29584}
        p4[ver]["P5_pca_captures_probe_subspace_median"] = {k: float(np.median([R[m]["reps"]["B1_trained"][ver][f"pca{k}_captures_Q"] for m in maps])) for k in (1, 2, 4, 8, 16, 64)}
        p4[ver]["P5_pca_variance_share_median"] = {k: float(np.median([R[m]["reps"]["B1_trained"][ver][f"pca{k}_var_share"] for m in maps])) for k in (1, 2, 4, 8, 16, 64)}
    out["P4_P5"] = p4
    out["n_pairs"] = {"excl": int((keep0 & ~sp).sum()), "incl": int(keep0.sum())}
    # P2
    mp = os.path.join(HERE, "results_ps_mlp.json")
    if os.path.exists(mp):
        M = json.load(open(mp))["maps"]
        mm = sorted(M, key=int)
        p2 = {}
        for key in M[mm[0]]:
            rep, ver = key.split("|")
            mlp = np.array([M[m][key]["r2_fold0"][2] for m in mm])
            rid = np.array([R[m]["reps"][rep][ver]["r2_fold0"][2] for m in mm if m in R])
            p2[key] = {"mlp_r2_fold0_median": float(np.median(mlp)), "ridge_r2_fold0_median": float(np.median(rid)),
                       "mlp_minus_ridge_median": float(np.median(mlp[:len(rid)] - rid)), "n_maps": len(mm),
                       "mlp_err_cells_median": float(np.median([M[m][key]["err_cells_median"] for m in mm]))}
        out["P2_mlp"] = p2
    # verdict (prereg)
    rb = p4["excl"]["b_probe_coords_crossfit"]["radius"].get("R_maze")
    rb_num = rb if isinstance(rb, (int, float)) else None
    cond_r2 = b1 >= 0.90
    cond_rad_B = rb_num is not None and rb_num >= 6
    cond_ctrl = (b2 / b1 < 0.90) and (b4 / b1 < 0.90)
    cond_A = rb_num is not None and rb_num <= 3
    invalid_c1 = True  # C1 judged (나) fixed full view before measurement
    invalid_p3 = (b2 / b1 >= 0.90) or (b4 / b1 >= 0.90)
    verdict = "B" if (cond_r2 and cond_rad_B and cond_ctrl) else ("A" if cond_A else "구분 실패")
    out["verdict"] = {"R2_B1_excl_median": b1, "R_maze_b_crossfit_excl": rb, "cond_R2_ge_0.90": cond_r2, "cond_radius_ge_6": cond_rad_B,
                      "cond_controls_below_0.90x": cond_ctrl, "cond_A_radius_le_3": cond_A, "verdict": verdict,
                      "interpretation_invalid_C1": invalid_c1, "interpretation_invalid_P3": invalid_p3}
    json.dump(out, open(os.path.join(HERE, "results_ps_summary.json"), "w"), indent=1)
    print(json.dumps(out["verdict"], indent=1))


if __name__ == "__main__":
    main()
