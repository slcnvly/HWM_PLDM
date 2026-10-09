# COST_GEOMETRY analysis: saturation displacement per (shape, s, rep) with ab_saturation.curves/radius
# (xy part, 90% rise) in two unit conventions, regression vs s + r_eff, M2-M4, preregistered verdict, figures.
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import cg_common as cg  # noqa: E402

ab = cg.ab
HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = [3, 6, 9, 18, 36, 72]
SHAPES = ["disc", "square", "arm"]
REAL = {"B4_pixels": 0.45, "B2_random": 0.52, "B1_trained": 1.35, "B3_dinov2_cls": 2.25, "B3_dinov2_patch8x8": 2.35}
TASK_PX = 49.0


def sat(ds, dists, unit):
    """ab.curves / ab.radius xy branch with displacement expressed in `unit` px; returns px (or str/None)."""
    d_xy = ds / unit
    c = ab.curves(dists, -np.ones(len(ds), int), d_xy, np.ones(len(ds), bool))
    r = ab.radius(c).get("R_xy")
    return (r * unit if isinstance(r, (int, float)) else r), c


def main():
    Z = np.load(os.path.join(HERE, "cg_m1_sweeps.npz"))
    rf = json.load(open(os.path.join(HERE, "results_cg_c3_rf.json")))
    reff = {n: rf[n]["r_eff_px_median"] for n in rf if n != "analytic"}
    nstart = len(Z["angles"])
    res = {"r_eff_px": reff, "sat": {}, "curves_px": {}}
    jobs = [(sh, s) for sh in SHAPES for s in SIZES] + [("disc", 4)]
    for sh, s in jobs:
        for rep in cg.REPS:
            ds = np.concatenate([Z[f"{sh}|{s}|{k}|d"] for k in range(nstart)]).astype(float)
            dv = np.concatenate([Z[f"{sh}|{s}|{k}|{rep}"] for k in range(nstart)]).astype(float)
            dmax = cg.W if sh != "arm" else min(2 * s, cg.W)
            r_sweep, _ = sat(ds, dv, dmax / 4)
            r_cell, _ = sat(ds, dv, cg.PX_PER_CELL)
            key = f"{sh}|{s}|{rep}"
            res["sat"][key] = {"shape": sh, "s": s, "rep": rep, "R_px_sweep_unit": r_sweep, "R_px_cell_unit": r_cell,
                               "R_cells_cell_unit": (r_cell / cg.PX_PER_CELL if isinstance(r_cell, float) else r_cell), "dmax": dmax}
            ud = np.unique(ds)
            res["curves_px"][key] = {"d": ud.tolist(), "q25": [float(np.percentile(dv[ds == x], 25)) for x in ud],
                                     "q50": [float(np.median(dv[ds == x])) for x in ud], "q75": [float(np.percentile(dv[ds == x], 75)) for x in ud]}
    # regression (prereg: definition (i), x = s + r_eff; DINO CLS excluded; disc s=4 is an M2 extra, excluded)
    pts = [(v["shape"], v["s"], v["rep"], v["s"] + reff[v["rep"]], v["s"] + 2 * reff[v["rep"]], v["R_px_sweep_unit"])
           for v in res["sat"].values() if v["s"] in SIZES and reff.get(v["rep"]) is not None and isinstance(v["R_px_sweep_unit"], float)]
    excluded = [k for k, v in res["sat"].items() if v["s"] in SIZES and (reff.get(v["rep"]) is None or not isinstance(v["R_px_sweep_unit"], float))]

    def fit(xs, ys):
        a, b = np.polyfit(xs, ys, 1)
        pred = a * np.asarray(xs) + b
        r2 = 1 - ((np.asarray(ys) - pred) ** 2).sum() / ((np.asarray(ys) - np.mean(ys)) ** 2).sum()
        return {"slope": float(a), "intercept": float(b), "r2": float(r2), "n": len(xs)}
    X = [p[3] for p in pts]; X2 = [p[4] for p in pts]; Y = [p[5] for p in pts]
    reg = {"all": fit(X, Y), "all_x_s_plus_2reff": fit(X2, Y), "excluded_points": excluded}
    for sh in SHAPES:
        q = [p for p in pts if p[0] == sh]
        reg[f"shape_{sh}"] = fit([p[3] for p in q], [p[5] for p in q])
    for rep in cg.REPS:
        q = [p for p in pts if p[2] == rep]
        if len(q) >= 3:
            reg[f"rep_{rep}"] = fit([p[3] for p in q], [p[5] for p in q])
    res["regression"] = reg
    # arm vs disc ratio
    ratios = []
    for s in SIZES:
        for rep in cg.REPS:
            a, d = res["sat"][f"arm|{s}|{rep}"]["R_px_sweep_unit"], res["sat"][f"disc|{s}|{rep}"]["R_px_sweep_unit"]
            if isinstance(a, float) and isinstance(d, float) and d > 0:
                ratios.append((s, rep, a / d))
    res["arm_over_disc"] = {"median": float(np.median([r[2] for r in ratios])), "by_s_median": {s: float(np.median([r[2] for r in ratios if r[0] == s])) for s in SIZES if any(r[0] == s for r in ratios)},
                            "n": len(ratios)}
    # M2: disc s=4 vs LATENT_LAW B real radii (cells)
    m2 = {}
    for rep in cg.REPS:
        real = REAL.get(rep, REAL["B2_random"] if rep.startswith("B2") else None)
        syn = res["sat"][f"disc|4|{rep}"]["R_cells_cell_unit"]
        pred = None if reff.get(rep) is None else (reg["all"]["slope"] * (4 + reff[rep]) + reg["all"]["intercept"]) / cg.PX_PER_CELL
        m2[rep] = {"real_R_xy_cells": real, "synthetic_disc4_R_xy_cells": syn, "regression_prediction_cells": pred,
                   "synthetic_minus_real": (syn - real if isinstance(syn, float) and real is not None else None),
                   "prediction_minus_real": (pred - real if pred is not None and real is not None else None)}
    res["M2"] = m2
    # M3 coverage ratio
    res["M3_coverage"] = {k: (v["R_px_sweep_unit"] / TASK_PX if isinstance(v["R_px_sweep_unit"], float) else None) for k, v in res["sat"].items()}
    res["M3_maze_reference"] = 1.35 / 13
    # M4
    d_reff = reff["B1_trained"] - np.mean([reff[f"B2_random_seed{s}"] for s in range(3)])
    explained = reg["all"]["slope"] * d_reff / cg.PX_PER_CELL
    syn_b1 = m2["B1_trained"]["synthetic_disc4_R_xy_cells"]
    syn_b2 = [m2[f"B2_random_seed{s}"]["synthetic_disc4_R_xy_cells"] for s in range(3)]
    res["M4"] = {"delta_r_eff_px": float(d_reff), "explained_cells": float(explained), "real_delta_cells": 1.35 - 0.52,
                 "residual_cells": float(1.35 - 0.52 - explained),
                 "synthetic_disc4_B1_minus_B2mean_cells": (syn_b1 - float(np.mean(syn_b2)) if isinstance(syn_b1, float) and all(isinstance(x, float) for x in syn_b2) else None),
                 "dino_patch_prediction_cells": m2["B3_dinov2_patch8x8"]["regression_prediction_cells"]}
    # verdict
    r = reg["all"]
    slopes = [reg[f"shape_{sh}"]["slope"] for sh in SHAPES]
    shape_split = (min(slopes) <= 0) or (max(slopes) / min(slopes) >= 2)
    support = r["r2"] >= 0.80 and 0.7 <= r["slope"] <= 1.3
    reject = r["r2"] < 0.50 or shape_split
    res["verdict"] = {"geometry": "지지" if support and not reject else ("기각" if reject else "구분 실패"),
                      "r2": r["r2"], "slope": r["slope"], "shape_slopes": dict(zip(SHAPES, slopes)), "shape_split_2x": bool(shape_split),
                      "learning_residual_gt_0.3": bool(res["M4"]["residual_cells"] > 0.3),
                      "arm_less_severe_1.5x": bool(res["arm_over_disc"]["median"] >= 1.5), "arm_over_disc_median": res["arm_over_disc"]["median"]}
    json.dump(res, open(os.path.join(HERE, "results_cg_analysis.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("regression", "arm_over_disc", "M2", "M4", "verdict")}, indent=1, default=str))
    # figures
    fig, axes = plt.subplots(3, 4, figsize=(17, 11))
    for i, sh in enumerate(SHAPES):
        for j, rep in enumerate(["B1_trained", "B2_random_seed0", "B4_pixels", "B3_dinov2_patch8x8"]):
            ax = axes[i, j]
            for s in SIZES:
                c = res["curves_px"][f"{sh}|{s}|{rep}"]
                q50 = np.array(c["q50"]); top = np.median(q50[-max(1, len(q50) // 4):])
                ax.plot(c["d"], q50 / top if top > 0 else q50, label=f"s={s}")
                R = res["sat"][f"{sh}|{s}|{rep}"]["R_px_sweep_unit"]
                if isinstance(R, float):
                    ax.axvline(R, ls=":", lw=0.7, c=ax.lines[-1].get_color())
            ax.set_title(f"{sh} / {rep}", fontsize=9); ax.set_xlabel("displacement d (px)"); ax.set_ylabel("median dist / plateau")
            ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "m1_curves.png"), dpi=90)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    mk = {"disc": "o", "square": "s", "arm": "^"}
    for p in pts:
        ax[0].scatter(p[3], p[5], marker=mk[p[0]], c=f"C{cg.REPS.index(p[2])}", s=25)
    xs = np.linspace(0, max(X), 10)
    ax[0].plot(xs, r["slope"] * xs + r["intercept"], "k--", lw=1, label=f"fit slope {r['slope']:.2f}, R2 {r['r2']:.2f}")
    ax[0].plot(xs, xs, c="0.6", lw=0.8, label="y = x")
    for k, rep in enumerate(cg.REPS):
        ax[0].scatter([], [], c=f"C{k}", label=rep, s=15)
    for sh in SHAPES:
        ax[0].scatter([], [], marker=mk[sh], c="k", label=sh, s=15)
    ax[0].set_xlabel("s + r_eff (px)"); ax[0].set_ylabel("saturation displacement (px, sweep unit)"); ax[0].legend(fontsize=6)
    for sh in SHAPES:
        for rep in ["B1_trained", "B2_random_seed0", "B4_pixels", "B3_dinov2_cls"]:
            ys = [res["M3_coverage"][f"{sh}|{s}|{rep}"] for s in SIZES]
            ax[1].plot([s for s, y in zip(SIZES, ys) if y is not None], [y for y in ys if y is not None], marker=mk[sh], c=f"C{cg.REPS.index(rep)}", lw=0.8)
    ax[1].axhline(res["M3_maze_reference"], c="r", ls=":", label="maze real 1.35/13")
    ax[1].axhline(1.0, c="k", lw=0.5)
    ax[1].set_xscale("log"); ax[1].set_xlabel("content size s (px)"); ax[1].set_ylabel("coverage = saturation px / 49 px"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "m1_regression_m3_coverage.png"), dpi=100)


if __name__ == "__main__":
    main()
