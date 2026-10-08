# DISPLACEMENT_PAIRS step 0, M1-M5 (DISPLACEMENT_PAIRS_PREREG.md). Positions, BFS, cell
# types from existing code paths (grid_utils.obs_to_ij / GRID_SIZE, ab_saturation.bfs_all,
# grid_utils.open_neighbor_count, followup/common.cell_type); latent distances and spike flags
# from dp_latents.npz (ab_saturation.features / pair_mse). Every metric: spike-incl and spike-excl.
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
AW = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(AW, "latent_law"))
sys.path.insert(0, os.path.join(AW, "followup"))
import ab_saturation as ab  # noqa: E402
from ab_saturation import GRID_SIZE, bfs_all, obs_to_ij  # noqa: E402
from common import cell_type, open_neighbor_count  # noqa: E402

GAPS = [14, 30, 60, 80, 100]
QS = [50, 90, 95, 99, 99.9]
DSTAR = [2, 3, 4, 5, 6]
KS = [1, 2, 4, 8, 16]
DEDUP = 10
SIM200 = os.path.abspath(os.path.join(AW, "..", "..", "..", "..", "experiments", "kaggle_sim_random_r50_200", "output", "sim_random_r50_200.npz"))
VERSIONS = ("incl_spikes", "excl_spikes")


def dist_array(layout):
    """bfs_all (dict of dicts) packed into D[i, j, k, l]; -1 = not reachable / wall."""
    H, W = len(layout), len(layout[0])
    D = -np.ones((H, W, H, W), np.int32)
    for (i, j), dd in bfs_all(layout).items():
        for (k, l), v in dd.items():
            D[i, j, k, l] = v
    return D


def episode_mats(pos, D):
    ij = obs_to_ij(pos.astype(np.float64))
    B = D[ij[:, 0][:, None], ij[:, 1][:, None], ij[:, 0][None, :], ij[:, 1][None, :]]
    X = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1) / GRID_SIZE
    return ij, B, X


def greedy_dedup(pairs):
    """pairs: (n,2) sorted lexicographically by (i, j). Keep a pair unless an already kept pair
    has |i-i'| <= 10 and |j-j'| <= 10."""
    kept = []
    for a, b in pairs:
        if kept:
            k = np.asarray(kept)
            if np.any((np.abs(k[:, 0] - a) <= DEDUP) & (np.abs(k[:, 1] - b) <= DEDUP)):
                continue
        kept.append((a, b))
    return np.asarray(kept, int).reshape(-1, 2)


def disp_hops(B):
    """Displacement hops: from s, cut at first t with B[s,t] >= 1. Returns boundaries, uncut tail length."""
    T = B.shape[0]
    bnd, s = [0], 0
    while True:
        nxt = np.nonzero(B[s, s + 1:] >= 1)[0]
        if len(nxt) == 0:
            return bnd, T - 1 - s
        s = s + 1 + int(nxt[0])
        bnd.append(s)


def time_hops(T, step=10):
    return list(range(0, T, step)) if (T - 1) % step == 0 else list(range(0, T - (T - 1) % step, step))


def chains(bnd, K):
    return [(bnd[h], bnd[h + K]) for h in range(len(bnd) - K)]


def qd(v, qs=QS):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return None
    return {**{f"p{q}": float(np.percentile(v, q)) for q in qs}, "mean": float(v.mean()), "n": int(len(v))}


def slope(ks, vals):
    pts = [(k, v) for k, v in zip(ks, vals) if v is not None and v > 0]
    if len(pts) < 2:
        return None
    return float(np.polyfit(np.log([p[0] for p in pts]), np.log([p[1] for p in pts]), 1)[0])


def auroc(pos, neg):
    s = np.r_[pos, neg]
    r = rankdata(s)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def chain_analysis(eps, bnds, spike_of, Bs, map_of, ks):
    """eps: list of episode ids, bnds: dict e -> boundaries. Returns per-version stats per K."""
    out = {}
    for ver in VERSIONS:
        out[ver] = {}
        for K in ks:
            tot, n_eps, stacked = [], set(), {}
            for e in eps:
                cs = chains(bnds[e], K)
                if ver == "excl_spikes" and spike_of is not None:
                    cs = [(a, b) for a, b in cs if not (spike_of[e][a] or spike_of[e][b])]
                if not cs:
                    continue
                n_eps.add(e)
                d = np.array([Bs[e][a, b] for a, b in cs], float)
                tot += d.tolist()
                ok = [c for c, v in zip(cs, d) if v >= 0.7 * K]
                stacked.setdefault(map_of[e], []).append((len(ok), len(greedy_dedup(np.array(sorted(ok)))) if ok else 0))
            tot = np.array(tot)
            per_map_eff = {m: int(sum(x[1] for x in v)) for m, v in stacked.items()}
            per_map_raw = {m: int(sum(x[0] for x in v)) for m, v in stacked.items()}
            out[ver][K] = {
                "total_bfs": qd(tot, [50, 90]) if len(tot) else None,
                "n_chains": int(len(tot)), "n_episodes": len(n_eps),
                "stacked_share": float((tot >= 0.7 * K).mean()) if len(tot) else None,
                "stacked_raw_total": int(sum(per_map_raw.values())),
                "stacked_effective_per_map": per_map_eff,
                "stacked_effective_per_map_median": float(np.median(list(per_map_eff.values()))) if per_map_eff else 0.0,
                "maps_below_100_effective": int(sum(v < 100 for v in per_map_eff.values())),
            }
        means = [out[ver][K]["total_bfs"]["mean"] if out[ver][K]["total_bfs"] else None for K in ks]
        meds = [out[ver][K]["total_bfs"]["p50"] if out[ver][K]["total_bfs"] else None for K in ks]
        out[ver]["slope_mean"] = slope(ks, means)
        out[ver]["slope_median"] = slope(ks, meds)
        sub = [k for k in ks if k <= 8]
        out[ver]["slope_mean_K1to8"] = slope(sub, [m for k, m in zip(ks, means) if k <= 8])
        out[ver]["slope_median_K1to8"] = slope(sub, [m for k, m in zip(ks, meds) if k <= 8])
    return out


def main():
    data = torch.load(os.path.join(ab.DATA, "data.p"), weights_only=False)
    maps = torch.load(os.path.join(ab.DATA, "train_maps.pt"), weights_only=False)
    L = np.load(os.path.join(HERE, "dp_latents.npz"))
    spike, pi, pj, lat = L["spike"], L["pair_i"], L["pair_j"], L["lat"]
    map_of = {e: int(ep["map_idx"]) for e, ep in enumerate(data)}
    layouts = {m: maps[m].split("\\") for m in set(map_of.values())}
    Dm = {m: dist_array(layouts[m]) for m in layouts}
    open_cells = {m: {(i, j) for i in range(len(l)) for j in range(len(l[0])) if l[i][j] == "O"} for m, l in layouts.items()}
    T = len(data[0]["observations"])
    res = {"C4": {"n_episodes": len(data), "frames_per_episode": sorted({len(e["observations"]) for e in data}),
                  "n_frames": int(sum(len(e["observations"]) for e in data)), "n_maps": len(layouts),
                  "episodes_per_map": sorted({sum(1 for v in map_of.values() if v == m) for m in layouts}),
                  "actions_per_episode": sorted({len(e["actions"]) for e in data})},
           "spike_rate": float(spike.mean())}
    Bs, IJ, Xs = {}, {}, {}
    for e, ep in enumerate(data):
        IJ[e], Bs[e], Xs[e] = episode_mats(ep["observations"][:, :2], Dm[map_of[e]])
        assert (Bs[e] >= 0).all(), f"unreachable cell pair in episode {e}"
    eps = list(range(len(data)))

    # ---------------- M1 ----------------
    m1 = {}
    for ver in VERSIONS:
        m1[ver] = {"by_gap": {}}
        r1_all, r05_all = [], []
        for g in GAPS:
            xs, bs = [], []
            for e in eps:
                a = np.arange(T - g); b = a + g
                if ver == "excl_spikes":
                    k = ~(spike[e][a] | spike[e][b]); a, b = a[k], b[k]
                xs.append(Xs[e][a, b]); bs.append(Bs[e][a, b])
            xs, bs = np.concatenate(xs), np.concatenate(bs).astype(float)
            r1 = bs[xs >= 1] / xs[xs >= 1]; r05 = bs[xs >= 0.5] / xs[xs >= 0.5]
            r1_all.append(r1); r05_all.append(r05)
            m1[ver]["by_gap"][g] = {"xy": qd(xs), "bfs": qd(bs), "ratio_bfs_over_xy_xy_ge1": qd(r1), "ratio_xy_ge05": qd(r05),
                                    "share_xy_ge1": float((xs >= 1).mean())}
        m1[ver]["ratio_pooled_xy_ge1"] = qd(np.concatenate(r1_all))
        m1[ver]["ratio_pooled_xy_ge05"] = qd(np.concatenate(r05_all))
    res["M1"] = m1
    print("M1 ratio p90 (excl, xy>=1):", m1["excl_spikes"]["ratio_pooled_xy_ge1"]["p90"], flush=True)

    # ---------------- M2 ----------------
    m2 = {}
    visited = {m: set() for m in layouts}
    for e in eps:
        visited[map_of[e]] |= {tuple(c) for c in IJ[e]}
    for ver in VERSIONS:
        m2[ver] = {}
        for d in DSTAR:
            raw_c, eff_c, cov_s, cov_e, cov_s_raw, cov_e_raw = {}, {}, {}, {}, {}, {}
            starts_eff = {m: set() for m in layouts}; ends_eff = {m: set() for m in layouts}
            starts_raw = {m: set() for m in layouts}; ends_raw = {m: set() for m in layouts}
            for m in layouts:
                raw_c[m] = 0; eff_c[m] = 0
            for e in eps:
                a, b = np.nonzero(np.triu(Bs[e] == d, 1))
                if ver == "excl_spikes":
                    k = ~(spike[e][a] | spike[e][b]); a, b = a[k], b[k]
                m = map_of[e]
                raw_c[m] += len(a)
                if len(a) == 0:
                    continue
                order = np.lexsort((b, a))
                kept = greedy_dedup(np.stack([a[order], b[order]], 1))
                eff_c[m] += len(kept)
                starts_raw[m] |= {tuple(IJ[e][x]) for x in a}; ends_raw[m] |= {tuple(IJ[e][x]) for x in b}
                starts_eff[m] |= {tuple(IJ[e][x]) for x in kept[:, 0]}; ends_eff[m] |= {tuple(IJ[e][x]) for x in kept[:, 1]}
            for m in layouts:
                n_open = len(open_cells[m])
                cov_s[m] = len(starts_eff[m]) / n_open; cov_e[m] = len(ends_eff[m]) / n_open
                cov_s_raw[m] = len(starts_raw[m]) / n_open; cov_e_raw[m] = len(ends_raw[m]) / n_open
            m2[ver][d] = {"raw_per_map": raw_c, "effective_per_map": eff_c,
                          "raw_median": float(np.median(list(raw_c.values()))), "effective_median": float(np.median(list(eff_c.values()))),
                          "effective_min": int(min(eff_c.values())),
                          "start_coverage_effective": cov_s, "end_coverage_effective": cov_e,
                          "start_coverage_effective_median": float(np.median(list(cov_s.values()))),
                          "end_coverage_effective_median": float(np.median(list(cov_e.values()))),
                          "start_coverage_raw_median": float(np.median(list(cov_s_raw.values()))),
                          "end_coverage_raw_median": float(np.median(list(cov_e_raw.values()))),
                          "maps_start_coverage_below_30pct": int(sum(v < 0.30 for v in cov_s.values()))}
    res["M2"] = m2
    res["M2_visited_cell_share_per_map"] = {m: len(visited[m]) / len(open_cells[m]) for m in layouts}
    print("M2 d*=4 excl: eff median", m2["excl_spikes"][4]["effective_median"], "start cov median", m2["excl_spikes"][4]["start_coverage_effective_median"], flush=True)

    # ---------------- M3 ----------------
    dbnd, tails = {}, []
    for e in eps:
        dbnd[e], tail = disp_hops(Bs[e])
        tails.append(tail)
    tbnd = {e: time_hops(T) for e in eps}
    hop_counts = np.array([len(dbnd[e]) - 1 for e in eps])
    hop_lens = np.concatenate([np.diff(dbnd[e]) for e in eps if len(dbnd[e]) > 1])
    tails = np.array(tails)
    res["M3"] = {
        "a": {"hops_per_episode": qd(hop_counts, [10, 50, 90]), "hop_count_hist": np.bincount(hop_counts).tolist(),
              "hop_length_steps": qd(hop_lens, [10, 25, 50, 75, 90, 99]),
              "share_episodes_with_uncut_tail": float((tails > 0).mean()), "uncut_tail_length": qd(tails[tails > 0], [50, 90]),
              "share_episodes_zero_hops": float((hop_counts == 0).mean()),
              "share_hops_bfs_exactly_1": float(np.mean([Bs[e][a, b] == 1 for e in eps for a, b in chains(dbnd[e], 1)]))},
        "b_displacement": chain_analysis(eps, dbnd, spike, Bs, map_of, KS),
        "c_time10_r50": chain_analysis(eps, tbnd, spike, Bs, map_of, [1, 2, 4, 8]),
    }
    print("M3 slopes disp", {v: res["M3"]["b_displacement"][v]["slope_mean"] for v in VERSIONS},
          "time", {v: res["M3"]["c_time10_r50"][v]["slope_mean"] for v in VERSIONS}, flush=True)
    if os.path.exists(SIM200):
        sim = np.load(SIM200)
        sB, sd, st, smap = {}, {}, {}, {}
        for k, e0 in enumerate(sim["ep"]):
            pos = sim["a1"][k][:, :2].astype(np.float64)
            m = map_of[int(e0)]
            _, sB[k], _ = episode_mats(pos, Dm[m])
            sd[k], _ = disp_hops(sB[k]); st[k] = time_hops(len(pos)); smap[k] = m
        ks = list(sB)
        res["M3"]["c_sim200_displacement"] = chain_analysis(ks, sd, None, sB, smap, KS)
        res["M3"]["c_sim200_time10"] = chain_analysis(ks, st, None, sB, smap, KS)
        res["M3"]["c_sim200_hops_per_episode"] = qd([len(sd[k]) - 1 for k in ks], [10, 50, 90])
        for key in ("c_sim200_displacement", "c_sim200_time10"):
            del res["M3"][key]["excl_spikes"]  # no images in the simulation -> no spike flags

    # ---------------- M4 ----------------
    m4 = {}
    Bp = np.stack([Bs[e][pi, pj] for e in eps]).astype(int)  # (eps, pairs)
    ep_idx = np.repeat(np.arange(len(eps))[:, None], len(pi), 1)
    sp_pair = spike[:, pi] | spike[:, pj]
    for ver in VERSIONS:
        keep = np.ones_like(sp_pair) if ver == "incl_spikes" else ~sp_pair
        s, b, ee = lat[keep], Bp[keep], ep_idx[keep]
        neg, pos = s[b <= 1], s[b >= 3]
        thr = float(np.percentile(neg, 95))
        groups = {"3-4": (b >= 3) & (b <= 4), "5-8": (b >= 5) & (b <= 8), ">=9": b >= 9}
        g = {}
        for name, msk in groups.items():
            v = s[msk]
            g[name] = {"n": int(len(v)), "n_episodes": int(len(np.unique(ee[msk]))),
                       "tpr_at_fpr5": float((v > thr).mean()) if len(v) else None,
                       "auroc_vs_folded": auroc(v, neg) if len(v) else None,
                       "latent_median": float(np.median(v)) if len(v) else None}
        # ROC curve
        ths = np.unique(np.percentile(np.r_[neg, pos], np.linspace(0, 100, 201)))
        roc = [[float(t), float((neg > t).mean()), float((pos > t).mean())] for t in ths]
        # episode bootstrap of TPR(>=9) - TPR(3-4)
        boot = None
        if g[">=9"]["n"] >= 20:
            rng = np.random.default_rng(0)
            uniq = np.unique(ee)
            by_e = {u: np.nonzero(ee == u)[0] for u in uniq}
            diffs = []
            for _ in range(1000):
                idx = np.concatenate([by_e[u] for u in rng.choice(uniq, len(uniq), replace=True)])
                sb, bb = s[idx], b[idx]
                nb = sb[bb <= 1]
                t = np.percentile(nb, 95)
                hi, lo = sb[bb >= 9], sb[(bb >= 3) & (bb <= 4)]
                if len(hi) and len(lo):
                    diffs.append((hi > t).mean() - (lo > t).mean())
            boot = {"diff_tpr_ge9_minus_3to4": float(g[">=9"]["tpr_at_fpr5"] - g["3-4"]["tpr_at_fpr5"]),
                    "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))], "n_boot": len(diffs)}
        m4[ver] = {"n_folded": int(len(neg)), "n_unfolded": int(len(pos)), "auroc": auroc(pos, neg), "threshold_fpr5": thr,
                   "tpr_at_fpr5": float((pos > thr).mean()), "folded_latent_median": float(np.median(neg)),
                   "unfolded_latent_median": float(np.median(pos)), "groups": g, "bootstrap": boot, "roc_thr_fpr_tpr": roc,
                   "bfs_hist_gap_ge60": np.bincount(b).tolist()}
    res["M4"] = m4
    print("M4 auroc", {v: m4[v]["auroc"] for v in VERSIONS}, flush=True)

    # ---------------- M5 ----------------
    m5 = {}
    acts = {e: np.asarray(data[e]["actions"], float) for e in eps}
    def autocorr(intervals):
        out = {}
        for lag in (1, 2, 4):
            X, Y = [], []
            for e, a, b in intervals:  # actions a .. b-1 belong to frames a..b
                A = acts[e][a:b]
                if len(A) > lag:
                    X.append(A[:-lag]); Y.append(A[lag:])
            X, Y = np.concatenate(X), np.concatenate(Y)
            cx = [float(np.corrcoef(X[:, d], Y[:, d])[0, 1]) for d in range(2)]
            out[lag] = {"x": cx[0], "y": cx[1], "mean": float(np.mean(cx)), "n_pairs": int(len(X))}
        return out
    def cell_stats(intervals):
        visits = {m: {} for m in layouts}
        for e, a, b in intervals:
            m = map_of[e]
            for c in IJ[e][a:b + 1]:
                c = tuple(int(v) for v in c)
                visits[m][c] = visits[m].get(c, 0) + 1
        return visits
    def type_share(visits):
        cnt = {"dead_end": 0, "straight": 0, "corner": 0, "junction": 0, "room_interior_4nb": 0}
        tot = 0
        for m, vv in visits.items():
            for (i, j), n in vv.items():
                cnt[cell_type(layouts[m], i, j)] += n
                if open_neighbor_count(layouts[m], i, j) == 4:
                    cnt["room_interior_4nb"] += n
                tot += n
        return {k: v / tot for k, v in cnt.items()}, tot
    all_int = [(e, 0, T - 1) for e in eps]
    v_all = cell_stats(all_int)
    ts_all, _ = type_share(v_all)
    for ver in VERSIONS:
        sel, uns = [], []
        for e in eps:
            for a, b in chains(dbnd[e], 4):
                if ver == "excl_spikes" and (spike[e][a] or spike[e][b]):
                    continue
                (sel if Bs[e][a, b] >= 0.7 * 4 else uns).append((e, a, b))
        v_sel, v_uns = cell_stats(sel), cell_stats(uns)
        ts_sel, n_sel = type_share(v_sel); ts_uns, n_uns = type_share(v_uns)
        def tvd(v1, v2, m):
            cells = set(v1[m]) | set(v2[m])
            s1, s2 = sum(v1[m].values()), sum(v2[m].values())
            if s1 == 0 or s2 == 0:
                return None
            return 0.5 * sum(abs(v1[m].get(c, 0) / s1 - v2[m].get(c, 0) / s2) for c in cells)
        tv_sel = [tvd(v_sel, v_all, m) for m in layouts]; tv_uns = [tvd(v_uns, v_all, m) for m in layouts]
        m5[ver] = {"n_selected_chains": len(sel), "n_unselected_chains": len(uns),
                   "action_autocorr_selected": autocorr(sel), "action_autocorr_unselected": autocorr(uns),
                   "cell_type_share_selected": ts_sel, "cell_type_share_unselected": ts_uns, "cell_type_share_all_data": ts_all,
                   "dropout_rate_selected_vs_all": {k: 1 - ts_sel[k] / ts_all[k] for k in ts_all if ts_all[k] > 0},
                   "tvd_selected_vs_all_median_over_maps": float(np.median([x for x in tv_sel if x is not None])),
                   "tvd_unselected_vs_all_median_over_maps": float(np.median([x for x in tv_uns if x is not None])),
                   "cells_covered_selected_share_of_open_median": float(np.median([len(v_sel[m]) / len(open_cells[m]) for m in layouts]))}
    m5["action_autocorr_all_data"] = autocorr(all_int)
    res["M5"] = m5
    json.dump(res, open(os.path.join(HERE, "results_dp_measure.json"), "w"), indent=1)
    print("saved")


if __name__ == "__main__":
    main()
