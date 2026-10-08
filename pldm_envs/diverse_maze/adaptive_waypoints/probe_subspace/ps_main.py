# PROBE_SUBSPACE P1, P3, P4, P5 (PROBE_SUBSPACE_PREREG.md). Map by map (RAM), one representation
# at a time. Per-map dual ridge (episode-grouped 5-fold cross-fitting, alpha grid), full-map fit
# for the probe subspace (B1), exact PCA via centred Gram eigendecomposition (B1), and pair
# distances on displacement_pairs/eval_pairs_seed1.json. Probes are diagnostic instruments only.
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
AW = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(AW, "latent_law"))
import ab_saturation as ab  # noqa: E402

REPS = ["B1_trained", "B2_random_seed0", "B2_random_seed1", "B2_random_seed2", "B4_pixels"]
S_GRID = [10.0 ** k for k in range(-6, 3)]
PCA_K = [1, 2, 4, 8, 16, 64]
VERSIONS = ("excl", "incl")
OUT = os.path.join(HERE, "results_ps_main.json")
PAIRS_OUT = os.path.join(HERE, "ps_pairs.npz")


def center_kernel(K, tr, te):
    Ktt = K[np.ix_(tr, tr)]
    kbar = Ktt.mean(0)
    m = kbar.mean()
    Kc_tt = Ktt - kbar[None, :] - kbar[:, None] + m
    Kst = K[np.ix_(te, tr)]
    Kc_st = Kst - kbar[None, :] - Kst.mean(1, keepdims=True) + m
    return Kc_tt, Kc_st


def ridge_preds(K, Y, tr, te):
    """Dual ridge for every s in S_GRID. Returns {s: preds (len(te), 2)} and the dual coefs."""
    Kc_tt, Kc_st = center_kernel(K, tr, te)
    ymu = Y[tr].mean(0)
    lam, V = torch.linalg.eigh(torch.from_numpy(Kc_tt))
    lam, V = lam.numpy().clip(min=0), V.numpy()
    VtY = V.T @ (Y[tr] - ymu)
    scale = np.trace(Kc_tt) / len(tr)
    out, coefs = {}, {}
    for s in S_GRID:
        a = V @ (VtY / (lam + s * scale)[:, None])
        out[s] = Kc_st @ a + ymu
        coefs[s] = a
    return out, coefs, ymu


def r2(pred, Y):
    r = [1 - ((pred[:, d] - Y[:, d]) ** 2).sum() / ((Y[:, d] - Y[:, d].mean()) ** 2).sum() for d in range(2)]
    return [float(r[0]), float(r[1]), float(np.mean(r))]


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 10)))
    data = torch.load(os.path.join(ab.DATA, "data.p"), weights_only=False)
    maps = torch.load(os.path.join(ab.DATA, "train_maps.pt"), weights_only=False)
    images = np.load(os.path.join(ab.DATA, "images.npy"), mmap_mode="r")
    starts = np.concatenate([[0], np.cumsum([len(e["observations"]) for e in data])])
    spike_all = np.load(os.path.join(AW, "displacement_pairs", "dp_latents.npz"))["spike"]
    ev = json.load(open(os.path.join(AW, "displacement_pairs", "eval_pairs_seed1.json")))["frames_episode_t_by_map"]
    encs = {"B1_trained": ab.load_model().level1.backbone}
    for s in range(3):
        encs[f"B2_random_seed{s}"] = ab.random_backbone(s)
    encs["B4_pixels"] = None
    by_map = {}
    for e, ep in enumerate(data):
        by_map.setdefault(int(ep["map_idx"]), []).append(e)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"maps": {}}
    pairs = dict(np.load(PAIRS_OUT)) if os.path.exists(PAIRS_OUT) else {}
    for m, eps in sorted(by_map.items()):
        if str(m) in res["maps"]:
            continue
        L = maps[m].split("\\")
        dist = ab.bfs_all(L)
        T = len(data[eps[0]]["observations"])
        fe = np.array([(e, t) for e in eps for t in range(T)])
        Y = np.concatenate([data[e]["observations"][:, :2] for e in eps]).astype(np.float64) / ab.GRID_SIZE
        spike = np.concatenate([spike_all[e] for e in eps])
        perm = np.random.default_rng(0).permutation(len(eps))
        fold_of_ep = {eps[k]: int(np.nonzero(perm == k)[0][0] // 10) for k in range(len(eps))}  # 10 episodes per fold
        fold = np.array([fold_of_ep[e] for e, _ in fe])
        # eval frames and pairs (same construction as dp_eval_set.run / A2)
        pick = [tuple(x) for x in ev[str(m)]]
        row = {(e, t): k for k, (e, t) in enumerate(map(tuple, fe))}
        eidx = np.array([row[p] for p in pick])
        xy = np.array([data[e]["observations"][t, :2] for e, t in pick], dtype=np.float64)
        ij = [tuple(int(v) for v in c) for c in ab.obs_to_ij(xy)]
        ep_id = np.array([e for e, _ in pick]); t_id = np.array([t for _, t in pick])
        iu = np.triu_indices(len(pick), 1)
        d_maze = np.array([dist[ij[a]].get(ij[b], -1) for a, b in zip(*iu)])
        d_xy = np.linalg.norm(xy[iu[0]] - xy[iu[1]], axis=1) / ab.GRID_SIZE
        keep0 = ~((d_maze == 0) & (ep_id[iu[0]] == ep_id[iu[1]]) & (np.abs(t_id[iu[0]] - t_id[iu[1]]) < 20)) & (d_maze >= 0)
        sp_pair = spike[eidx][iu[0]] | spike[eidx][iu[1]]
        mp = {"pairs_d_maze": d_maze, "pairs_d_xy": d_xy, "pairs_keep0": keep0, "pairs_spike": sp_pair}
        mres = {"n_frames": int(len(fe)), "spike_rate": float(spike.mean()), "reps": {}}
        for rep in REPS:
            X = torch.cat([ab.features(rep, np.array(images[starts[e]:starts[e + 1]]), encs[rep]) for e in eps]).numpy()
            D = X.shape[1]
            K = (torch.from_numpy(X).double() @ torch.from_numpy(X).double().T).numpy()
            rres = {}
            for ver in VERSIONS:
                mask = ~spike if ver == "excl" else np.ones(len(fe), bool)
                cv = {s: np.full((len(fe), 2), np.nan) for s in S_GRID}
                for f in range(5):
                    tr = np.nonzero(mask & (fold != f))[0]; te = np.nonzero(fold == f)[0]  # predict all frames of the held-out fold
                    p, _, _ = ridge_preds(K, Y, tr, te)
                    for s in S_GRID:
                        cv[s][te] = p[s]
                scores = {s: r2(cv[s][mask], Y[mask]) for s in S_GRID}
                best = max(S_GRID, key=lambda s: scores[s][2])
                err = np.linalg.norm(cv[best][mask] - Y[mask], axis=1)
                fold0 = mask & (fold == 0)
                rres[ver] = {"s_best": best, "r2_xy_mean": scores[best], "r2_by_s": {str(s): scores[s][2] for s in S_GRID},
                             "err_cells_median": float(np.median(err)), "err_cells_p90": float(np.percentile(err, 90)),
                             "r2_fold0": r2(cv[best][fold0], Y[fold0])}
                pe = cv[best][eidx]
                mp[f"{rep}|{ver}|pred_dist_crossfit"] = np.linalg.norm(pe[iu[0]] - pe[iu[1]], axis=1)
                if rep == "B1_trained":
                    tr = np.nonzero(mask)[0]
                    p, coefs, ymu = ridge_preds(K, Y, tr, eidx)
                    pe_full = p[best]
                    mp[f"B1|{ver}|pred_dist_fullfit"] = np.linalg.norm(pe_full[iu[0]] - pe_full[iu[1]], axis=1)
                    mu = X[tr].mean(0)
                    W = (X[tr] - mu).T @ coefs[best]  # (D, 2)
                    Q, _ = np.linalg.qr(W)
                    Ze = X[eidx]
                    proj = (Ze - mu) @ Q
                    full_sq = ab.pair_mse(torch.from_numpy(Ze)).numpy()[iu] * D
                    proj_sq = ((proj[iu[0]] - proj[iu[1]]) ** 2).sum(1)
                    mp[f"B1|{ver}|full"] = full_sq / D
                    mp[f"B1|{ver}|subspace"] = proj_sq / D
                    mp[f"B1|{ver}|subspace_share"] = proj_sq / np.maximum(full_sq, 1e-12)
                    rres[ver]["probe_W_fro"] = float(np.linalg.norm(W))
                    # PCA (label-free): centred Gram over the version's frames
                    Kc, Kce = center_kernel(K, tr, eidx)
                    lam, V = torch.linalg.eigh(torch.from_numpy(Kc))
                    lam, V = lam.numpy()[::-1], V.numpy()[:, ::-1]
                    kmax = max(PCA_K)
                    comp = V[:, :kmax] / np.sqrt(lam[:kmax])[None, :]  # scores = Kc @ comp
                    sc = Kce @ comp  # (400, kmax) PC scores of eval frames
                    Xc_Q = (X[tr] - mu) @ Q  # (n, 2)
                    for k in PCA_K:
                        mp[f"B1|{ver}|pca{k}"] = ((sc[iu[0], :k] - sc[iu[1], :k]) ** 2).sum(1) / D
                        # share of Q captured by top-k PCs: ||C_k^T Q||_F^2 / 2, C_k = Xc^T V_k / sqrt(lam_k)
                        CtQ = comp[:, :k].T @ Xc_Q
                        rres[ver][f"pca{k}_captures_Q"] = float((CtQ ** 2).sum() / 2)
                        rres[ver][f"pca{k}_var_share"] = float(lam[:k].sum() / lam.clip(min=0).sum())
            mres["reps"][rep] = rres
            print(f"map {m} {rep}: R2 excl {rres['excl']['r2_xy_mean'][2]:.3f} (s {rres['excl']['s_best']:g}) incl {rres['incl']['r2_xy_mean'][2]:.3f}", flush=True)
            del X, K
        res["maps"][str(m)] = mres
        for k, v in mp.items():
            pairs[f"{m}|{k}"] = v
        json.dump(res, open(OUT, "w"), indent=1)
        np.savez_compressed(PAIRS_OUT, **pairs)


if __name__ == "__main__":
    main()
