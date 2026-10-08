# POST-HOC (not preregistered): duration-matched comparison for M3. For displacement-hop chains of
# K hops, record elapsed env steps and BFS displacement; compare with all fixed-gap pairs of the same
# elapsed time (time-based pairs). Answers: at equal elapsed time, do displacement-hop chains go
# farther than time-based pairs? Uses the same helpers as dp_measure.py.
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dp_measure as dm  # noqa: E402

BINS = [(1, 10), (11, 20), (21, 40), (41, 60), (61, 100)]


def run(trajs, maps_of, Dm, spike=None):
    out = {}
    for ver in (("incl_spikes", "excl_spikes") if spike is not None else ("incl_spikes",)):
        rows = []  # (K, duration, bfs)
        tb = {b: [] for b in BINS}
        for e, pos in trajs.items():
            _, B, _ = dm.episode_mats(pos, Dm[maps_of[e]])
            bnd, _ = dm.disp_hops(B)
            T = len(pos)
            for K in dm.KS:
                for a, b in dm.chains(bnd, K):
                    if ver == "excl_spikes" and (spike[e][a] or spike[e][b]):
                        continue
                    rows.append((K, b - a, B[a, b]))
            for lo, hi in BINS:
                for g in range(lo, hi + 1):
                    a = np.arange(T - g); bb = a + g
                    if ver == "excl_spikes":
                        k = ~(spike[e][a] | spike[e][bb]); a, bb = a[k], bb[k]
                    tb[(lo, hi)].append(np.stack([np.full(len(a), g), B[a, bb]], 1))
        rows = np.array(rows, float)
        res = {"by_K": {}, "by_duration_bin": {}}
        for K in dm.KS:
            r = rows[rows[:, 0] == K]
            if len(r):
                res["by_K"][K] = {"duration_mean": float(r[:, 1].mean()), "duration_median": float(np.median(r[:, 1])),
                                  "bfs_mean": float(r[:, 2].mean()), "n": int(len(r))}
        for lo, hi in BINS:
            r = rows[(rows[:, 1] >= lo) & (rows[:, 1] <= hi)]
            t = np.concatenate(tb[(lo, hi)])
            # time-based pairs reweighted to the displacement chains' duration histogram within the bin
            w_mean = None
            if len(r):
                durs, cnt = np.unique(r[:, 1].astype(int), return_counts=True)
                tm = {int(g): t[t[:, 0] == g, 1].mean() for g in durs}
                w_mean = float(np.sum([tm[int(g)] * c for g, c in zip(durs, cnt)]) / cnt.sum())
            res["by_duration_bin"][f"{lo}-{hi}"] = {
                "disp_chains_bfs_mean": float(r[:, 2].mean()) if len(r) else None, "disp_chains_n": int(len(r)),
                "time_pairs_bfs_mean_duration_matched": w_mean,
                "disp_chains_share_bfs_ge_0.7K": float(np.mean(r[:, 2] >= 0.7 * r[:, 0])) if len(r) else None}
        out[ver] = res
    return out


def main():
    data = torch.load(os.path.join(dm.ab.DATA, "data.p"), weights_only=False)
    maps = torch.load(os.path.join(dm.ab.DATA, "train_maps.pt"), weights_only=False)
    spike = np.load(os.path.join(HERE, "dp_latents.npz"))["spike"]
    map_of = {e: int(ep["map_idx"]) for e, ep in enumerate(data)}
    Dm = {m: dm.dist_array(maps[m].split("\\")) for m in set(map_of.values())}
    res = {"note": "POST-HOC, not preregistered",
           "r50": run({e: ep["observations"][:, :2] for e, ep in enumerate(data)}, map_of, Dm, spike)}
    sim = np.load(dm.SIM200)
    res["sim200"] = run({k: sim["a1"][k][:, :2] for k in range(len(sim["ep"]))}, {k: map_of[int(e)] for k, e in enumerate(sim["ep"])}, Dm)
    json.dump(res, open(os.path.join(HERE, "results_dp_posthoc_duration.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
