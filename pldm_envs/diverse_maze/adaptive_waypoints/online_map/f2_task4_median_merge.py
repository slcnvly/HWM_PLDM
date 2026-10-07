# Follow-up 2, task 4: decide merges with the element-wise median latent of the last
# k frames (k = 1, 3, 5, 7; causal, within episode) instead of a single frame.
# Same growth/evaluation as stage 3; thresholds conservative and medium from stage 2.
# Also: how many spike frames survive the median (smoothed-series spike rate).
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stage3_grow_map as s3  # noqa: E402

s2 = s3.s2
KS = (1, 3, 5, 7)


def causal_median(obs, ep_of, k):
    if k == 1:
        return obs
    out = torch.empty_like(obs)
    start = 0
    for i in range(len(obs)):
        if i == 0 or ep_of[i] != ep_of[i - 1]:
            start = i
        lo = max(start, i - k + 1)
        out[i] = obs[lo:i + 1].median(dim=0).values
    return out


def spike_rate(z, ep_of):
    d = (z[1:] - z[:-1]).pow(2).mean(1).numpy()
    same = ep_of[1:] == ep_of[:-1]
    d = np.where(same, d, 0.0)
    prev, nxt = np.r_[0.0, d], np.r_[d, 0.0]
    return float(((prev > s3.SPIKE) & (nxt > s3.SPIKE)).mean())


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    st2 = json.load(open(os.path.join(HERE, "results_stage2.json")))
    taus = {n: st2["spaces"]["obs_full"]["best_at_wall_le"][k]["tau"] for n, k in (("conservative", "0.02"), ("medium", "0.05"))}
    model = s2.get_model()
    data = torch.load(os.path.join(s2.DATA, "main", "data.p"), weights_only=False)
    maps = torch.load(os.path.join(s2.DATA, "main", "train_maps.pt"), weights_only=False)
    images = np.load(os.path.join(s2.DATA, "main", "images.npy"), mmap_mode="r")
    starts = s2.episode_starts(data)
    by_map = {}
    for e, ep in enumerate(data):
        by_map.setdefault(int(ep["map_idx"]), []).append(e)
    out = {"taus": taus, "maps": {}}
    for m, eps in sorted(by_map.items()):
        _, dist_maze = s2.bfs_all(maps[m].split("\\"))
        obs, _, meta = s2.encode_map(model, data, images, starts, eps)
        cells = [tuple(int(v) for v in c) for c in s3.obs_to_ij(meta[:, 2:4].astype(np.float64))]
        ep_of = meta[:, 0].astype(int)
        out["maps"][m] = {}
        for k in KS:
            zk = causal_median(obs, ep_of, k)
            out["maps"][m][f"k{k}_spike_rate"] = spike_rate(zk, ep_of)
            for name, tau in taus.items():
                r = s3.grow(zk, cells, ep_of, dist_maze, tau, False, np.random.default_rng(m))
                r.pop("curve_full", None)
                out["maps"][m][f"k{k}_{name}"] = r
            del zk
        print(f"map {m}: " + " ".join(f"k{k}: sp {out['maps'][m][f'k{k}_conservative']['curve'][-1].get('spearman_map_path', float('nan')):.3f} fake {out['maps'][m][f'k{k}_conservative']['fake_edges']}/{out['maps'][m][f'k{k}_conservative']['n_edges']} spikes {out['maps'][m][f'k{k}_spike_rate']:.3f}" for k in KS), flush=True)
        json.dump(out, open(os.path.join(HERE, "results_f2_task4_median_merge.json"), "w"))
        del obs
    # summary
    summ = {}
    for k in KS:
        summ[f"k{k}"] = {"spike_rate_mean": float(np.mean([out["maps"][m][f"k{k}_spike_rate"] for m in out["maps"]]))}
        for name in taus:
            rows = [out["maps"][m][f"k{k}_{name}"] for m in out["maps"]]
            last = [x["curve"][-1] for x in rows if x["curve"]]
            peak = [max(c.get("spearman_map_path", -1) for c in x["curve"]) for x in rows if x["curve"]]
            summ[f"k{k}"][name] = {
                "final_spearman_map_path": float(np.nanmean([c.get("spearman_map_path", np.nan) for c in last])),
                "peak_spearman_map_path": float(np.mean(peak)),
                "final_spearman_fake_removed": float(np.nanmean([c.get("spearman_map_path_fake_edges_removed") or np.nan for c in last])),
                "disconnected_after_removal": float(np.nanmean([c.get("share_pairs_disconnected_after_removal", np.nan) for c in last])),
                "final_spearman_latent_straight": float(np.nanmean([c.get("spearman_latent_straight", np.nan) for c in last])),
                "fake_edge_share": float(np.mean([x["fake_edges"] / max(x["n_edges"], 1) for x in rows])),
                "fake_merge_share": float(np.mean([x["fake_merge_share"] for x in rows])),
                "nodes_mean": float(np.mean([x["n_nodes"] for x in rows])),
            }
    out["summary"] = summ
    json.dump(out, open(os.path.join(HERE, "results_f2_task4_median_merge.json"), "w"))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
