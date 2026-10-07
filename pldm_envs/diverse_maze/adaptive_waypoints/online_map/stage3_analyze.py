# Aggregate stage 3 over maps: Spearman(map path, maze) vs Spearman(latent straight, maze)
# along the growth, fake merges/edges and their damage, map size and shortest-path time.
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    r = json.load(open(os.path.join(HERE, "results_stage3.json")))
    variants = list(next(iter(r["maps"].values())).keys())
    out = {"taus": r["taus"], "n_maps": len(r["maps"]), "variants": {}}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for v in variants:
        rows = [m[v] for m in r["maps"].values()]
        curves = [x["curve"] for x in rows]
        steps = sorted({c["step"] for cv in curves for c in cv})
        agg = {}
        for key in ("spearman_map_path", "spearman_map_path_fake_edges_removed", "spearman_latent_straight"):
            agg[key] = [float(np.nanmean([c[key] for cv in curves for c in cv if c["step"] == s and c.get(key) is not None] or [np.nan])) for s in steps]
        final = {k: float(np.nanmean([cv[-1].get(k) if cv and cv[-1].get(k) is not None else np.nan for cv in curves]))
                 for k in ("spearman_map_path", "spearman_map_path_fake_edges_removed", "spearman_latent_straight", "share_pairs_disconnected_after_removal")}
        out["variants"][v] = {
            "final_mean_over_maps": final,
            "at_steps": {s: {k: agg[k][i] for k in agg} for i, s in enumerate(steps) if s in (500, 1000, 2500, 5050)},
            "nodes_mean": float(np.mean([x["n_nodes"] for x in rows])), "nodes_max": int(max(x["n_nodes"] for x in rows)),
            "edges_mean": float(np.mean([x["n_edges"] for x in rows])),
            "frames_used_mean": float(np.mean([x["n_frames_used"] for x in rows])),
            "fake_merge_share_mean": float(np.mean([x["fake_merge_share"] for x in rows])),
            "fake_merges_mean": float(np.mean([x["fake_merges"] for x in rows])),
            "fake_edges_mean": float(np.mean([x["fake_edges"] for x in rows])),
            "fake_edge_share_mean": float(np.mean([x["fake_edges"] / max(x["n_edges"], 1) for x in rows])),
            "all_pairs_shortest_path_seconds_max": float(max(x["all_pairs_sp_seconds"] for x in rows)),
            "grow_seconds_mean_per_5050_frames": float(np.mean([x["grow_seconds"] for x in rows])),
        }
        if v in ("medium", "medium_spike_filter"):
            ls = "-" if v == "medium" else "--"
            axes[0].plot(steps, agg["spearman_map_path"], ls, c="tab:blue", label=f"{v}: map path")
            axes[0].plot(steps, agg["spearman_map_path_fake_edges_removed"], ls, c="tab:green", label=f"{v}: map path, fake edges removed")
            axes[0].plot(steps, agg["spearman_latent_straight"], ls, c="tab:red", label=f"{v}: latent straight")
        axes[1].plot(steps, [np.mean([c["n_nodes"] for cv in curves for c in cv if c["step"] == s]) for s in steps], label=v)
    axes[0].axhline(0.442, c="0.5", ls=":", lw=0.8, label="C1 latent (within episode) 0.442")
    axes[0].set_xlabel("frames seen on this map"); axes[0].set_ylabel("Spearman with maze BFS"); axes[0].legend(fontsize=6)
    axes[1].set_xlabel("frames seen"); axes[1].set_ylabel("map nodes (mean over maps)"); axes[1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "stage3_curves.png"), dpi=110)
    json.dump(out, open(os.path.join(HERE, "results_stage3_summary.json"), "w"), indent=2)
    for v, d in out["variants"].items():
        f = d["final_mean_over_maps"]
        print(f"{v:28s} final map {f['spearman_map_path']:.3f} | fake removed {f['spearman_map_path_fake_edges_removed']:.3f} (disconn {f['share_pairs_disconnected_after_removal']:.2f}) | latent {f['spearman_latent_straight']:.3f} | nodes {d['nodes_mean']:.0f} (max {d['nodes_max']}) | fake merge {d['fake_merge_share_mean']:.3f} fake edges {d['fake_edges_mean']:.1f} ({d['fake_edge_share_mean']:.3f}) | APSP {d['all_pairs_shortest_path_seconds_max']*1000:.1f} ms | grow {d['grow_seconds_mean_per_5050_frames']:.1f}s")


if __name__ == "__main__":
    main()
