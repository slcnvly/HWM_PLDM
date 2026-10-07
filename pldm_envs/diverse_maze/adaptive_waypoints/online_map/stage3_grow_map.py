# Online-map feasibility, stage 3 (ONLINE_MAP_PREREG.md): grow a latent map along
# r50 episodes (one map at a time, episodes in order) and check whether shortest
# paths on the map track maze (BFS) distance.
#   node = founding frame's obs latent; a new frame joins the nearest node if its
#   obs-MSE <= tau, else founds a new node; consecutive frames' nodes are joined.
#   fake merge = joined a node whose founding cell is >= 2 maze steps away.
# Pre-registered run: no filter. Post-hoc variant "spike_filter": skip frames whose
# latent jumps > 1.0 from the previous frame (online-computable).
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stage2_place_recognition as s2  # noqa: E402
from grid_utils import obs_to_ij  # noqa: E402

EVAL_EVERY, N_EVAL_PAIRS, SPIKE = 50, 200, 1.0


def grow(obs, cells, ep_of, dist_maze, tau, spike_filter, rng):
    R = torch.empty_like(obs)  # node representatives (preallocated; first n rows used)
    rsq = torch.empty(len(obs))
    n = 0
    found, node_of = [], np.full(len(obs), -1)
    edges = set()
    fake_merges = merges = 0
    curve = []
    prev_node, prev_ep, prev_z = None, None, None
    o2 = (obs ** 2).sum(1)
    D = obs.shape[1]
    t0 = time.time()
    for k in range(len(obs)):
        if ep_of[k] != prev_ep:
            prev_node, prev_z = None, None
        z = obs[k]
        if spike_filter and prev_z is not None and float((z - prev_z).pow(2).mean()) > SPIKE:
            prev_z, prev_ep = z, ep_of[k]
            continue
        prev_z = z
        node = None
        if n:
            d = (rsq[:n] + o2[k] - 2 * (R[:n] @ z)) / D
            j = int(torch.argmin(d))
            if float(d[j]) <= tau:
                node = j
                merges += 1
                if dist_maze[found[j]].get(cells[k], 99) >= 2:
                    fake_merges += 1
        if node is None:
            R[n], rsq[n] = z, o2[k]
            found.append(cells[k])
            node = n
            n += 1
        node_of[k] = node
        if prev_node is not None and prev_node != node and ep_of[k] == prev_ep:
            edges.add((min(prev_node, node), max(prev_node, node)))
        prev_node, prev_ep = node, ep_of[k]
        if (k + 1) % EVAL_EVERY == 0 and n >= 5:
            curve.append(evaluate(k + 1, R[:n], found, edges, dist_maze, rng))
    A = adjacency(n, edges)
    t1 = time.time()
    shortest_path(A, unweighted=True, directed=False)
    sp_time = time.time() - t1
    return {"n_nodes": n, "n_edges": len(edges), "n_frames_used": int((node_of >= 0).sum()), "merges": merges,
            "fake_merges": fake_merges, "fake_merge_share": fake_merges / max(merges, 1),
            "fake_edges": count_fake_edges(edges, found, dist_maze), "all_pairs_sp_seconds": sp_time,
            "grow_seconds": time.time() - t0, "curve": curve}


def adjacency(n, edges):
    if not edges:
        return csr_matrix((n, n))
    e = np.array(list(edges))
    return csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))


def count_fake_edges(edges, found, dist_maze):
    return int(sum(dist_maze[found[a]].get(found[b], 99) >= 2 for a, b in edges))


def evaluate(step, R, found, edges, dist_maze, rng):
    n = R.shape[0]
    A = adjacency(n, edges)
    good = {(a, b) for a, b in edges if dist_maze[found[a]].get(found[b], 99) < 2}
    A_clean = adjacency(n, good)
    src = rng.choice(n, size=min(20, n), replace=False)
    D = shortest_path(A, unweighted=True, directed=False, indices=src)
    Dc = shortest_path(A_clean, unweighted=True, directed=False, indices=src)
    g, gc, lat, maze = [], [], [], []
    for _ in range(N_EVAL_PAIRS):
        si = rng.integers(len(src))
        a, b = int(src[si]), int(rng.integers(n))
        mz = dist_maze[found[a]].get(found[b])
        if a == b or mz is None or not np.isfinite(D[si, b]):
            continue
        g.append(D[si, b])
        gc.append(Dc[si, b] if np.isfinite(Dc[si, b]) else np.nan)
        lat.append(float((R[a] - R[b]).pow(2).mean()))
        maze.append(mz)
    out = {"step": step, "n_nodes": n, "n_pairs": len(g)}
    if len(g) >= 10:
        out["spearman_map_path"] = float(spearmanr(g, maze).correlation)
        out["spearman_latent_straight"] = float(spearmanr(lat, maze).correlation)
        ok = ~np.isnan(gc)
        out["spearman_map_path_fake_edges_removed"] = float(spearmanr(np.array(gc)[ok], np.array(maze)[ok]).correlation) if ok.sum() >= 10 else None
        out["share_pairs_disconnected_after_removal"] = float(1 - ok.mean())
    return out


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    st2 = json.load(open(os.path.join(HERE, "results_stage2.json")))
    sp = st2["best_space_at_wall5"]
    assert sp == "obs_full", sp
    taus = {name: st2["spaces"][sp]["best_at_wall_le"][key]["tau"] for name, key in (("conservative", "0.02"), ("medium", "0.05"), ("aggressive", "0.10"))}
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
        layout = maps[m].split("\\")
        _, dist_maze = s2.bfs_all(layout)
        obs, _, meta = s2.encode_map(model, data, images, starts, eps)
        cells = [tuple(int(v) for v in c) for c in obs_to_ij(meta[:, 2:4].astype(np.float64))]
        ep_of = meta[:, 0].astype(int)
        out["maps"][m] = {}
        for name, tau in taus.items():
            for filt in (False, True):
                key = f"{name}{'_spike_filter' if filt else ''}"
                out["maps"][m][key] = grow(obs, cells, ep_of, dist_maze, tau, filt, np.random.default_rng(m))
        r = out["maps"][m]["medium"]
        print(f"map {m}: medium nodes {r['n_nodes']} fake merges {r['fake_merges']}/{r['merges']} last {r['curve'][-1] if r['curve'] else None}", flush=True)
        json.dump(out, open(os.path.join(HERE, "results_stage3.json"), "w"))
        del obs
    print("done", flush=True)


if __name__ == "__main__":
    main()
