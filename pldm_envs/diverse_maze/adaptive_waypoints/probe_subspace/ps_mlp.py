# PROBE_SUBSPACE P2: per-map MLP probe (1 hidden layer, 256, ReLU) on fold 0 of the same
# episode-grouped split as ps_main.py. B1 excl+incl, controls excl only (CPU time, prereg).
# Early stopping on 5 training episodes held out internally. Diagnostic instrument only.
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
AW = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(AW, "latent_law"))
import ab_saturation as ab  # noqa: E402

RUNS = [("B1_trained", "excl"), ("B1_trained", "incl"), ("B2_random_seed0", "excl"), ("B2_random_seed1", "excl"),
        ("B2_random_seed2", "excl"), ("B4_pixels", "excl")]
OUT = os.path.join(HERE, "results_ps_mlp.json")
EPOCHS, BS = 60, 256


def r2(pred, Y):
    r = [1 - ((pred[:, d] - Y[:, d]) ** 2).sum() / ((Y[:, d] - Y[:, d].mean()) ** 2).sum() for d in range(2)]
    return [float(r[0]), float(r[1]), float(np.mean(r))]


def fit(X, Y, tr, va, te, seed=0):
    torch.manual_seed(seed)
    mu = X[tr].mean(0, keepdim=True); sd = (X[tr] - mu).std()
    ymu = Y[tr].mean(0, keepdim=True)
    f = lambda idx: (X[idx] - mu) / sd  # noqa: E731
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, 2))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, best_ep = np.inf, None, -1
    g = torch.Generator().manual_seed(seed)
    for ep in range(EPOCHS):
        net.train()
        order = tr[torch.randperm(len(tr), generator=g).numpy()]
        for i in range(0, len(order), BS):
            b = order[i:i + BS]
            loss = (net(f(b)) - (Y[b] - ymu)).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float((net(f(va)) - (Y[va] - ymu)).pow(2).mean())
        if v < best:
            best, best_ep = v, ep
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(best_state)
    with torch.no_grad():
        p = (net(f(te)) + ymu).numpy()
    return p, best_ep


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 10)))
    data = torch.load(os.path.join(ab.DATA, "data.p"), weights_only=False)
    images = np.load(os.path.join(ab.DATA, "images.npy"), mmap_mode="r")
    starts = np.concatenate([[0], np.cumsum([len(e["observations"]) for e in data])])
    spike_all = np.load(os.path.join(AW, "displacement_pairs", "dp_latents.npz"))["spike"]
    encs = {"B1_trained": ab.load_model().level1.backbone, "B4_pixels": None}
    for s in range(3):
        encs[f"B2_random_seed{s}"] = ab.random_backbone(s)
    by_map = {}
    for e, ep in enumerate(data):
        by_map.setdefault(int(ep["map_idx"]), []).append(e)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"maps": {}}
    for m, eps in sorted(by_map.items()):
        if str(m) in res["maps"]:
            continue
        T = len(data[eps[0]]["observations"])
        Y = torch.from_numpy(np.concatenate([data[e]["observations"][:, :2] for e in eps]).astype(np.float32) / ab.GRID_SIZE)
        spike = np.concatenate([spike_all[e] for e in eps])
        perm = np.random.default_rng(0).permutation(len(eps))  # same fold assignment as ps_main.py
        fold_of_ep = {eps[k]: int(np.nonzero(perm == k)[0][0] // 10) for k in range(len(eps))}
        ep_arr = np.repeat(eps, T)
        fold = np.array([fold_of_ep[e] for e in ep_arr])
        train_eps = [e for e in eps if fold_of_ep[e] != 0]
        inner_va_eps = set(np.random.default_rng(1).choice(train_eps, 5, replace=False).tolist())
        mres = {}
        X = None
        for rep, ver in RUNS:
            if X is None or cur != rep:
                X = torch.cat([ab.features(rep, np.array(images[starts[e]:starts[e + 1]]), encs[rep]) for e in eps]).float()
                cur = rep
            mask = ~spike if ver == "excl" else np.ones(len(ep_arr), bool)
            te = np.nonzero(mask & (fold == 0))[0]
            va = np.nonzero(mask & (fold != 0) & np.isin(ep_arr, list(inner_va_eps)))[0]
            tr = np.nonzero(mask & (fold != 0) & ~np.isin(ep_arr, list(inner_va_eps)))[0]
            p, best_ep = fit(X, Y, tr, va, te)
            err = np.linalg.norm(p - Y[te].numpy(), axis=1)
            mres[f"{rep}|{ver}"] = {"r2_fold0": r2(p, Y[te].numpy()), "best_epoch": best_ep, "err_cells_median": float(np.median(err))}
            print(f"map {m} {rep} {ver}: MLP R2 fold0 {mres[f'{rep}|{ver}']['r2_fold0'][2]:.4f} best ep {best_ep}", flush=True)
        res["maps"][str(m)] = mres
        json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
