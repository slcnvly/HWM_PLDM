# COST_GEOMETRY C1 (agent appearance) and C2 (background fixed?) on all r50 main frames.
# Background per map = per-pixel median over that map's 5,050 frames (the agent moves, so the
# median is the agent-free render). Agent pixel position from obs xy via the linear map measured
# in probe_subspace/ps_c1_frames.py (9.34 px/cell). One map at a time.
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
AW = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(AW, "latent_law"))
import ab_saturation as ab  # noqa: E402

THRS = [0, 30]


def agent_px(xy, coef):
    return np.c_[xy, np.ones(len(xy))] @ coef  # (row, col)


def main():
    data = torch.load(os.path.join(ab.DATA, "data.p"), weights_only=False)
    images = np.load(os.path.join(ab.DATA, "images.npy"), mmap_mode="r")
    starts = np.concatenate([[0], np.cumsum([len(e["observations"]) for e in data])])
    by_map = {}
    for e, ep in enumerate(data):
        by_map.setdefault(int(ep["map_idx"]), []).append(e)
    # pixel <- position map, refit here on map 0 with the agent centroid (same procedure as ps_c1_frames.py)
    res = {"maps": {}}
    rows = []  # per-frame: map, speed, area0, area30, bbox_h, bbox_w, far_px, mean_rgb(3)
    bgs = {}
    for m, eps in sorted(by_map.items()):
        F = np.asarray(images[starts[eps[0]]:starts[eps[-1] + 1]]).astype(np.int16)
        bg = np.median(F, 0).astype(np.int16)
        bgs[m] = bg.astype(np.uint8)
        D = np.abs(F - bg).sum(-1)  # (N, 98, 98)
        obs = np.concatenate([data[e]["observations"] for e in eps])
        cen = []
        for k in range(len(F)):
            rr, cc = np.nonzero(D[k] > 30)
            cen.append([rr.mean(), cc.mean()] if len(rr) else [np.nan, np.nan])
        cen = np.array(cen)
        ok = ~np.isnan(cen[:, 0])
        coef, *_ = np.linalg.lstsq(np.c_[obs[ok, :2], np.ones(ok.sum())], cen[ok], rcond=None)
        ap = agent_px(obs[:, :2], coef)
        far, nfar_px = [], []
        for k in range(len(F)):
            rr, cc = np.nonzero(D[k] > 0)
            dist = np.hypot(rr - ap[k, 0], cc - ap[k, 1])
            far.append(dist.max() if len(dist) else 0.0)
            nfar_px.append(int((dist > 6).sum()))
            m30 = D[k] > 30
            r30, c30 = np.nonzero(m30)
            rgb = F[k][m30].mean(0) if m30.any() else np.full(3, np.nan)
            sp = float(np.linalg.norm(obs[k, 2:4]))
            rows.append([m, sp, int((D[k] > 0).sum()), int(m30.sum()),
                         int(r30.max() - r30.min() + 1) if len(r30) else 0, int(c30.max() - c30.min() + 1) if len(c30) else 0,
                         float(dist.max()) if len(dist) else 0.0, *rgb.tolist()])
        far = np.array(far); nfar_px = np.array(nfar_px)
        res["maps"][m] = {"frames": int(len(F)), "frames_with_any_diff_px_farther_than_6px_from_agent": int((nfar_px > 0).sum()),
                          "max_far_px": float(far.max()), "px_per_cell_fit": [float(np.hypot(*coef[:2, 0]) * ab.GRID_SIZE), float(np.hypot(*coef[:2, 1]) * ab.GRID_SIZE)]}
        print(m, res["maps"][m], flush=True)
        del F, D
    R = np.array(rows)
    sp, a0, a30, bh, bw = R[:, 1], R[:, 2], R[:, 3], R[:, 4], R[:, 5]
    rgb = R[:, 7:10]
    q = lambda v: {p: float(np.nanpercentile(v, p)) for p in (0, 1, 50, 99, 100)}  # noqa: E731
    res["C1"] = {"area_px_diff_gt0": q(a0), "area_px_diff_gt30": q(a30), "bbox_h_px_gt30": q(bh), "bbox_w_px_gt30": q(bw),
                 "mean_rgb_in_agent_gt30": [float(np.nanmedian(rgb[:, i])) for i in range(3)],
                 "corr_speed_vs_area_gt0": float(np.corrcoef(sp, a0)[0, 1]), "corr_speed_vs_area_gt30": float(np.corrcoef(sp, a30)[0, 1]),
                 "corr_speed_vs_rgb": [float(np.corrcoef(sp[~np.isnan(rgb[:, i])], rgb[~np.isnan(rgb[:, i]), i])[0, 1]) for i in range(3)],
                 "area_gt30_by_speed_quartile": [float(np.median(a30[(sp >= lo) & (sp < hi)])) for lo, hi in zip(np.percentile(sp, [0, 25, 50, 75]), list(np.percentile(sp, [25, 50, 75])) + [np.inf])],
                 "n_frames": int(len(R))}
    res["C2"] = {"total_frames_with_diff_far_from_agent": int(sum(v["frames_with_any_diff_px_farther_than_6px_from_agent"] for v in res["maps"].values())),
                 "max_far_px_over_maps": float(max(v["max_far_px"] for v in res["maps"].values()))}
    np.savez_compressed(os.path.join(HERE, "cg_backgrounds.npz"), **{str(m): b for m, b in bgs.items()})
    json.dump(res, open(os.path.join(HERE, "results_cg_c1_c2.json"), "w"), indent=1)
    # figure: agent crops at low / high speed, and a background
    m0 = 0
    eps = by_map[m0]
    obs = np.concatenate([data[e]["observations"] for e in eps])
    spd = np.linalg.norm(obs[:, 2:4], axis=1)
    order = np.argsort(spd)
    picks = list(order[:4]) + list(order[-4:])
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes[0, 0].imshow(bgs[m0]); axes[0, 0].set_title("map 0 background (median)", fontsize=8)
    axes[1, 0].axis("off")
    for n, k in enumerate(picks):
        ax = axes[n // 4, 1 + n % 4]
        img = np.asarray(images[starts[eps[0]] + k])
        d = np.abs(img.astype(int) - bgs[m0].astype(int)).sum(-1)
        rr, cc = np.nonzero(d > 30)
        r0, c0 = int(rr.mean()), int(cc.mean())
        crop = img[max(r0 - 8, 0):r0 + 9, max(c0 - 8, 0):c0 + 9]
        ax.imshow(crop, interpolation="nearest"); ax.set_title(f"speed {spd[k]:.2f}, px>30: {(d > 30).sum()}", fontsize=8); ax.axis("off")
    fig.suptitle("C1: agent crops (17x17 px) at the 4 lowest (top) and 4 highest (bottom) speeds, map 0", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "c1_agent_crops.png"), dpi=120)
    print(json.dumps({k: res[k] for k in ("C1", "C2")}, indent=1))


if __name__ == "__main__":
    main()
