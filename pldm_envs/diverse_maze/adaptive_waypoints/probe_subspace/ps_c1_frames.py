# PROBE_SUBSPACE C1/C2: save raw r50 frames (two maps, several times in one episode) and measure
# step displacement, px/cell, visible cells. Positions via grid_utils (GRID_SIZE, OBS_MIN_TOTAL).
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
from grid_utils import OBS_MIN_TOTAL  # noqa: E402

data = torch.load(os.path.join(ab.DATA, "data.p"), weights_only=False)
images = np.load(os.path.join(ab.DATA, "images.npy"), mmap_mode="r")
starts = np.concatenate([[0], np.cumsum([len(e["observations"]) for e in data])])
fig, axes = plt.subplots(2, 4, figsize=(13, 6.8))
for row, e in enumerate([0, 50 * 7]):
    for col, t in enumerate([0, 33, 66, 100]):
        ax = axes[row, col]
        ax.imshow(images[starts[e] + t])
        x, y = data[e]["observations"][t, :2]
        ax.set_title(f"ep {e} (map {int(data[e]['map_idx'])}) t={t}\nxy=({x:.2f},{y:.2f})", fontsize=8)
        ax.axis("off")
fig.suptitle("r50 raw frames (98x98): same episode over time, two maps", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "c1_frames.png"), dpi=110)

# C2: one-step displacement in cells
d = np.concatenate([np.linalg.norm(np.diff(e["observations"][:, :2], axis=0), axis=1) for e in data]) / ab.GRID_SIZE
print("one-frame displacement cells: median %.4f p90 %.4f mean %.4f" % (np.median(d), np.percentile(d, 90), d.mean()))
# px/cell: fit agent pixel centroid vs position. Agent = pixels that differ most from the episode's median image.
P, X = [], []
for e in range(0, 1250, 25):
    ims = np.asarray(images[starts[e]:starts[e + 1]]).astype(float)
    med = np.median(ims, 0)
    for t in range(0, 101, 5):
        diff = np.abs(ims[t] - med).sum(-1)
        m = diff > 0.5 * diff.max()
        if m.sum() < 3:
            continue
        r, c = np.nonzero(m)
        P.append([r.mean(), c.mean()]); X.append(data[e]["observations"][t, :2])
P, X = np.array(P), np.array(X)
A = np.c_[X, np.ones(len(X))]
coef, *_ = np.linalg.lstsq(A, P, rcond=None)
resid = P - A @ coef
print("pixel(row,col) = A @ [x, y, 1]; coef:\n", coef, "\nresid px median", np.median(np.linalg.norm(resid, axis=1)))
px_per_unit = np.sqrt((coef[:2] ** 2).sum(0))
print("px per obs unit (row from x?, col from y?)", px_per_unit, "-> px per cell", px_per_unit * ab.GRID_SIZE)
print("visible cells across 98 px:", 98 / (px_per_unit * ab.GRID_SIZE))
print("maze layout size (rows x cols):", len(torch.load(os.path.join(ab.DATA, 'train_maps.pt'), weights_only=False)[0].split('\\')), "OBS_MIN_TOTAL", OBS_MIN_TOTAL)
