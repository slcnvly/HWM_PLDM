# POST-HOC (not preregistered): recompute saturation displacements of the translation sweeps using
# only points whose content centre lies inside an OPEN maze cell of map 0 and whose content is fully
# on screen (the states a real agent can occupy). Same curves/radius; reported next to the prereg values.
import json
import os

import numpy as np
import torch

import cg_common as cg

ab = cg.ab
HERE = os.path.dirname(os.path.abspath(__file__))
# pixel (row, col) = A @ [x, y, 1] measured in probe_subspace/ps_c1_frames.py; inverted here
A = np.array([[3.00092599e-03, 9.46012472e+00], [-9.45758650e+00, 1.11380388e-02], [8.91619347e+01, 7.78860918e+00]])


def px_to_xy(rc):
    M = A[:2].T  # [[a_rx, a_ry], [a_cx, a_cy]]
    return np.linalg.solve(M, (rc - A[2]).T).T


def main():
    Z = np.load(os.path.join(HERE, "cg_m1_sweeps.npz"))
    maps = torch.load(os.path.join(ab.DATA, "train_maps.pt"), weights_only=False)
    L = maps[0].split("\\")
    starts, angles = Z["starts"], Z["angles"]
    out = {"note": "POST-HOC: centre in an open cell of map 0 and content fully on screen", "sat": {}}
    for sh in ("disc", "square"):
        for s in ((3, 4, 6, 9, 18) if sh == "disc" else (3, 6, 9, 18)):
            for rep in cg.REPS:
                ds_all, dv_all = [], []
                for k in range(len(angles)):
                    ds = Z[f"{sh}|{s}|{k}|d"].astype(float)
                    u = np.array([np.cos(angles[k]), np.sin(angles[k])])
                    cen = starts[k][None, :] + ds[:, None] * u[None, :]
                    ij = ab.obs_to_ij(px_to_xy(cen))
                    inside = (ij[:, 0] >= 0) & (ij[:, 0] < len(L)) & (ij[:, 1] >= 0) & (ij[:, 1] < len(L[0]))
                    open_ = np.array([inside[n] and L[ij[n, 0]][ij[n, 1]] == "O" for n in range(len(ds))])
                    onscreen = np.all((cen - s / 2 >= 0) & (cen + s / 2 <= cg.W), axis=1)
                    ok = open_ & onscreen
                    if not ok[0]:
                        continue  # start itself must be a valid state
                    ds_all.append(ds[ok]); dv_all.append(Z[f"{sh}|{s}|{k}|{rep}"][ok])
                if not ds_all:
                    continue
                ds, dv = np.concatenate(ds_all), np.concatenate(dv_all).astype(float)
                c = ab.curves(dv, -np.ones(len(ds), int), ds / cg.PX_PER_CELL, np.ones(len(ds), bool))
                r = ab.radius(c).get("R_xy")
                ud = np.unique(ds)
                out["sat"][f"{sh}|{s}|{rep}"] = {"R_xy_cells_cell_unit": r, "n_points": int(len(ds)), "n_valid_starts": len(ds_all),
                                                 "median_curve_px": {int(x): float(np.median(dv[ds == x])) for x in ud if (ds == x).sum() >= 5}}
                print(sh, s, rep, r, len(ds_all), flush=True)
    json.dump(out, open(os.path.join(HERE, "results_cg_posthoc_open_cells.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
