# COST_GEOMETRY M1: synthetic displacement sweeps (COST_GEOMETRY_PREREG.md). For each
# (shape, s, start) render the sweep once, compute every representation, store pair_mse(f(x_0), f(x_d)).
import os

import numpy as np
import torch

import cg_common as cg

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = [3, 6, 9, 18, 36, 72]
SHAPES = ["disc", "square", "arm"]
N_START = 20
OUT = os.path.join(HERE, "cg_m1_sweeps.npz")


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 10)))
    encs = cg.load_encoders()
    bg = cg.background()
    rng = np.random.default_rng(0)
    starts = rng.uniform(0, cg.W, size=(N_START, 2))
    angles = rng.uniform(0, 2 * np.pi, size=N_START)
    out = {"starts": starts, "angles": angles}
    jobs = [(sh, s) for sh in SHAPES for s in SIZES] + [("disc", 4)]
    for sh, s in jobs:
        for k in range(N_START):
            u = np.array([np.cos(angles[k]), np.sin(angles[k])])
            dmax = cg.W if sh != "arm" else min(2 * s, cg.W)
            ds = np.arange(0, dmax + 1)
            imgs = []
            for d in ds:
                if sh == "arm":
                    cov = cg.coverage("arm", s, arm=cg.arm_pose(starts[k], u, s, d))
                else:
                    cov = cg.coverage(sh, s, center=starts[k] + d * u)
                imgs.append(cg.compose(bg, cov))
            imgs = np.stack(imgs)
            out[f"{sh}|{s}|{k}|d"] = ds
            for rep in cg.REPS:
                F = cg.feats(rep, imgs, encs).double()
                out[f"{sh}|{s}|{k}|{rep}"] = ((F - F[0:1]) ** 2).mean(1).float().numpy()  # pair_mse vs d = 0
        print(f"done {sh} s={s}", flush=True)
        np.savez_compressed(OUT, **out)


if __name__ == "__main__":
    main()
