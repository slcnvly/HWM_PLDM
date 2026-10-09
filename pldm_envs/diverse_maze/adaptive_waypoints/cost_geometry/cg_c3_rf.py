# COST_GEOMETRY C3: measured effective receptive field. One input pixel of the map-0 background is
# set to the agent colour at 25 grid positions; per output spatial unit, change = L2 norm over channels.
# r_eff = farthest unit centre (input px) whose change >= 10% of the max change; median over positions.
import json
import os

import numpy as np
import torch

import cg_common as cg

HERE = os.path.dirname(os.path.abspath(__file__))
SPATIAL = {  # name -> (reshape, centre(k) in input px)
    "B1_trained": ((16, 43, 43), lambda k: 2 * k + 6),
    "B2_random_seed0": ((16, 43, 43), lambda k: 2 * k + 6),
    "B2_random_seed1": ((16, 43, 43), lambda k: 2 * k + 6),
    "B2_random_seed2": ((16, 43, 43), lambda k: 2 * k + 6),
    "B4_pixels": ((3, 98, 98), lambda k: k + 0.5),
    "B3_dinov2_patch8x8": ((384, 8, 8), lambda k: (k + 0.5) * 12.25),
}


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 10)))
    encs = cg.load_encoders()
    bg = np.round(cg.background()).astype(np.uint8)
    pos = [(r, c) for r in np.linspace(10, 88, 5).astype(int) for c in np.linspace(10, 88, 5).astype(int)]
    imgs = [bg]
    for r, c in pos:
        x = bg.copy(); x[r, c] = cg.AGENT_RGB.astype(np.uint8); imgs.append(x)
    imgs = np.stack(imgs)
    res = {}
    for name, (shape, cen) in SPATIAL.items():
        F = cg.feats(name, imgs, encs).reshape(len(imgs), *shape)
        base = F[0]
        reff, rnz, frac_in_analytic = [], [], []
        for k, (r, c) in enumerate(pos):
            ch = (F[k + 1] - base).pow(2).sum(0).sqrt().numpy()  # (H, W)
            H = ch.shape[0]
            cr = cen(np.arange(H))
            dr = cr[:, None] - (r + 0.5); dc = cr[None, :] - (c + 0.5)
            dist = np.hypot(dr, dc)
            mx = ch.max()
            reff.append(float(dist[ch >= 0.1 * mx].max()))
            rnz.append(float(dist[ch > 1e-6 * mx].max()))
            frac_in_analytic.append(float((ch[dist <= 8.5] ** 2).sum() / (ch ** 2).sum()))
        res[name] = {"r_eff_px_median": float(np.median(reff)), "r_eff_px_all": reff, "r_eff_px_iqr": [float(np.percentile(reff, 25)), float(np.percentile(reff, 75))],
                     "r_nonzero_px_median": float(np.median(rnz)), "share_sq_change_within_8.5px_median": float(np.median(frac_in_analytic)),
                     "unit_spacing_px": float(cen(1) - cen(0))}
        print(name, {k: v for k, v in res[name].items() if k != "r_eff_px_all"}, flush=True)
    # CLS: no spatial map; report how the change in CLS compares to the patch map (global by construction)
    F = cg.feats("B3_dinov2_cls", imgs, encs)
    res["B3_dinov2_cls"] = {"r_eff_px_median": None, "note": "no spatial output (global token); excluded from regression",
                            "cls_change_norm_median": float(np.median([(F[k + 1] - F[0]).norm() for k in range(len(pos))]))}
    res["analytic"] = {"B1_B2_menet6_d4rl_a": {"rf_px": 17, "radius_px": 8, "unit_centre": "2k+6", "note": "GroupNorm after each conv -> exact RF is the whole image"},
                       "B4_pixels": {"rf_px": 1}, "DINOv2_vits14": {"note": "global self-attention -> whole image; patch 14px at 224 = 6.125px at 98; pooled 2x2 -> 12.25px"}}
    json.dump(res, open(os.path.join(HERE, "results_cg_c3_rf.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
