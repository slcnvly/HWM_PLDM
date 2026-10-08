# DISPLACEMENT_PAIRS step 0: encode every r50 frame (eval-path preprocessing via
# ab_saturation.features), per-frame spike flags (A2 rule), and the latent distance
# (pair_mse definition) of every within-episode pair with gap >= 60 (for M4).
# One episode at a time; nothing larger than one episode's latents is kept in memory.
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
AW = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(AW, "latent_law"))
import ab_saturation as ab  # noqa: E402

GAP_MIN = 60
OUT = os.path.join(HERE, "dp_latents.npz")


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 10)))
    data = torch.load(os.path.join(ab.DATA, "data.p"), weights_only=False)
    images = np.load(os.path.join(ab.DATA, "images.npy"), mmap_mode="r")
    starts = np.concatenate([[0], np.cumsum([len(e["observations"]) for e in data])])
    enc = ab.load_model().level1.backbone
    T = len(data[0]["observations"])
    spike = np.zeros((len(data), T), bool)
    cons = np.zeros((len(data), T - 1), np.float32)
    ii, jj = np.triu_indices(T, GAP_MIN)
    lat = np.zeros((len(data), len(ii)), np.float32)
    for e in range(len(data)):
        assert len(data[e]["observations"]) == T
        z = ab.features("B1_trained", np.array(images[starts[e]:starts[e + 1]]), enc)
        D = ab.pair_mse(z).numpy()
        d = np.diag(D, 1)
        cons[e] = d
        sp = np.zeros(T, bool)
        sp[1:-1] = (d[:-1] > ab.SPIKE) & (d[1:] > ab.SPIKE)  # prev and next both > 1; end frames never spikes (A2)
        spike[e] = sp
        lat[e] = D[ii, jj]
        if (e + 1) % 125 == 0:
            print(f"{e + 1}/{len(data)} spike rate so far {spike[:e + 1].mean():.4f}", flush=True)
    np.savez_compressed(OUT, spike=spike, consecutive=cons, pair_i=ii, pair_j=jj, lat=lat)
    print("spike rate", spike.mean(), "saved", OUT)


if __name__ == "__main__":
    main()
