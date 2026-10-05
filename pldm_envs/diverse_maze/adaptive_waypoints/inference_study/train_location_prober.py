# Diagnostic location prober for decoding L2 waypoints (INFERENCE_PREREG.md SS5).
# Same architecture the evaluator would build (Prober arch=conv, subclass=c, input
# obs_component (16,43,43), output normalized xy), trained on frozen pretrained-L1
# encodings of the r50 probe split. Model weights are never updated.
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from omegaconf import OmegaConf  # noqa: E402
from pldm.configs import DataclassArgParser  # noqa: E402
from pldm.models.hjepa import HJEPA, HJEPAConfig  # noqa: E402
from pldm.models.misc import Prober  # noqa: E402
from pldm_envs.diverse_maze.adaptive_waypoints.compute_changepoints import _strip_enum_prefixes  # noqa: E402

DATA = os.path.join(ROOT, "pldm_envs/diverse_maze/datasets/r50_local/r50_dataset")
CKPT = os.path.abspath(os.path.join(ROOT, "..", "experiments/hwm_finetune_checkpoints_dataset/pretrained_baseline.ckpt"))
CONFIG = os.path.join(ROOT, "pldm/configs/diverse_maze/icml/large_diverse_25maps_l2.yaml")
OUT = os.path.join(HERE, "location_prober_c.pt")
LOC_MEAN = torch.tensor([4.3646, 4.2948])
LOC_STD = torch.tensor([2.3662, 2.3378])
PVEL_MEAN = torch.tensor([-0.0291, -0.0461])
PVEL_STD = torch.tensor([1.4084, 1.4102])
GRID = (8.248 - 0.351) / 8


def load_model():
    full = OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=False)
    _strip_enum_prefixes(full)
    cfg = DataclassArgParser._populate_dataclass_from_dict(HJEPAConfig, dict(full["hjepa"]))
    m = HJEPA(cfg, input_dim=(3, 98, 98), ppos_dim=0, pvel_dim=2, loc_dim=2)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    m.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()}, strict=False)
    m.eval()
    return m


STATE_MEAN = torch.tensor([146.5709, 120.0509, 93.3956]).view(1, 3, 1, 1)
STATE_STD = torch.tensor([84.9847, 45.3689, 10.3962]).view(1, 3, 1, 1)


def encode_split(model, split, ep_ids, every=2):
    data = torch.load(os.path.join(DATA, split, "data.p"), weights_only=False)
    images = np.load(os.path.join(DATA, split, "images.npy"), mmap_mode="r")
    starts = np.concatenate([[0], np.cumsum([len(e["observations"]) for e in data])])
    X, Y = [], []
    for n, e in enumerate(ep_ids):
        ep = data[e]
        idx = np.arange(0, len(ep["observations"]), every)
        img = torch.from_numpy(np.array(images[starts[e] + idx])).float().permute(0, 3, 1, 2)
        img = (img - STATE_MEAN) / (STATE_STD + 1e-6)
        pv = (torch.from_numpy(ep["observations"][idx, 2:4]).float() - PVEL_MEAN) / (PVEL_STD + 1e-6)
        with torch.no_grad():
            o = model.level1.backbone(img, proprio=pv)
        X.append(o.obs_component.half())
        Y.append((torch.from_numpy(ep["observations"][idx, :2]).float() - LOC_MEAN) / LOC_STD)
        if (n + 1) % 50 == 0:
            print(f"  {split}: encoded {n + 1}/{len(ep_ids)} episodes", flush=True)
    return torch.cat(X), torch.cat(Y)


def err_cells(prober, X, Y):
    with torch.no_grad():
        p = torch.cat([prober(X[i:i + 512].float()) for i in range(0, len(X), 512)])
    e = ((p - Y) * LOC_STD).norm(dim=1) / GRID
    return float(e.mean()), float(e.median()), float(np.percentile(e.numpy(), 95))


def main():
    torch.manual_seed(0)
    torch.set_num_threads(10)
    t0 = time.time()
    model = load_model()
    rng = np.random.default_rng(0)
    perm = rng.permutation(1000)
    Xtr, Ytr = encode_split(model, "probe", perm[:250])
    Xva, Yva = encode_split(model, "probe", perm[900:960])
    Xmain, Ymain = encode_split(model, "main", rng.permutation(1250)[:60])
    print(f"encoded: train {len(Xtr)}, val {len(Xva)}, main(cross-map) {len(Xmain)} [{time.time() - t0:.0f}s]", flush=True)

    prober = Prober((16, 43, 43), arch="conv", output_shape=2, input_dim=(16, 43, 43), arch_subclass="c")
    opt = torch.optim.Adam(prober.parameters(), lr=0.0032)
    epochs, bs = 20, 128
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * (len(Xtr) // bs + 1))
    for ep in range(epochs):
        prober.train()
        order = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(Xtr), bs):
            b = order[i:i + bs]
            loss = (prober(Xtr[b].float()) - Ytr[b]).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(b)
        prober.eval()
        print(f"epoch {ep}: train mse {tot / len(Xtr):.4f}, val err cells (mean,median,p95) {err_cells(prober, Xva, Yva)} [{time.time() - t0:.0f}s]", flush=True)

    res = {
        "val_probe_maps_err_cells_mean_median_p95": err_cells(prober, Xva, Yva),
        "cross_map_main_err_cells_mean_median_p95": err_cells(prober, Xmain, Ymain),
        "n_train": len(Xtr), "epochs": epochs, "grid_size": GRID,
        "loc_mean": LOC_MEAN.tolist(), "loc_std": LOC_STD.tolist(),
        "checkpoint_for_encoder": os.path.basename(CKPT),
    }
    torch.save({"state_dict": prober.state_dict(), "meta": res}, OUT)
    json.dump(res, open(os.path.join(HERE, "location_prober_c_metrics.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
