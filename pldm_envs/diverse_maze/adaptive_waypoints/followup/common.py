# Shared helpers for the 2026-10-06 follow-up analyses. All model inputs go
# through adaptive_waypoints/preprocess.py (the evaluation-identical path).
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
AW = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(AW, "..", "..", ".."))
for p in (ROOT, os.path.join(AW, "boundary_study"), os.path.join(AW, "inference_study")):
    if p not in sys.path:
        sys.path.insert(0, p)

from pldm_envs.diverse_maze.adaptive_waypoints.preprocess import normalize_images, normalize_proprio_vel, normalize_actions  # noqa: E402
from grid_utils import obs_to_ij, open_neighbor_count  # noqa: E402

DATA = os.path.join(ROOT, "pldm_envs/diverse_maze/datasets/r50_local/r50_dataset")
WINDOW = 61
D_OBS_CH = 16


def load_full_model():
    """Pretrained hierarchical checkpoint (L1 identical to the boundary-study L1)."""
    from train_location_prober import load_model
    return load_model()


def load_prober():
    from pldm.models.misc import Prober
    p = Prober((16, 43, 43), arch="conv", output_shape=2, input_dim=(16, 43, 43), arch_subclass="c")
    p.load_state_dict(torch.load(os.path.join(AW, "inference_study", "location_prober_c.pt"), weights_only=False)["state_dict"])
    return p.eval()


LOC_MEAN = torch.tensor([4.3646, 4.2948])
LOC_STD = torch.tensor([2.3662, 2.3378])


def decode_xy(prober, obs):
    with torch.no_grad():
        return prober(obs.float()) * LOC_STD + LOC_MEAN


def episode_inputs(ep, images, ep_start, T=WINDOW):
    obs = ep["observations"][:T]
    img = normalize_images(torch.from_numpy(np.array(images[ep_start:ep_start + T])).float().permute(0, 3, 1, 2))
    pv = normalize_proprio_vel(torch.from_numpy(obs[:, 2:4]).float())
    acts = normalize_actions(torch.from_numpy(ep["actions"][:T - 1]).float())
    return img, pv, acts


def encode(model, img, pv):
    with torch.no_grad():
        out = model.level1.backbone(img, proprio=pv)
    return out.encodings, out.proprio_component  # (T,18,43,43), (T,2,43,43)


def episode_starts(splits):
    return np.concatenate([[0], np.cumsum([len(e["observations"]) for e in splits])])


def cell_type(layout, i, j):
    n = open_neighbor_count(layout, i, j)
    if n >= 3:
        return "junction"
    if n <= 1:
        return "dead_end"
    dirs = [(di, dj) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)) if layout[i + di][j + dj] != "#"]
    return "straight" if dirs[0][0] == -dirs[1][0] and dirs[0][1] == -dirs[1][1] else "corner"


def cell_types_for_frames(xy, layout):
    ij = obs_to_ij(np.asarray(xy, dtype=np.float64))
    return [cell_type(layout, int(a), int(b)) for a, b in ij]
