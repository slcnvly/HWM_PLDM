# Boundary-signal study, shared helpers for signal 1 (window-start-anchored
# open-loop error, the ORIGINAL pipeline's signal) and signal 1b (re-anchored
# one-step error, Amendment 1). See PREREGISTRATION.md Amendment 1.
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from pldm_envs.diverse_maze.adaptive_waypoints.compute_changepoints import load_level1  # noqa: E402

HERE = os.path.dirname(__file__)
DATA_ROOT = os.path.join(HERE, "..", "..", "datasets", "r50_local", "r50_dataset")
CONFIG_PATH = os.path.join(HERE, "..", "..", "..", "..", "pldm/configs/diverse_maze/icml/large_diverse_25maps_l2.yaml")
CKPT_PATH = os.path.join(DATA_ROOT, "3-9-1-seed248_epoch=3_sample_step=15465472.ckpt")

WINDOW = 61  # 61 frames = 60-step window, matches compute_changepoints.py


def get_model(device="cpu"):
    return load_level1(CONFIG_PATH, CKPT_PATH, device)


def compute_episode(model, ep, images, ep_start, device="cpu"):
    """Runs ONE forward_posterior call over the full 61-frame window, then a
    SECOND, single batched forward_multiple call (T=1, batch=60) to get the
    re-anchored one-step predictions for all 60 t's at once (Amendment 1's
    "batch, not a loop" requirement).

    Returns:
        err1: (60,) window-start-anchored open-loop error (signal 1, unchanged)
        err1b: (60,) re-anchored one-step error (signal 1b)
        encodings: (61, D) flattened real per-frame encodings (D = C*H*W)
    """
    obs = ep["observations"][:WINDOW]
    proprio_vel = torch.from_numpy(obs[:, 2:4]).float().unsqueeze(1).to(device)
    img_seq = torch.from_numpy(np.array(images[ep_start : ep_start + WINDOW])).float().permute(0, 3, 1, 2)
    states = img_seq.unsqueeze(1).to(device)
    actions = torch.from_numpy(ep["actions"][: WINDOW - 1]).float().unsqueeze(1).to(device)

    with torch.no_grad():
        result = model.level1.forward_posterior(states, actions, proprio_vel=proprio_vel, encode_only=False)

    full_encodings = result.backbone_output.encodings  # (61, 1, C, H, W)
    predictions = result.pred_output.predictions  # (61, 1, C, H, W); index i = prediction feeding into step i
    # signal 1 (unchanged): predictions[1:] are the rollout's own predictions
    # for frames 1..60, compared to the real frame's encoding.
    err1 = (full_encodings[1:] - predictions[1:]).pow(2).flatten(2).mean(dim=-1).squeeze(1).cpu()  # (60,)

    # signal 1b: batch of 60 independent 1-step predictions, each re-anchored
    # on the REAL encoding at t (not the rollout's accumulated state).
    proprio_component = result.backbone_output.proprio_component  # (61, 1, ...) or None
    state_encs_b = full_encodings[:60].squeeze(1).unsqueeze(0)  # (1, 60, C, H, W): time=1, batch=60
    actions_b = actions[:60].squeeze(1).unsqueeze(0)  # (1, 60, A)
    proprio_b = None
    if proprio_component is not None:
        proprio_b = proprio_component[:60].squeeze(1).unsqueeze(0)  # (1, 60, ...)

    with torch.no_grad():
        pred_out_b = model.level1.predictor.forward_multiple(
            state_encs=state_encs_b,
            actions=actions_b,
            T=1,
            proprio=proprio_b,
            compute_posterior=False,
        )
    predicted_next = pred_out_b.predictions[1]  # (60, C, H, W): the 1-step-ahead prediction for each t
    real_next = full_encodings[1:61].squeeze(1)  # (60, C, H, W): real z_{t+1}
    err1b = (real_next - predicted_next).pow(2).flatten(1).mean(dim=-1).cpu()  # (60,)

    encodings_flat = full_encodings.squeeze(1).flatten(1).cpu()  # (61, D)
    return err1.numpy(), err1b.numpy(), encodings_flat.numpy()


def iter_episodes(split_name, limit=None):
    """Yields (ep_idx, ep_dict, images_memmap, ep_start_offset) for a split."""
    splits = torch.load(os.path.join(DATA_ROOT, split_name, "data.p"), weights_only=False)
    images = np.load(os.path.join(DATA_ROOT, split_name, "images.npy"), mmap_mode="r")
    cum_lengths = np.cumsum([len(s["observations"]) for s in splits])
    n = len(splits) if limit is None else min(limit, len(splits))
    for ep_idx in range(n):
        ep_start = 0 if ep_idx == 0 else cum_lengths[ep_idx - 1]
        yield ep_idx, splits[ep_idx], images, ep_start
