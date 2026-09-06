# HWM_PLDM 원본 대비 추가된 실험 코드 (fork: slcnvly/HWM_PLDM, branch: adaptive-waypoints).
# See DESIGN.md. One-time offline preprocessing: run the frozen, pretrained
# level1 model over each episode's raw 60-step window, compute per-step
# one-step prediction error, and cache changepoint-based waypoint boundaries
# for a range of min_seg values (swept later at training time via config,
# not recomputed -- this script runs once per min_seg value of interest).
#
# Does NOT need mujoco/d4rl/gym: operates purely on already-rendered images
# (images.npy) + already-generated proprio (data.p), both produced upstream
# by render_data.py/postprocess_images.py. Needs torch + omegaconf + the
# model code only.
import argparse
import os
import re
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from pldm.configs import DataclassArgParser  # noqa: E402
from pldm.models.hjepa import HJEPA, HJEPAConfig  # noqa: E402
from pldm_envs.diverse_maze.adaptive_waypoints.segmentation import (  # noqa: E402
    pick_changepoints,
)

_ENUM_STR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")


def _strip_enum_prefixes(obj):
    """Same trick as experiments/verify_optimizer_equivalence.py: the real
    pipeline resolves "ClassName.member" yaml strings via OmegaConf.
    structured(cls) schema merging; we load the yaml directly (no full
    TrainConfig, to avoid needing d4rl-adjacent import chains), so replicate
    the normalization by hand."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and _ENUM_STR_RE.match(v):
                obj[k] = v.split(".", 1)[1]
            else:
                _strip_enum_prefixes(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, str) and _ENUM_STR_RE.match(v):
                obj[i] = v.split(".", 1)[1]
            else:
                _strip_enum_prefixes(v)


def load_level1(config_path: str, checkpoint_path: str, device: str):
    full = OmegaConf.load(config_path)
    full_dict = OmegaConf.to_container(full, resolve=True)
    _strip_enum_prefixes(full_dict)

    hjepa_dict = dict(full_dict["hjepa"])
    hjepa_dict["disable_l2"] = True  # only need level1 for this script
    hjepa_cfg = DataclassArgParser._populate_dataclass_from_dict(HJEPAConfig, hjepa_dict)

    model = HJEPA(hjepa_cfg, input_dim=(3, 98, 98), ppos_dim=0, pvel_dim=2, loc_dim=2)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    for k in list(state_dict.keys()):
        if "level2" in k or "posterior" in k or "decoder.converter" in k or "decoder" in k:
            del state_dict[k]
    res = model.load_state_dict(state_dict, strict=False)
    assert len(res.unexpected_keys) == 0, f"unexpected keys: {res.unexpected_keys}"

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def compute_error_series(
    model: HJEPA, states: torch.Tensor, actions: torch.Tensor, proprio_vel: torch.Tensor
):
    """states: (T,1,3,98,98) float, actions: (T-1,1,2) float, proprio_vel:
    (T,1,2) float (level1's backbone uses proprio -- confirmed by the
    ValueError raised without it). Returns (T-1,) per-step squared
    prediction error (mean over feature dims), matching exactly what
    PredictionObjective averages over time -- here left per-step instead."""
    with torch.no_grad():
        result = model.level1.forward_posterior(
            states, actions, proprio_vel=proprio_vel, encode_only=False
        )
    encodings = result.backbone_output.encodings[1:]  # (T-1,1,...)
    predictions = result.pred_output.predictions[1:]
    err = (encodings - predictions).pow(2).flatten(2).mean(dim=-1)  # (T-1,1)
    return err.squeeze(1).cpu()  # (T-1,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--l2_step_skip", type=int, default=10)
    parser.add_argument("--l2_n_steps", type=int, default=6)
    parser.add_argument("--min_segs", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--limit_episodes", type=int, default=None)
    args = parser.parse_args()

    model = load_level1(args.config_path, args.checkpoint_path, args.device)

    splits = torch.load(os.path.join(args.data_path, "data.p"), weights_only=False)
    images = np.load(os.path.join(args.data_path, "images.npy"), mmap_mode="r")

    window = args.l2_n_steps * args.l2_step_skip + 1  # 61 for defaults
    n_boundaries = args.l2_n_steps - 1  # 5 for defaults

    cum_lengths = np.cumsum([len(s["observations"]) for s in splits])

    results = {ms: {} for ms in args.min_segs}
    n_episodes = len(splits) if args.limit_episodes is None else min(args.limit_episodes, len(splits))

    for ep_idx in range(n_episodes):
        ep_len = len(splits[ep_idx]["observations"])
        if ep_len < window:
            continue  # too short for a full l2 window; caller falls back to uniform
        start = 0 if ep_idx == 0 else cum_lengths[ep_idx - 1]

        obs = splits[ep_idx]["observations"][:window]
        proprio_vel = (
            torch.from_numpy(obs[:, 2:4]).float().unsqueeze(1).to(args.device)
        )  # (window, 1, 2) -- non-ant envs: obs[:, 2:] is qvel (d4rl.py _load_qvel)
        img_seq = torch.from_numpy(
            np.array(images[start : start + window])
        ).float().permute(0, 3, 1, 2)  # (window, 3, 98, 98)

        states = img_seq.unsqueeze(1).to(args.device)  # (window, 1, 3, 98, 98)
        actions = torch.from_numpy(
            splits[ep_idx]["actions"][: window - 1]
        ).float().unsqueeze(1).to(args.device)  # (window-1, 1, 2)

        err = compute_error_series(model, states, actions, proprio_vel)  # (window-1,) == 60
        assert err.shape[0] == args.l2_n_steps * args.l2_step_skip

        for ms in args.min_segs:
            boundaries = pick_changepoints(err, n_boundaries=n_boundaries, min_seg=ms)
            results[ms][ep_idx] = boundaries

        if ep_idx % 50 == 0:
            print(f"episode {ep_idx}/{n_episodes} done", flush=True)

    for ms, boundaries_by_ep in results.items():
        out_path = os.path.join(args.data_path, f"changepoints_minseg{ms}.pt")
        torch.save(boundaries_by_ep, out_path)
        print(f"saved {out_path}: {len(boundaries_by_ep)} episodes")


if __name__ == "__main__":
    main()
