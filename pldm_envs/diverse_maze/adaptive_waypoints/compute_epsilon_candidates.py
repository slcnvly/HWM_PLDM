# HWM_PLDM 원본 대비 추가된 실험 코드 (fork: slcnvly/HWM_PLDM, branch: causal-waypoint-student).
# Kaggle GPU에서 실행. 도달 기반 종료(eval/arrival_based_termination.py)의 ε 후보를
# 학습 데이터에서 계산한다: "구간 마지막 스텝의 실제 latent"와 "level2가 그 구간에
# 대해 예측한 latent" 사이 거리 분포의 25/50/75 퍼센타일.
#
# level2는 raw pixel을 직접 안 보고 level1의 latent(identity_encoder로 pass-through)만
# 본다 (조사 결과 확인, hjepa.py:65-73/162-187). 그래서 HJEPA.forward_posterior를
# l2_states/l2_actions로 호출하면 내부에서
#   1) model.level1.forward_posterior(l2_states, actions=None, proprio_vel=..., encode_only=True)
#      로 각 경계 프레임을 인코딩하고
#   2) model.level2.forward_posterior(l1_obs, proprio=l1_proprio, actions=l2_actions, goal=None)
#      로 그 latent 시퀀스에 대해 구간별 one-segment-ahead 예측을 만든다
# (hjepa.py:158-187). l2_actions는 d4rl_adaptive.py::_build_l2_sample과 똑같이
# boundaries_to_segments + resample_to_fixed_length로 만든 고정 10스텝 청크다 --
# 학습 때와 완전히 같은 입력 구성이라야 "학습 시점 예측 오차" 분포가 맞다.
#
# 체크포인트: surprise_minseg8_finetuned (재학습 없음, disable_l2=False로 로드해서
# level2 가중치까지 씀 -- compute_changepoints.py::load_level1은 L1-only라 여긴 못 씀).
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
    boundaries_to_segments,
    resample_to_fixed_length,
)

_ENUM_STR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")


def _strip_enum_prefixes(obj):
    """Same as compute_changepoints.py's own helper -- replicated here (not
    imported) since that module's load_level1 is L1-only and we don't want
    to couple this script to it changing later."""
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


def load_full_hjepa(config_path: str, checkpoint_path: str, device: str):
    """Like compute_changepoints.py::load_level1, but disable_l2=False and
    keeps level2/posterior weights (only decoder is stripped) -- mirrors
    Trainer.maybe_load_model's load_l1_only=False branch (pldm/train.py:369-407)."""
    full = OmegaConf.load(config_path)
    full_dict = OmegaConf.to_container(full, resolve=True)
    _strip_enum_prefixes(full_dict)

    hjepa_dict = dict(full_dict["hjepa"])
    hjepa_dict["disable_l2"] = False
    hjepa_cfg = DataclassArgParser._populate_dataclass_from_dict(HJEPAConfig, hjepa_dict)

    model = HJEPA(hjepa_cfg, input_dim=(3, 98, 98), ppos_dim=0, pvel_dim=2, loc_dim=2)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    for k in list(state_dict.keys()):
        # same decoder-stripping as compute_changepoints.py::load_level1, minus
        # its "level2"/"posterior" conditions (we want those weights this time)
        if "decoder.converter" in k or "decoder" in k:
            del state_dict[k]
    res = model.load_state_dict(state_dict, strict=False)
    assert len(res.unexpected_keys) == 0, f"unexpected keys: {res.unexpected_keys}"

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def compute_segment_gap(model, l2_states, l2_actions, l2_proprio_vel, device):
    """l2_states: (n_anchors,1,3,98,98), l2_actions: (n_segments,1,10,2),
    l2_proprio_vel: (n_anchors,1,2). Returns (n_segments,) Euclidean distance
    between level2's predicted encoding for each segment and the actual
    encoding at that segment's end anchor -- same metric (torch.norm, no
    reduction beyond the feature dim) as eval/arrival_based_termination.py's
    online arrival check, for a directly comparable epsilon scale."""
    with torch.no_grad():
        result = model.forward_posterior(
            l2_states=l2_states.to(device),
            l2_actions=l2_actions.to(device),
            l2_proprio_vel=l2_proprio_vel.to(device),
            l2_proprio_pos=None,
            goal=None,
        )
    l2_result = result.level2
    predictions = l2_result.pred_output.predictions[1:]  # (n_segments,1,repr_dim)
    encodings = l2_result.backbone_output.encodings[1:]  # (n_segments,1,repr_dim)
    dist = torch.norm(predictions - encodings, dim=-1).squeeze(1).cpu()  # (n_segments,)
    return dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="r50 main split dir")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True, help="surprise_minseg8_finetuned .ckpt (full L1+L2)")
    parser.add_argument("--changepoints_path", type=str, required=True, help="changepoints_minseg8_surprise_main.pt")
    parser.add_argument("--out_path", type=str, default="outputs/epsilon_candidates.json")
    parser.add_argument("--l2_step_skip", type=int, default=10)
    parser.add_argument("--l2_n_steps", type=int, default=6)
    parser.add_argument("--limit_episodes", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    print(f"device={device}", flush=True)

    model = load_full_hjepa(args.config_path, args.checkpoint_path, device)
    assert model.level2 is not None, "level2 missing after load -- disable_l2 mismatch?"

    boundaries_by_ep = torch.load(args.changepoints_path, weights_only=False)
    splits = torch.load(os.path.join(args.data_path, "data.p"), weights_only=False)
    images = np.load(os.path.join(args.data_path, "images.npy"), mmap_mode="r")
    cum_lengths = np.cumsum([len(s["observations"]) for s in splits])

    window = args.l2_n_steps * args.l2_step_skip + 1  # 61
    action_len = window - 1  # 60

    ep_indices = sorted(boundaries_by_ep.keys())
    if args.limit_episodes is not None:
        ep_indices = ep_indices[: args.limit_episodes]
    print(f"{len(ep_indices)} episodes with cached surprise changepoints", flush=True)

    all_dists = []
    for count, ep_idx in enumerate(ep_indices):
        ep_len = len(splits[ep_idx]["observations"])
        if ep_len < window:
            continue
        boundaries = boundaries_by_ep[ep_idx]
        start = 0 if ep_idx == 0 else cum_lengths[ep_idx - 1]

        obs = splits[ep_idx]["observations"][:window]
        actions_raw = torch.from_numpy(splits[ep_idx]["actions"][:action_len]).float()
        img_seq = torch.from_numpy(np.array(images[start : start + window])).float().permute(0, 3, 1, 2)

        anchor_idx = [0] + list(boundaries) + [window - 1]
        l2_states = img_seq[anchor_idx].unsqueeze(1)  # (n_anchors,1,3,98,98)
        l2_proprio_vel = torch.from_numpy(obs[anchor_idx, 2:4]).float().unsqueeze(1)  # (n_anchors,1,2)

        segments = boundaries_to_segments(boundaries, action_len)
        assert len(segments) == args.l2_n_steps
        l2_actions = torch.stack(
            [resample_to_fixed_length(actions_raw[s:e], args.l2_step_skip) for s, e in segments],
            dim=0,
        ).unsqueeze(1)  # (n_segments,1,10,2)

        dist = compute_segment_gap(model, l2_states, l2_actions, l2_proprio_vel, device)
        all_dists.append(dist)

        if count % 100 == 0:
            print(f"episode {count}/{len(ep_indices)} (ep_idx={ep_idx}) done", flush=True)

    all_dists = torch.cat(all_dists).numpy()
    p25, p50, p75 = np.percentile(all_dists, [25, 50, 75])
    print(
        f"n_segments={len(all_dists)} mean={all_dists.mean():.4f} std={all_dists.std():.4f} "
        f"p25={p25:.4f} p50={p50:.4f} p75={p75:.4f}",
        flush=True,
    )

    import json

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(
            {
                "n_segments": int(len(all_dists)),
                "mean": float(all_dists.mean()),
                "std": float(all_dists.std()),
                "epsilon_candidates": {"p25": float(p25), "p50": float(p50), "p75": float(p75)},
            },
            f,
            indent=2,
        )
    print(f"saved {args.out_path}", flush=True)


if __name__ == "__main__":
    main()
