# HWM_PLDM 원본 대비 추가된 실험 코드 (fork: slcnvly/HWM_PLDM, branch: causal-waypoint-student).
# Kaggle GPU에서 실행. 도달 기반 종료(eval/arrival_based_termination.py)의 ε 후보를
# 학습 데이터에서 계산한다.
#
# v2 (실제 Kaggle 실행에서 잡힌 버그 2개 수정):
#   1. 온라인 도달 체크는 level2 예측의 "obs_component"만 본다 (level1의
#      backbone_output.obs_component -- proprio 채널 제외, 33282 대신 29584차원,
#      encode_real_obs의 shape-mismatch 버그로 확인됨). level2 predictor 출력도
#      마찬가지로 pred_output.obs_component가 이미 분리된 필드로 존재한다
#      (sequence_predictor.py:396-399, PredictorOutput -- pred_output.predictions는
#      proprio까지 합쳐진 전체 공간이라 다른 스케일). v1은 실수로 .predictions/
#      .encodings(전체 공간)를 썼음 -- v2는 .obs_component로 수정.
#   2. ε는 "구간이 끝나는 시점"이 아니라 실제 온라인 체크가 일어나는 창
#      (재계획 직후 min_gap~max_k 스텝, 기본 2~8) 안에서 측정해야 한다. v1은
#      구간 끝(최대 20+ 스텝 뒤)의 거리로 ε를 잡아서, max_k=8 창 안에서는
#      한 번도 도달 못 하는 ε가 나왔다(실제 실행에서 3개 ε 후보 전부
#      arrival=0으로 확인됨). v2는 각 구간 시작 이후 1..max_k 스텝 지점마다
#      "그 시점의 실제 latent"와 "그 구간의 level2 예측 타겟" 사이 거리를 다
#      모아서 분포를 낸다 -- 이게 온라인 트리거가 실제로 보는 것과 같다.
#
# level2는 raw pixel을 직접 안 보고 level1의 latent(identity_encoder로 pass-through)만
# 본다 (hjepa.py:65-73/162-187). l2_actions는 d4rl_adaptive.py::_build_l2_sample과
# 똑같이 boundaries_to_segments + resample_to_fixed_length로 만든 고정 10스텝 청크다.
#
# 체크포인트: surprise_minseg8_finetuned (재학습 없음, disable_l2=False로 로드).
import argparse
import json
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
        if "decoder.converter" in k or "decoder" in k:
            del state_dict[k]
    res = model.load_state_dict(state_dict, strict=False)
    assert len(res.unexpected_keys) == 0, f"unexpected keys: {res.unexpected_keys}"

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def compute_segment_targets(model, l2_states, l2_actions, l2_proprio_vel, device):
    """Returns (n_segments, repr_dim) -- level2's predicted obs_component
    target for each segment (what reset_targets would store online)."""
    with torch.no_grad():
        result = model.forward_posterior(
            l2_states=l2_states.to(device),
            l2_actions=l2_actions.to(device),
            l2_proprio_vel=l2_proprio_vel.to(device),
            l2_proprio_pos=None,
            goal=None,
        )
    l2_result = result.level2
    targets = l2_result.pred_output.obs_component[1:]  # (n_segments,1,repr_dim) -- obs-only, matches target_enc's space
    return targets.squeeze(1).cpu()  # (n_segments, repr_dim)


def encode_frames_obs_component(model, img_seq, proprio_vel, frame_indices, device, batch_size=32):
    """img_seq: (window,3,98,98) raw frames, proprio_vel: (window,2). Encodes
    the requested frame indices one small batch at a time via level1's
    backbone alone (single-frame, causal -- matches
    eval/arrival_based_termination.py::encode_real_obs's online convention),
    returns (len(frame_indices), repr_dim) obs_component encodings."""
    outs = []
    for i in range(0, len(frame_indices), batch_size):
        idx = frame_indices[i : i + batch_size]
        obs_batch = img_seq[idx].to(device)  # (b,3,98,98)
        proprio_batch = proprio_vel[idx].to(device)  # (b,2)
        with torch.no_grad():
            backbone_output = model.level1.backbone(obs_batch, proprio=proprio_batch, locations=None)
        outs.append(backbone_output.obs_component.flatten(1).cpu())
    return torch.cat(outs, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="r50 main split dir")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True, help="surprise_minseg8_finetuned .ckpt (full L1+L2)")
    parser.add_argument("--changepoints_path", type=str, required=True, help="changepoints_minseg8_surprise_main.pt")
    parser.add_argument("--out_path", type=str, default="outputs/epsilon_candidates.json")
    parser.add_argument("--l2_step_skip", type=int, default=10)
    parser.add_argument("--l2_n_steps", type=int, default=6)
    parser.add_argument("--max_k", type=int, default=8, help="online arrival check window -- must match arrival_max_k")
    parser.add_argument("--min_gap", type=int, default=2, help="online arrival check window start -- must match arrival_min_gap")
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
    print(f"{len(ep_indices)} episodes with cached surprise changepoints, "
          f"measuring within reachable window [{args.min_gap},{args.max_k}] steps post-replan", flush=True)

    all_dists = []
    for count, ep_idx in enumerate(ep_indices):
        ep_len = len(splits[ep_idx]["observations"])
        if ep_len < window:
            continue
        boundaries = boundaries_by_ep[ep_idx]
        start = 0 if ep_idx == 0 else cum_lengths[ep_idx - 1]

        obs = splits[ep_idx]["observations"][:window]
        proprio_vel_all = torch.from_numpy(obs[:, 2:4]).float()  # (window,2)
        actions_raw = torch.from_numpy(splits[ep_idx]["actions"][:action_len]).float()
        img_seq = torch.from_numpy(np.array(images[start : start + window])).float().permute(0, 3, 1, 2)  # (window,3,98,98)

        anchor_idx = [0] + list(boundaries) + [window - 1]
        l2_states = img_seq[anchor_idx].unsqueeze(1)  # (n_anchors,1,3,98,98)
        l2_proprio_vel = proprio_vel_all[anchor_idx].unsqueeze(1)  # (n_anchors,1,2)

        segments = boundaries_to_segments(boundaries, action_len)
        assert len(segments) == args.l2_n_steps
        l2_actions = torch.stack(
            [resample_to_fixed_length(actions_raw[s:e], args.l2_step_skip) for s, e in segments],
            dim=0,
        ).unsqueeze(1)  # (n_segments,1,10,2)

        targets = compute_segment_targets(model, l2_states, l2_actions, l2_proprio_vel, device)  # (n_segments, repr_dim)

        # for each segment, gather the raw frame indices within the reachable window
        query_frame_idx, query_seg_idx = [], []
        for seg_i, (s, e) in enumerate(segments):
            hi = min(s + args.max_k, e, window - 1)
            lo = min(s + args.min_gap, hi)
            for t in range(lo, hi + 1):
                query_frame_idx.append(t)
                query_seg_idx.append(seg_i)
        if not query_frame_idx:
            continue

        encs = encode_frames_obs_component(model, img_seq, proprio_vel_all, query_frame_idx, device)  # (n_queries, repr_dim)
        seg_idx_t = torch.tensor(query_seg_idx, dtype=torch.long)
        dist = torch.norm(encs - targets[seg_idx_t], dim=-1)  # (n_queries,)
        all_dists.append(dist)

        if count % 100 == 0:
            print(f"episode {count}/{len(ep_indices)} (ep_idx={ep_idx}) done", flush=True)

    all_dists = torch.cat(all_dists).numpy()
    p25, p50, p75 = np.percentile(all_dists, [25, 50, 75])
    print(
        f"n_samples={len(all_dists)} mean={all_dists.mean():.4f} std={all_dists.std():.4f} "
        f"p25={p25:.4f} p50={p50:.4f} p75={p75:.4f}",
        flush=True,
    )

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(
            {
                "n_samples": int(len(all_dists)),
                "mean": float(all_dists.mean()),
                "std": float(all_dists.std()),
                "max_k": args.max_k,
                "min_gap": args.min_gap,
                "epsilon_candidates": {"p25": float(p25), "p50": float(p50), "p75": float(p75)},
            },
            f,
            indent=2,
        )
    print(f"saved {args.out_path}", flush=True)


if __name__ == "__main__":
    main()
