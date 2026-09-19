# HWM_PLDM 원본 대비 추가된 실험 코드 (fork: slcnvly/HWM_PLDM, branch: causal-waypoint-student).
# Kaggle GPU에서 실행. Offline 단계 1: causal online student(Δτ 예측기)를 학습시킬
# (features, label) 쌍을 전체 학습 데이터셋(모든 궤적)에 대해 만든다.
#
# 재사용 원칙: 모델 로딩/forward/에러 계산은 variance_head.py의
# compute_error_and_encoding_series를 그대로 쓴다 -- 새로 만들지 않음.
# pick_changepoints(segmentation.py)도 그대로 재사용해서 라벨(다음 waypoint까지
# 남은 스텝 수)의 "정답"을 만든다. 이 라벨은 RAW error 기반 pick_changepoints
# 결과다 -- eval_comparison.py의 (b) oracle과 동일한 정의(surprise 정규화 버전
# 아님)라서, causal student는 "raw-error 기반 오프라인 oracle을 causal하게
# 근사하는 것"이 목표가 된다.
#
# 인덱스 규약 (실시간 t 기준, t=1..T-1, T=60):
#   e_t          = encodings_before[t]      (frame t 도착 직후의 상태)
#   e_{t-1}      = encodings_before[t-1]
#   a_{t-1}      = actions[t-1]             (t-1 -> t 전이에 쓰인 행동)
#   err_t        = err[t-1]                 (그 전이의 실제 예측오차, t 도착 시 드러남)
#   surprise_t   = err_t^2 / (sigma^2(e_{t-1}) + eps)   (SS8과 동일 공식)
# t=T(=60)는 e_T가 encodings_before 범위 밖이라 제외 (경계적 케이스라 스킵해도
# 정보 손실 거의 없음 -- Δτ=0인 자명한 지점이기도 함).
#
# Δτ(t) = 다음 waypoint까지 남은 스텝. boundaries 중 t 이상인 것 중 최솟값 - t,
# 없으면(마지막 waypoint 이후) 윈도우 끝까지 남은 스텝 T-t (버리지 않고 유효한
# 큰 값으로 남겨서 "당분간 waypoint 없음"이라는 정보를 회귀 타깃에 그대로 반영).
#
# 저장 형식: 에피소드당 encodings_before를 한 번만 저장(e_t/e_{t-1} 쌍으로
# 중복 저장하면 용량이 2배가 됨 -- 학습 스크립트가 인접 인덱스로 슬라이싱해서
# 페어를 구성). train/val은 에피소드 단위로 분할(고정 시드) -- 같은 궤적의
# 스텝들이 train/val에 걸쳐 섞이는 leakage 방지.
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from pldm_envs.diverse_maze.adaptive_waypoints.segmentation import pick_changepoints  # noqa: E402
from pldm_envs.diverse_maze.adaptive_waypoints.variance_head import (  # noqa: E402
    VarianceHead,
    compute_error_and_encoding_series,
    load_variance_head_for_l1only,
    standardize,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="r50 main split dir")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True, help="L1-only frozen checkpoint")
    parser.add_argument("--vhead_checkpoint_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="data/student_labels")
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--min_seg", type=int, default=8)
    parser.add_argument("--n_boundaries", type=int, default=5)
    parser.add_argument("--l2_step_skip", type=int, default=10)
    parser.add_argument("--l2_n_steps", type=int, default=6)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--limit_episodes", type=int, default=None, help="debug: cap episode count")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    print(f"device={device}", flush=True)

    l1_model = load_variance_head_for_l1only(args.config_path, args.checkpoint_path, device)

    ckpt = torch.load(args.vhead_checkpoint_path, map_location="cpu", weights_only=False)
    input_mean, input_std, target_scale = ckpt["input_mean"], ckpt["input_std"], ckpt["target_scale"]
    mlp = VarianceHead(input_mean.shape[0], hidden=128)
    mlp.load_state_dict(ckpt["model_state"])
    mlp.to(device).eval()
    repr_dim = input_mean.shape[0]
    print(f"repr_dim={repr_dim}", flush=True)

    window = args.l2_n_steps * args.l2_step_skip + 1  # 61
    T = window - 1  # 60

    splits = torch.load(os.path.join(args.data_path, "data.p"), weights_only=False)
    images = np.load(os.path.join(args.data_path, "images.npy"), mmap_mode="r")
    cum_lengths = np.cumsum([len(s["observations"]) for s in splits])

    eligible = [i for i in range(len(splits)) if len(splits[i]["observations"]) >= window]
    if args.limit_episodes is not None:
        eligible = eligible[: args.limit_episodes]
    print(f"{len(eligible)} eligible episodes", flush=True)

    rng = np.random.RandomState(args.split_seed)
    shuffled = rng.permutation(eligible)
    n_val = max(1, int(len(shuffled) * args.val_frac))
    val_eps = set(shuffled[:n_val].tolist())
    print(f"train episodes={len(shuffled) - n_val}, val episodes={n_val}", flush=True)

    buffers = {
        "train": {"encodings": [], "err": [], "surprise": [], "actions": [], "boundaries": [], "ep_idx": []},
        "val": {"encodings": [], "err": [], "surprise": [], "actions": [], "boundaries": [], "ep_idx": []},
    }

    for count, ep_idx in enumerate(eligible):
        start = 0 if ep_idx == 0 else cum_lengths[ep_idx - 1]
        obs = splits[ep_idx]["observations"][:window]
        proprio_vel = torch.from_numpy(obs[:, 2:4]).float().unsqueeze(1).to(device)
        img_seq = torch.from_numpy(np.array(images[start : start + window])).float().permute(0, 3, 1, 2)
        states = img_seq.unsqueeze(1).to(device)
        actions = torch.from_numpy(splits[ep_idx]["actions"][: window - 1]).float().unsqueeze(1).to(device)

        err, enc = compute_error_and_encoding_series(l1_model, states, actions, proprio_vel)
        enc = enc.flatten(1)  # (T, repr_dim) -- encodings_before, e_0..e_{T-1}
        with torch.no_grad():
            enc_std = standardize(enc.half(), input_mean, input_std).to(device)
            log_var_norm = mlp(enc_std)
            var = torch.exp(log_var_norm).cpu() * target_scale
        surprise = err.pow(2) / (var + args.eps)

        boundaries = pick_changepoints(err, n_boundaries=args.n_boundaries, min_seg=args.min_seg)

        split = "val" if ep_idx in val_eps else "train"
        buf = buffers[split]
        buf["encodings"].append(enc.cpu().half().numpy())  # (T, repr_dim) fp16
        buf["err"].append(err.numpy().astype(np.float16))
        buf["surprise"].append(surprise.numpy().astype(np.float16))
        buf["actions"].append(splits[ep_idx]["actions"][: window - 1].astype(np.float16))
        buf["boundaries"].append(np.array(boundaries, dtype=np.int16))
        buf["ep_idx"].append(ep_idx)

        if count % 100 == 0:
            print(f"episode {count}/{len(eligible)} (ep_idx={ep_idx}, split={split}) done", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    for split, buf in buffers.items():
        if len(buf["ep_idx"]) == 0:
            print(f"WARNING: 0 episodes in {split} split, skipping save", flush=True)
            continue
        out_path = os.path.join(args.out_dir, f"{split}.npz")
        np.savez(
            out_path,
            encodings=np.stack(buf["encodings"]),  # (N, T, repr_dim) fp16
            err=np.stack(buf["err"]),  # (N, T)
            surprise=np.stack(buf["surprise"]),  # (N, T)
            actions=np.stack(buf["actions"]),  # (N, T, 2)
            boundaries=np.stack(buf["boundaries"]),  # (N, n_boundaries)
            ep_idx=np.array(buf["ep_idx"], dtype=np.int32),
            T=np.array(T),
            repr_dim=np.array(repr_dim),
            min_seg=np.array(args.min_seg),
            n_boundaries=np.array(args.n_boundaries),
            eps=np.array(args.eps),
        )
        size_gb = os.path.getsize(out_path) / 1e9
        print(f"saved {out_path} ({size_gb:.2f} GB): {len(buf['ep_idx'])} episodes", flush=True)


if __name__ == "__main__":
    main()
