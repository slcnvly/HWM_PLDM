# HWM_PLDM 원본 대비 추가된 실험 코드 (fork: slcnvly/HWM_PLDM, branch: adaptive-waypoints).
# Kaggle GPU 노트북에서 실행. waypoint 배치 진단(analysis/waypoint_diagnostics.py,
# 로컬 WSL에서 실행, GPU/torch/모델 의존성 없음)을 위해 per-step surprise score
# 시계열 + 현재 min_seg=8 기준 waypoint boundary를 뽑아 outputs/embeddings.npz로
# 저장한다.
#
# 신호: raw one-step prediction error(||encoding_t - predicted_t||^2)가 아니라
# RESULTS.md SS8에서 검증된 surprise = err^2 / (predicted_variance + eps) 를 쓴다
# (사용자 확인: 최근 실험 라인, surprise_minseg8_finetuned가 이 신호로 학습됨).
# 계산 로직은 experiments/kaggle_surprise_finetune/run.py의 STAGE 1 전처리
# 블록을 그대로 재사용 -- pick_changepoints/모델 로딩 어느 것도 새로 만들지 않음.
#
# 전체 main split(~1250 에피소드, 75000 step)을 다 뽑지 않고 고정 시드로
# n_episodes개만 무작위 샘플링 -- 진단 목적엔 충분하고 추출 시간/다운로드
# 용량을 크게 줄인다 (200 에피소드 x 60 step x float16 두 배열 = 몇십 KB 수준).
#
# 사용 예 (Kaggle 커널 내부, r50 main 데이터셋 + L1-only 체크포인트 +
# variance_head_v2.pt가 이미 마운트되어 있다고 가정):
#   python kaggle/extract_embeddings.py \
#       --data_path /kaggle/input/hwm-r50-dataset/main \
#       --config_path pldm/configs/diverse_maze/icml/large_diverse_25maps_l2.yaml \
#       --checkpoint_path /kaggle/input/.../l1only.ckpt \
#       --vhead_checkpoint_path /kaggle/input/.../variance_head_v2.pt \
#       --out_path outputs/embeddings.npz \
#       --n_episodes 200 --seed 0
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pldm_envs.diverse_maze.adaptive_waypoints.segmentation import (  # noqa: E402
    pick_changepoints,
)
from pldm_envs.diverse_maze.adaptive_waypoints.variance_head import (  # noqa: E402
    VarianceHead,
    compute_error_and_encoding_series,
    load_variance_head_for_l1only,
    standardize,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True, help="L1-only frozen checkpoint")
    parser.add_argument("--vhead_checkpoint_path", type=str, required=True, help="trained variance_head_v2.pt")
    parser.add_argument("--out_path", type=str, default="outputs/embeddings.npz")
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_seg", type=int, default=8)
    parser.add_argument("--n_boundaries", type=int, default=5)
    parser.add_argument("--l2_step_skip", type=int, default=10)
    parser.add_argument("--l2_n_steps", type=int, default=6)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    print(f"device={device}", flush=True)

    l1_model = load_variance_head_for_l1only(args.config_path, args.checkpoint_path, device)

    # map_location="cpu": standardize()'s convention is CPU-resident
    # input_mean/input_std, .to(device) applied after standardizing (same as
    # kaggle_surprise_finetune/run.py's STAGE 1 -- avoids the v1 CPU/CUDA
    # mismatch documented in RESULTS.md SS8.5).
    ckpt = torch.load(args.vhead_checkpoint_path, map_location="cpu", weights_only=False)
    input_mean, input_std, target_scale = ckpt["input_mean"], ckpt["input_std"], ckpt["target_scale"]
    mlp = VarianceHead(input_mean.shape[0], hidden=128)
    mlp.load_state_dict(ckpt["model_state"])
    mlp.to(device).eval()

    window = args.l2_n_steps * args.l2_step_skip + 1  # 61 for defaults
    T = window - 1  # 60: per-step error/surprise series length, fixed per episode

    splits = torch.load(os.path.join(args.data_path, "data.p"), weights_only=False)
    images = np.load(os.path.join(args.data_path, "images.npy"), mmap_mode="r")
    cum_lengths = np.cumsum([len(s["observations"]) for s in splits])

    eligible = [i for i in range(len(splits)) if len(splits[i]["observations"]) >= window]
    print(f"{len(eligible)}/{len(splits)} episodes long enough for a full {window}-step window", flush=True)

    rng = np.random.RandomState(args.seed)
    n_sample = min(args.n_episodes, len(eligible))
    sampled = sorted(rng.choice(eligible, size=n_sample, replace=False).tolist())
    print(f"sampling {n_sample} episodes (seed={args.seed})", flush=True)

    all_surprise = np.zeros((n_sample, T), dtype=np.float16)
    all_err = np.zeros((n_sample, T), dtype=np.float16)
    all_boundaries = np.zeros((n_sample, args.n_boundaries), dtype=np.int16)
    kept_ep_indices = np.zeros((n_sample,), dtype=np.int32)

    for row, ep_idx in enumerate(sampled):
        start = 0 if ep_idx == 0 else cum_lengths[ep_idx - 1]
        obs = splits[ep_idx]["observations"][:window]
        proprio_vel = torch.from_numpy(obs[:, 2:4]).float().unsqueeze(1).to(device)
        img_seq = torch.from_numpy(
            np.array(images[start : start + window])
        ).float().permute(0, 3, 1, 2)
        states = img_seq.unsqueeze(1).to(device)
        actions = torch.from_numpy(
            splits[ep_idx]["actions"][: window - 1]
        ).float().unsqueeze(1).to(device)

        err, enc = compute_error_and_encoding_series(l1_model, states, actions, proprio_vel)
        enc = enc.flatten(1)  # (T, *repr_shape) -> (T, repr_dim)
        with torch.no_grad():
            enc_std = standardize(enc.half(), input_mean, input_std).to(device)
            log_var_norm = mlp(enc_std)
            var = torch.exp(log_var_norm).cpu() * target_scale
        surprise = err.pow(2) / (var + args.eps)
        boundaries = pick_changepoints(surprise, n_boundaries=args.n_boundaries, min_seg=args.min_seg)

        all_surprise[row] = surprise.numpy().astype(np.float16)
        all_err[row] = err.numpy().astype(np.float16)
        all_boundaries[row] = np.array(boundaries, dtype=np.int16)
        kept_ep_indices[row] = ep_idx

        if row % 50 == 0:
            print(f"episode {row}/{n_sample} (ep_idx={ep_idx}) done", flush=True)

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    np.savez_compressed(
        args.out_path,
        surprise_scores=all_surprise,
        raw_errors=all_err,
        boundaries=all_boundaries,
        episode_indices=kept_ep_indices,
        T=np.array(T),
        min_seg=np.array(args.min_seg),
        n_boundaries=np.array(args.n_boundaries),
        eps=np.array(args.eps),
        seed=np.array(args.seed),
        n_episodes_total=np.array(len(splits)),
    )
    size_kb = os.path.getsize(args.out_path) / 1024
    print(f"saved {args.out_path} ({size_kb:.1f} KB): {n_sample} episodes x {T} steps", flush=True)


if __name__ == "__main__":
    main()
