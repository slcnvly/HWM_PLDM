# HWM_PLDM 원본 대비 추가된 실험 코드 (fork: slcnvly/HWM_PLDM, branch: causal-waypoint-student).
# Offline 단계 3: eval/causal_waypoint_trigger.py의 CausalWaypointTrigger를
# 학습된 student로 "재생(replay)"해서, 학습 데이터의 모든 궤적에 대해
# changepoints_minseg{min_seg}.pt와 정확히 같은 포맷({ep_idx: [b1..b5]})의
# 파일을 만든다 -- level2 파인튜닝이 이 파일을 그대로 읽을 수 있게 하기 위함
# (d4rl_adaptive.py는 손대지 않음).
#
# 중요: 여기서도 "재생"이지 "미래를 안다"가 아니다 -- 각 궤적의 프레임을
# 0부터 순서대로 하나씩 trigger.step()에 넣고, 그 시점까지 causal하게 계산된
# 값만으로 나온 결정을 그대로 기록한다(트리거 내부 로직은 실제 온라인 MPC에서
# 쓰이는 것과 완전히 동일한 코드 경로).
#
# 문제: pick_changepoints는 항상 정확히 n_boundaries=5개를 반환하도록 설계돼
# 있지만(고정 길이 세그먼트 구조가 필요해서), causal trigger의
# threshold+min_seg 규칙은 5개가 나온다는 보장이 없다. 그래서 여기서
# reconcile_boundaries()로 후처리한다:
#   - 정확히 5개면 그대로.
#   - 5개 초과면 confirm 시점의 Δτ̂(작을수록 확신 높음)가 가장 작은 5개만 채택.
#   - 5개 미만이면 남은 슬롯을 segmentation.py::pick_changepoints의 fallback과
#     동일한 균등 배치 로직으로 채운다 (이 부분만 이미 있는 fallback을
#     그대로 재현 -- 새 알고리즘 아님).
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from eval.causal_waypoint_trigger import CausalWaypointTrigger  # noqa: E402


def reconcile_boundaries(confirmed, T, n_boundaries=5, min_seg=8):
    """confirmed: list of (t, delta_tau_hat) pairs with 0 < t < T, already
    respecting min_seg spacing (the trigger enforces this online via
    steps_since_waypoint), but re-checked here defensively."""
    ts_sorted = sorted(confirmed, key=lambda p: p[0])
    filtered = []
    for t, score in ts_sorted:
        if not filtered or t - filtered[-1][0] >= min_seg:
            filtered.append((t, score))

    if len(filtered) == n_boundaries:
        return [t for t, _ in filtered]

    if len(filtered) > n_boundaries:
        ranked = sorted(filtered, key=lambda p: p[1])  # smallest delta_tau_hat = most confident
        keep = sorted(t for t, _ in ranked[:n_boundaries])
        return keep

    # fewer than n_boundaries: fall back to uniform spacing for missing slots
    # (mirrors segmentation.py::pick_changepoints's own fallback exactly)
    chosen = [t for t, _ in filtered]
    uniform = [round(T * (i + 1) / (n_boundaries + 1)) for i in range(n_boundaries)]
    for u in uniform:
        if len(chosen) >= n_boundaries:
            break
        if all(abs(u - c) >= min_seg for c in chosen) and 0 < u < T:
            chosen.append(u)
    chosen = sorted(set(chosen))[:n_boundaries]
    while len(chosen) < n_boundaries:
        chosen.append(min(T - 1, (chosen[-1] if chosen else 0) + 1))
    return sorted(chosen)


def replay_episode(trigger, images, start, window, obs, actions, device):
    trigger.reset()
    confirmed = []
    for j in range(window):
        obs_j = torch.from_numpy(np.array(images[start + j : start + j + 1])).float().permute(0, 3, 1, 2).to(device)
        proprio_j = torch.from_numpy(obs[j : j + 1, 2:4]).float().to(device)
        action_arg = torch.from_numpy(actions[j - 1 : j] if j >= 1 else actions[0:1]).float().to(device)

        result = trigger.step(obs_j, action_arg, proprio_j)
        if result is None:
            continue
        is_wp, delta_tau_hat = result
        if bool(is_wp[0]):
            confirmed.append((j, float(delta_tau_hat[0])))
    return confirmed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--vhead_checkpoint_path", type=str, required=True)
    parser.add_argument("--student_checkpoint_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--min_seg", type=int, default=8)
    parser.add_argument("--n_boundaries", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--l2_step_skip", type=int, default=10)
    parser.add_argument("--l2_n_steps", type=int, default=6)
    parser.add_argument("--limit_episodes", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    print(f"device={device}", flush=True)

    trigger = CausalWaypointTrigger(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        vhead_checkpoint_path=args.vhead_checkpoint_path,
        student_checkpoint_path=args.student_checkpoint_path,
        normalizer=None,  # replaying raw recorded images.npy directly, see trigger docstring
        device=device,
        min_seg=args.min_seg,
        threshold=args.threshold,
    )

    window = args.l2_n_steps * args.l2_step_skip + 1  # 61
    T = window - 1  # 60

    splits = torch.load(os.path.join(args.data_path, "data.p"), weights_only=False)
    images = np.load(os.path.join(args.data_path, "images.npy"), mmap_mode="r")
    cum_lengths = np.cumsum([len(s["observations"]) for s in splits])

    eligible = [i for i in range(len(splits)) if len(splits[i]["observations"]) >= window]
    if args.limit_episodes is not None:
        eligible = eligible[: args.limit_episodes]
    print(f"{len(eligible)} eligible episodes", flush=True)

    boundaries_by_ep = {}
    n_needed_fallback = 0
    for count, ep_idx in enumerate(eligible):
        start = 0 if ep_idx == 0 else cum_lengths[ep_idx - 1]
        obs = splits[ep_idx]["observations"][:window]
        actions = splits[ep_idx]["actions"][: window - 1]

        confirmed = replay_episode(trigger, images, start, window, obs, actions, device)
        confirmed = [(t, s) for t, s in confirmed if 0 < t < T]
        n_raw_confirmed = len(confirmed)
        boundaries = reconcile_boundaries(confirmed, T, args.n_boundaries, args.min_seg)
        if n_raw_confirmed != args.n_boundaries:
            n_needed_fallback += 1
        boundaries_by_ep[ep_idx] = boundaries

        if count % 100 == 0:
            print(
                f"episode {count}/{len(eligible)} (ep_idx={ep_idx}): "
                f"{n_raw_confirmed} raw confirms -> {boundaries}",
                flush=True,
            )

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    torch.save(boundaries_by_ep, args.out_path)
    print(
        f"saved {args.out_path}: {len(boundaries_by_ep)} episodes "
        f"({n_needed_fallback} needed reconciliation, "
        f"{len(boundaries_by_ep) - n_needed_fallback} had exactly {args.n_boundaries} raw confirms)",
        flush=True,
    )


if __name__ == "__main__":
    main()
