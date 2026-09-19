# HWM_PLDM 원본 대비 추가된 실험 코드 (fork: slcnvly/HWM_PLDM, branch: causal-waypoint-student).
# Offline 단계 2: student_labels.py가 만든 data/student_labels/{train,val}.npz로
# causal Δτ(다음 waypoint까지 남은 스텝) 회귀 모델을 학습한다.
#
# 입력 feature (전부 실시간 t 시점에 causal하게 구할 수 있는 값만):
#   standardize(e_t) ++ standardize(e_{t-1}) ++ standardize(a_{t-1}) ++
#   standardize(surprise[t-K:t])   (K=8, min_seg=8과 맞춤 -- 최근 한 segment
#   분량의 surprise 이력을 보는 게 "다음 waypoint가 언제 올지" 판단에
#   자연스러운 스케일이라는 근거)
#
# 두 타깃 파라메터화를 둘 다 학습해서 val MAE(원 스텝 단위)가 더 낮은 쪽을
# 채택한다: (a) Δτ 직접 회귀 (MSE), (b) log1p(Δτ) 회귀 (MSE, Δτ>=0이라 로그
# 변환이 타깃 분포를 O(1) 스케일로 눌러줄 수 있어 시도할 가치가 있음 -- 다만
# Δτ 자체가 min_seg~T(=8~60) 범위로 이미 좁아서 log 변환 이득이 크지 않을
# 수도 있음, 그래서 "시도 후 채택" 원칙).
#
# variance_head.py의 fit_standardization/standardize를 그대로 재사용 (encoding
# 표준화 로직 중복 구현 안 함).
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from pldm_envs.diverse_maze.adaptive_waypoints.variance_head import (  # noqa: E402
    fit_standardization,
    standardize,
)

K_SURPRISE_HISTORY = 8  # matches min_seg=8 (see module docstring)


class StudentModel(nn.Module):
    """2-hidden-layer MLP, plain scalar regression output (no log-var head --
    unlike VarianceHead, this predicts Delta-tau directly, not a variance)."""

    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_delta_tau(boundaries: np.ndarray, T: int) -> np.ndarray:
    """boundaries: (n_boundaries,) sorted or unsorted ints in (0,T). Returns
    (T,) array: delta_tau[t] = steps until the next boundary >= t, or T-t if
    none remain (informative "no more waypoints in this window" target, not
    dropped)."""
    b = np.sort(boundaries)
    delta = np.zeros(T, dtype=np.float32)
    for t in range(T):
        candidates = b[b >= t]
        delta[t] = (candidates.min() - t) if candidates.size > 0 else (T - t)
    return delta


def build_pair_index(n_episodes: int, T: int, K: int):
    """Valid (ep, t) pairs: t in [1, T-1] (need e_{t-1}, and e_t itself must
    be within encodings_before's range 0..T-1 -- see student_labels.py's
    module docstring for why t=T is excluded)."""
    eps, ts = np.meshgrid(np.arange(n_episodes), np.arange(1, T), indexing="ij")
    return eps.flatten(), ts.flatten()


def gather_batch(encodings, surprise, actions, delta_tau, ep_idx, t_idx, K, device):
    """encodings: (N_ep,T,repr_dim) fp16 tensor. surprise/actions/delta_tau
    precomputed per-episode arrays. ep_idx/t_idx: (B,) index arrays for this
    batch. Returns raw (unstandardized) feature blocks + raw target, all on
    `device`."""
    e_t = encodings[ep_idx, t_idx].to(device).float()  # (B, repr_dim)
    e_prev = encodings[ep_idx, t_idx - 1].to(device).float()  # (B, repr_dim)
    a_prev = actions[ep_idx, t_idx - 1].to(device).float()  # (B, 2)

    B = len(ep_idx)
    T = surprise.shape[1]
    hist = torch.zeros(B, K, dtype=torch.float32, device=device)
    for k in range(K):
        src_t = t_idx - K + k  # may be negative -> left-zero-pad
        valid = src_t >= 0
        if valid.any():
            rows = np.nonzero(valid)[0]
            hist[rows, k] = surprise[ep_idx[rows], src_t[rows]].to(device).float()

    target = delta_tau[ep_idx, t_idx].to(device).float()  # (B,)
    return e_t, e_prev, a_prev, hist, target


def make_features(e_t, e_prev, a_prev, hist, enc_mean, enc_std, act_mean, act_std, surp_mean, surp_std):
    e_t_std = standardize(e_t, enc_mean, enc_std)
    e_prev_std = standardize(e_prev, enc_mean, enc_std)
    a_std = (a_prev - act_mean) / act_std
    h_std = (hist - surp_mean) / surp_std
    return torch.cat([e_t_std, e_prev_std, a_std, h_std], dim=1)


def load_split(path, device_for_small_arrays="cpu"):
    d = np.load(path)
    encodings = torch.from_numpy(d["encodings"])  # keep fp16, (N,T,repr_dim), stays CPU (large)
    surprise = torch.from_numpy(d["surprise"].astype(np.float32))
    actions = torch.from_numpy(d["actions"].astype(np.float32))
    boundaries = d["boundaries"]
    T = int(d["T"])
    n_ep = encodings.shape[0]
    delta_tau = np.stack([compute_delta_tau(boundaries[i], T) for i in range(n_ep)])
    delta_tau = torch.from_numpy(delta_tau)
    return {
        "encodings": encodings,
        "surprise": surprise,
        "actions": actions,
        "delta_tau": delta_tau,
        "T": T,
        "repr_dim": int(d["repr_dim"]),
        "n_ep": n_ep,
    }


def run_epoch(model, data, ep_idx_all, t_idx_all, stats, K, device, batch_size, opt=None, target_transform="linear"):
    """opt=None -> eval mode, no grad, no shuffle needed for a full pass."""
    train_mode = opt is not None
    model.train(train_mode)
    n = len(ep_idx_all)
    perm = np.random.permutation(n) if train_mode else np.arange(n)
    total_loss, total_mae_raw, count = 0.0, 0.0, 0

    for i in range(0, n, batch_size):
        idx = perm[i : i + batch_size]
        ep_b, t_b = ep_idx_all[idx], t_idx_all[idx]
        e_t, e_prev, a_prev, hist, target_raw = gather_batch(
            data["encodings"], data["surprise"], data["actions"], data["delta_tau"], ep_b, t_b, K, device
        )
        feat = make_features(e_t, e_prev, a_prev, hist, *stats)

        if target_transform == "log1p":
            target = torch.log1p(target_raw)
        else:
            target = target_raw

        pred = model(feat)
        loss = nn.functional.mse_loss(pred, target)

        if train_mode:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        with torch.no_grad():
            pred_raw = torch.expm1(pred) if target_transform == "log1p" else pred
            mae_raw = (pred_raw - target_raw).abs().mean().item()

        bs = len(idx)
        total_loss += loss.item() * bs
        total_mae_raw += mae_raw * bs
        count += bs

    return total_loss / count, total_mae_raw / count


def train_one_variant(train_data, val_data, stats, K, device, target_transform, epochs, batch_size, lr, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    input_dim = 2 * train_data["repr_dim"] + 2 + K
    model = StudentModel(input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_ep, train_t = build_pair_index(train_data["n_ep"], train_data["T"], K)
    val_ep, val_t = build_pair_index(val_data["n_ep"], val_data["T"], K)

    log = []
    best_val_mae, best_epoch = float("inf"), -1
    start_epoch = 0

    # resume from latest per-epoch checkpoint if present (survives a session interruption)
    existing = sorted(
        [f for f in os.listdir(ckpt_dir) if f.startswith("epoch_") and f.endswith(".pt")],
        key=lambda f: int(f.split("_")[1].split(".")[0]),
    )
    if existing:
        last = existing[-1]
        state = torch.load(os.path.join(ckpt_dir, last), map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        opt.load_state_dict(state["opt_state"])
        start_epoch = state["epoch"] + 1
        log = state.get("log", [])
        best_val_mae = state.get("best_val_mae", float("inf"))
        best_epoch = state.get("best_epoch", -1)
        print(f"[{target_transform}] resumed from epoch {state['epoch']} ({last})", flush=True)

    for epoch in range(start_epoch, epochs):
        train_loss, train_mae = run_epoch(
            model, train_data, train_ep, train_t, stats, K, device, batch_size, opt, target_transform
        )
        val_loss, val_mae = run_epoch(
            model, val_data, val_ep, val_t, stats, K, device, batch_size, opt=None, target_transform=target_transform
        )
        print(
            f"[{target_transform}] epoch {epoch}: train_loss={train_loss:.4f} train_mae={train_mae:.3f} "
            f"val_loss={val_loss:.4f} val_mae={val_mae:.3f}",
            flush=True,
        )
        log.append({"epoch": epoch, "train_loss": train_loss, "train_mae": train_mae, "val_loss": val_loss, "val_mae": val_mae})

        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae, best_epoch = val_mae, epoch

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "log": log,
                "best_val_mae": best_val_mae,
                "best_epoch": best_epoch,
                "target_transform": target_transform,
                "input_dim": input_dim,
            },
            os.path.join(ckpt_dir, f"epoch_{epoch}.pt"),
        )
        if is_best:
            torch.save(
                {"model_state": model.state_dict(), "target_transform": target_transform, "input_dim": input_dim},
                os.path.join(ckpt_dir, "best.pt"),
            )

    return {"best_val_mae": best_val_mae, "best_epoch": best_epoch, "log": log}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_dir", type=str, default="data/student_labels")
    parser.add_argument("--out_dir", type=str, default="checkpoints/student")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    print(f"device={device}", flush=True)

    train_data = load_split(os.path.join(args.labels_dir, "train.npz"))
    val_data = load_split(os.path.join(args.labels_dir, "val.npz"))
    print(f"train episodes={train_data['n_ep']}, val episodes={val_data['n_ep']}", flush=True)

    K = K_SURPRISE_HISTORY
    enc_mean, enc_std = fit_standardization(train_data["encodings"].flatten(0, 1))
    act_flat = train_data["actions"].flatten(0, 1)
    act_mean, act_std = act_flat.mean(dim=0), act_flat.std(dim=0).clamp_min(1e-6)
    surp_flat = train_data["surprise"].flatten()
    surp_mean, surp_std = surp_flat.mean(), surp_flat.std().clamp_min(1e-6)
    stats = (enc_mean.to(device), enc_std.to(device), act_mean.to(device), act_std.to(device), surp_mean.to(device), surp_std.to(device))

    results = {}
    for transform in ("linear", "log1p"):
        print(f"\n===== training variant: {transform} =====", flush=True)
        ckpt_dir = os.path.join(args.out_dir, transform)
        results[transform] = train_one_variant(
            train_data, val_data, stats, K, device, transform, args.epochs, args.batch_size, args.lr, ckpt_dir
        )

    winner = min(results, key=lambda k: results[k]["best_val_mae"])
    print(f"\nwinner: {winner} (val_mae={results[winner]['best_val_mae']:.3f} steps)", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    # copy the winning variant's best checkpoint to a stable top-level path,
    # plus save standardization stats alongside it (needed by the online trigger)
    winner_state = torch.load(os.path.join(args.out_dir, winner, "best.pt"), map_location="cpu", weights_only=False)
    torch.save(
        {
            **winner_state,
            "enc_mean": enc_mean, "enc_std": enc_std,
            "act_mean": act_mean.cpu(), "act_std": act_std.cpu(),
            "surp_mean": surp_mean.cpu(), "surp_std": surp_std.cpu(),
            "K": K,
        },
        os.path.join(args.out_dir, "student_final.pt"),
    )
    with open(os.path.join(args.out_dir, "train_summary.json"), "w") as f:
        json.dump({"winner": winner, "results": {k: {"best_val_mae": v["best_val_mae"], "best_epoch": v["best_epoch"]} for k, v in results.items()}, "full_log": results}, f, indent=2)
    print(f"saved {os.path.join(args.out_dir, 'student_final.pt')}", flush=True)


if __name__ == "__main__":
    main()
