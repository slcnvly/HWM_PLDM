# HWM_PLDM 원본 대비 추가된 실험 코드 (fork: slcnvly/HWM_PLDM, branch: adaptive-waypoints).
# See DESIGN.md and RESULTS.md SS8 for the full rationale.
#
# Post-hoc variance head: an approximation to "degree of surprise" for
# changepoint scoring, standing in for ensemble variance / MC-dropout
# variance -- both unavailable without retraining frozen level1
# (ensemble_size=1, dropout=0.0 in every checkpoint this project has; see
# RESULTS.md SS8 for the code-level confirmation of why those paths are
# closed). Trains a small MLP (or, as a training-free alternative, a kNN
# regressor) to predict, from the frozen level1 latent state, the expected
# squared magnitude of compute_changepoints.py's existing raw one-step
# prediction error at that state.
#
# v2 (this file): the v1 approach (softplus output, plain Gaussian NLL,
# standard init) failed -- converged to a near-constant sigma^2 ~40x too
# large (889 vs the true ~22.6), worse than the trivial constant-mean
# baseline, because standard Gaussian NLL's gradient w.r.t. variance is
# proportional to 1/sigma^2 and vanishes once sigma^2 overshoots into a
# too-large region (see RESULTS.md SS8 for the full failure writeup). Fixed
# here via: log-variance parameterization with an init that starts exactly
# at the constant-optimal solution; standardized inputs / normalized
# targets; beta-NLL (Seitzer et al. 2022, beta=0.5) so the loss weights
# high-variance samples by stopgrad(sigma^2)^beta instead of letting their
# gradient vanish; lower LR + gradient clipping + best-val-NLL checkpoint
# selection instead of returning the final epoch.
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pldm_envs.diverse_maze.adaptive_waypoints.compute_changepoints import load_level1


def compute_error_and_encoding_series(model, states, actions, proprio_vel):
    """Same forward pass as compute_changepoints.compute_error_series, but
    also returns the PRE-transition latent state (the state the one-step
    prediction was made FROM, not the state it predicted) aligned 1:1 with
    each error value -- this is the (x, y) pair the variance head trains on.

    Returns:
        err: (T-1,) per-step squared prediction error (same as
            compute_error_series).
        encodings_before: (T-1, *repr_shape) the latent state at each step
            i, from which the transition to step i+1 (scored by err[i]) was
            predicted.
    """
    with torch.no_grad():
        result = model.level1.forward_posterior(
            states, actions, proprio_vel=proprio_vel, encode_only=False
        )
    full_encodings = result.backbone_output.encodings  # (T,1,...)
    encodings_after = full_encodings[1:]
    encodings_before = full_encodings[:-1]
    predictions = result.pred_output.predictions[1:]
    err = (encodings_after - predictions).pow(2).flatten(2).mean(dim=-1)  # (T-1,1)
    return err.squeeze(1).cpu(), encodings_before.squeeze(1).cpu()  # (T-1,), (T-1,*repr_shape)


def build_dataset(
    model,
    data_path: str,
    device: str,
    l2_step_skip: int = 10,
    l2_n_steps: int = 6,
    limit_episodes: int = None,
):
    """Same per-episode windowing as compute_changepoints.py's main() (first
    `window` frames of each episode, skip episodes shorter than that) --
    reuses the exact same population of per-step errors changepoint
    computation already implicitly scores over, just also keeping the
    aligned latent states instead of discarding them.

    Returns:
        errs: (N,) all per-step errors, concatenated across episodes.
        encodings: (N, *repr_shape) aligned pre-transition latent states.
        ep_boundaries: list of (start, end) index ranges into errs/encodings
            for each kept episode, in order -- needed later to re-run
            per-episode peak-picking with the new score.
    """
    splits = torch.load(f"{data_path}/data.p", weights_only=False)
    images = np.load(f"{data_path}/images.npy", mmap_mode="r")

    window = l2_n_steps * l2_step_skip + 1
    cum_lengths = np.cumsum([len(s["observations"]) for s in splits])
    n_episodes = len(splits) if limit_episodes is None else min(limit_episodes, len(splits))

    all_errs, all_encs, ep_boundaries, ep_indices = [], [], [], []
    cursor = 0
    for ep_idx in range(n_episodes):
        ep_len = len(splits[ep_idx]["observations"])
        if ep_len < window:
            continue
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

        err, enc = compute_error_and_encoding_series(model, states, actions, proprio_vel)
        all_errs.append(err)
        all_encs.append(enc)
        ep_boundaries.append((cursor, cursor + err.shape[0]))
        ep_indices.append(ep_idx)
        cursor += err.shape[0]

        if ep_idx % 200 == 0:
            print(f"[dataset build: {data_path}] episode {ep_idx}/{n_episodes}", flush=True)

    errs = torch.cat(all_errs)
    encodings = torch.cat(all_encs).flatten(1)  # (N, repr_dim)
    return errs, encodings, ep_boundaries, ep_indices


# ---------------------------------------------------------------------------
# Standardization (fit on train only, applied to both train/val -- no leakage)
# ---------------------------------------------------------------------------


def fit_standardization(train_encs: torch.Tensor):
    mean = train_encs.mean(dim=0)
    std = train_encs.std(dim=0).clamp_min(1e-6)
    return mean, std


def standardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


# ---------------------------------------------------------------------------
# Constant-variance baseline (step A's "is this even worth doing" floor)
# ---------------------------------------------------------------------------


def constant_baseline_nll(train_err: torch.Tensor, eval_err: torch.Tensor):
    """sigma^2 fixed at train's E[err^2] (the analytically-optimal constant
    for the training set); reports its NLL on eval_err (val or train).
    Everything in raw, unnormalized err^2 units."""
    c = float((train_err.double() ** 2).mean())
    nll = 0.5 * (np.log(c) + (eval_err.double() ** 2).mean().item() / c)
    return c, float(nll)


# ---------------------------------------------------------------------------
# Trained variance head: log-variance parameterization + beta-NLL
# ---------------------------------------------------------------------------


class VarianceHead(nn.Module):
    """2-hidden-layer MLP outputting log(sigma^2) directly (no softplus --
    v1's softplus + plain-NLL combination is what produced the gradient-
    vanishing failure mode; see module docstring). Final layer initialized
    so the network's output AT INIT is exactly log_var_init (typically
    log(E[err^2]) computed from the training cache) regardless of input --
    i.e. training starts from the constant-optimal solution and only moves
    away from it where the input actually helps."""

    def __init__(self, input_dim: int, hidden: int = 128, log_var_init: float = 0.0):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.log_var_head = nn.Linear(hidden, 1)
        with torch.no_grad():
            self.log_var_head.weight.normal_(mean=0.0, std=1e-4)
            self.log_var_head.bias.fill_(log_var_init)

    def forward(self, x):
        """Returns log(sigma^2), shape (B,)."""
        h = self.trunk(x)
        return self.log_var_head(h).squeeze(-1)


def beta_nll_loss(log_var: torch.Tensor, target: torch.Tensor, beta: float = 0.5):
    """Seitzer et al. 2022: multiply the per-sample Gaussian NLL (mean fixed
    at 0, so NLL_i = 0.5*(log_var_i + target_i^2/var_i)) by
    stopgrad(var_i)^beta. Standard NLL (beta=0) has d(NLL)/d(theta) ~ 1/var
    for the variance-shaping term, which vanishes once var overshoots into a
    too-large region -- beta>0 compensates by up-weighting exactly those
    high-variance-prediction samples, without changing what the loss
    converges to (the multiplicative factor is detached)."""
    var = torch.exp(log_var)
    nll = 0.5 * (log_var + target**2 / var)
    weight = var.detach() ** beta
    return (weight * nll).mean()


def gaussian_nll_from_log_var(log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Plain (beta=0) per-batch mean NLL -- used for validation/model
    selection, never for the training gradient itself (see beta_nll_loss)."""
    var = torch.exp(log_var)
    return (0.5 * (log_var + target**2 / var)).mean()


def train_variance_head(
    train_errs: torch.Tensor,
    train_encs: torch.Tensor,
    val_errs: torch.Tensor,
    val_encs: torch.Tensor,
    device: str,
    hidden: int = 128,
    beta: float = 0.5,
    epochs: int = 30,
    batch_size: int = 1024,
    lr: float = 1e-4,
    grad_clip_norm: float = 1.0,
):
    """Standardizes inputs (fit on train), normalizes the target err^2 by
    train's global mean (keeps the loss landscape O(1)-scaled), trains via
    beta-NLL, and returns the BEST-val-NLL checkpoint (not the final epoch --
    v1's failure included a late destabilization spike it never recovered
    from). All returned/reported NLL and variance values are un-normalized
    back to raw err^2 units.

    Returns:
        model: the trained VarianceHead (raw-space predict_variance below
            handles standardization/un-normalization).
        input_mean, input_std: standardization stats (fit on train).
        target_scale: train's E[err^2], the normalization divisor.
        log: per-epoch train/val NLL (raw units) + which epoch was kept.
    """
    input_mean, input_std = fit_standardization(train_encs)
    train_encs_std = standardize(train_encs, input_mean, input_std)
    val_encs_std = standardize(val_encs, input_mean, input_std)

    target_scale = float((train_errs.double() ** 2).mean())
    # beta_nll_loss/gaussian_nll_from_log_var square `target` internally
    # (matching v1's "target=err" convention), so the value passed in here
    # is err/sqrt(target_scale) -- err is already >=0 (it's itself a mean-
    # of-squares), so this is exactly sqrt(err^2/target_scale), i.e. the
    # normalized-target convention target_norm^2 = err^2/target_scale.
    scale_sqrt = target_scale**0.5
    train_target_norm_sqrt = train_errs / scale_sqrt
    val_target_norm_sqrt = val_errs / scale_sqrt

    input_dim = train_encs.shape[1]
    log_var_init = 0.0  # normalized target has E[target_norm^2]=1 by construction -> log(1)=0
    model = VarianceHead(input_dim, hidden=hidden, log_var_init=log_var_init).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n_train = train_encs_std.shape[0]
    best_val_nll = float("inf")
    best_state = None
    best_epoch = -1
    log = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            x = train_encs_std[idx].to(device)
            y = train_target_norm_sqrt[idx].to(device)
            log_var = model(x)
            loss = beta_nll_loss(log_var, y, beta=beta)  # beta-weighted: gradient signal only
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            opt.step()

        # Plain (beta=0) NLL on train and val, both no_grad, both same
        # formula -- for a fair apples-to-apples train-vs-val comparison in
        # the log (the training step above uses the beta-weighted loss for
        # its gradient, which is NOT directly comparable to val's plain
        # NLL -- that mismatch was a bug in the first cut of this fix).
        model.eval()
        train_loss_sum, val_loss_sum = 0.0, 0.0
        n_val = val_encs_std.shape[0]
        with torch.no_grad():
            for i in range(0, n_train, batch_size):
                x = train_encs_std[i : i + batch_size].to(device)
                y = train_target_norm_sqrt[i : i + batch_size].to(device)
                log_var = model(x)
                train_loss_sum += gaussian_nll_from_log_var(log_var, y).item() * x.shape[0]
            for i in range(0, n_val, batch_size):
                x = val_encs_std[i : i + batch_size].to(device)
                y = val_target_norm_sqrt[i : i + batch_size].to(device)
                log_var = model(x)
                val_loss_sum += gaussian_nll_from_log_var(log_var, y).item() * x.shape[0]
        train_nll_norm = train_loss_sum / n_train
        val_nll_norm = val_loss_sum / n_val

        # report in raw units too: NLL in normalized space differs from raw
        # space by a constant additive term (0.5*log(target_scale)) since
        # var_raw = var_norm * target_scale -- log(var_raw)=log(var_norm)+log(target_scale),
        # and target_raw^2/var_raw = target_norm^2/var_norm (scale cancels).
        raw_offset = 0.5 * math.log(target_scale)  # plain float -- np.log would leak numpy.float64 into the JSON-dumped log
        train_nll_raw = train_nll_norm + raw_offset
        val_nll_raw = val_nll_norm + raw_offset

        log.append({
            "epoch": epoch,
            "train_nll_raw": train_nll_raw,
            "val_nll_raw": val_nll_raw,
        })
        print(
            f"[variance head] epoch {epoch}: train_nll_raw={train_nll_raw:.4f} "
            f"val_nll_raw={val_nll_raw:.4f}",
            flush=True,
        )

        if val_nll_raw < best_val_nll:
            best_val_nll = val_nll_raw
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

    model.load_state_dict(best_state)
    print(f"[variance head] kept epoch {best_epoch} (val_nll_raw={best_val_nll:.4f})", flush=True)

    return model, input_mean, input_std, target_scale, {"per_epoch": log, "best_epoch": best_epoch, "best_val_nll_raw": best_val_nll}


def predict_variance_mlp(
    model, encs: torch.Tensor, input_mean, input_std, target_scale, device: str, batch_size: int = 2048
) -> torch.Tensor:
    """Raw-units sigma^2 predictions from the trained VarianceHead, chunked
    to avoid the v1 CUDA-OOM (N x ~33k tensors are several GB moved whole)."""
    model.eval()
    encs_std = standardize(encs, input_mean, input_std)
    chunks = []
    with torch.no_grad():
        for i in range(0, encs_std.shape[0], batch_size):
            x = encs_std[i : i + batch_size].to(device)
            log_var_norm = model(x)
            var_raw = torch.exp(log_var_norm).cpu() * target_scale
            chunks.append(var_raw)
    return torch.cat(chunks)


# ---------------------------------------------------------------------------
# kNN regression: training-free alternative
# ---------------------------------------------------------------------------


def knn_predict_variance(
    train_encs_std: torch.Tensor,
    train_err_sq: torch.Tensor,
    query_encs_std: torch.Tensor,
    device: str,
    k: int = 50,
    exclude_self: bool = False,
    chunk_size: int = 1000,
) -> torch.Tensor:
    """E[err^2 | state] via k-nearest-neighbor averaging in the (already
    train-standardized) latent space. Neighbors are ALWAYS drawn from
    train_encs_std only (query_encs_std may be val, or train itself for a
    leave-one-out self-evaluation -- exclude_self=True assumes
    query_encs_std IS train_encs_std in the same row order, and masks each
    query's own index out of its own candidate neighbor set so val's
    "neighbors from train only" framing is honestly mirrored for train's
    self-evaluation too, rather than trivially returning err_sq itself at
    distance 0).

    fp16 + chunked distance matmul: train_encs_std alone is ~9GB in fp32
    (75000 x ~33k), which combined with a query chunk and the resulting
    distance matrix risks the same OOM v1 hit -- fp16 halves the resident
    train tensor and each chunk's distance matrix stays small regardless of
    chunk_size.
    """
    # Norms computed on CPU from the original fp32 tensor, in chunks --
    # `train_t.float()` on the whole fp16 GPU tensor would materialize a
    # second full-size fp32 copy (~9GB) just for this reduction, defeating
    # the point of using fp16 for train_t in the first place.
    norm_chunks = []
    for i in range(0, train_encs_std.shape[0], chunk_size):
        norm_chunks.append((train_encs_std[i : i + chunk_size].double() ** 2).sum(dim=1).float())
    train_norm_sq = torch.cat(norm_chunks).to(device)

    train_t = train_encs_std.to(device=device, dtype=torch.float16)
    train_err_sq_dev = train_err_sq.to(device)

    preds = []
    n_q = query_encs_std.shape[0]
    with torch.no_grad():
        for i in range(0, n_q, chunk_size):
            q = query_encs_std[i : i + chunk_size].to(device=device, dtype=torch.float16)
            q_norm_sq = (q.float() ** 2).sum(dim=1)
            cross = (q @ train_t.T).float()  # (chunk, n_train), fp16 matmul, fp32 accumulate view
            dist_sq = q_norm_sq[:, None] + train_norm_sq[None, :] - 2 * cross

            if exclude_self:
                rows = torch.arange(q.shape[0], device=device)
                cols = torch.arange(i, i + q.shape[0], device=device)
                dist_sq[rows, cols] = float("inf")

            topk_idx = torch.topk(dist_sq, k, largest=False, dim=1).indices  # (chunk, k)
            neighbor_vals = train_err_sq_dev[topk_idx]  # (chunk, k)
            preds.append(neighbor_vals.mean(dim=1).cpu())

    del train_t
    torch.cuda.empty_cache()
    return torch.cat(preds)


# ---------------------------------------------------------------------------
# Step B/C diagnostics: operate on a precomputed raw-units variance tensor,
# agnostic to whether it came from the MLP or the kNN estimator.
# ---------------------------------------------------------------------------


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ar = np.argsort(np.argsort(a)).astype(np.float64)
    br = np.argsort(np.argsort(b)).astype(np.float64)
    ar -= ar.mean()
    br -= br.mean()
    denom = np.sqrt((ar**2).sum() * (br**2).sum())
    return float((ar * br).sum() / denom) if denom > 0 else float("nan")


def variance_distribution_stats(var: torch.Tensor, err: torch.Tensor):
    """Step B: distribution of predicted variance (mean, quartiles, max/min
    ratio) + Spearman rank correlation between predicted variance and actual
    err^2 -- the "is this still effectively constant" check."""
    v = var.numpy()
    e_sq = (err.numpy()) ** 2

    q0, q25, q50, q75, q100 = np.percentile(v, [0, 25, 50, 75, 100])
    iqr_over_mean = float((q75 - q25) / v.mean()) if v.mean() != 0 else float("nan")

    return {
        "var_mean": float(v.mean()),
        "var_quantiles_0_25_50_75_100": [float(q0), float(q25), float(q50), float(q75), float(q100)],
        "var_iqr_over_mean": iqr_over_mean,
        "var_max_over_min": float(q100 / q0) if q0 > 0 else float("inf"),
        "spearman_var_vs_err_sq": _spearman(v, e_sq),
    }


def load_variance_head_for_l1only(config_path: str, checkpoint_path: str, device: str = "cuda"):
    """Convenience loader: same frozen L1-only model used everywhere else in
    this module family (compute_changepoints.py, error_adaptive_l1.py)."""
    return load_level1(config_path, checkpoint_path, device)


def diagnose_changepoint_overlap(
    var: torch.Tensor,
    errs: torch.Tensor,
    ep_boundaries: list,
    n_boundaries: int = 5,
    min_seg: int = 8,
    eps: float = 1e-3,
    near_tol: int = 2,
):
    """Step C: for every episode, re-run the exact same greedy peak-picking
    (pick_changepoints, unchanged) on two scores -- the existing raw error
    and the new err^2/(sigma_sq+eps) -- and report how much they agree.
    Takes a precomputed raw-units variance tensor (agnostic to estimator).
    """
    from pldm_envs.diverse_maze.adaptive_waypoints.segmentation import pick_changepoints

    exact_overlaps, near_overlaps = [], []
    for start, end in ep_boundaries:
        ep_err = errs[start:end]
        ep_var = var[start:end]
        new_score = ep_err.pow(2) / (ep_var + eps)

        old_b = pick_changepoints(ep_err, n_boundaries=n_boundaries, min_seg=min_seg)
        new_b = pick_changepoints(new_score, n_boundaries=n_boundaries, min_seg=min_seg)

        exact = len(set(old_b) & set(new_b))
        near = 0
        used = set()
        for ob in old_b:
            best = None
            for nb in new_b:
                if nb in used:
                    continue
                if abs(ob - nb) <= near_tol and (best is None or abs(ob - nb) < abs(ob - best)):
                    best = nb
            if best is not None:
                used.add(best)
                near += 1

        exact_overlaps.append(exact)
        near_overlaps.append(near)

    n_ep = len(ep_boundaries)
    return {
        "n_episodes": n_ep,
        "n_boundaries_per_episode": n_boundaries,
        "mean_exact_overlap": float(np.mean(exact_overlaps)),
        "mean_exact_overlap_ratio": float(np.mean(exact_overlaps) / n_boundaries),
        "mean_near_overlap_tol%d" % near_tol: float(np.mean(near_overlaps)),
        "mean_near_overlap_ratio_tol%d" % near_tol: float(np.mean(near_overlaps) / n_boundaries),
        "exact_overlap_histogram": {
            str(k): int((np.array(exact_overlaps) == k).sum()) for k in range(n_boundaries + 1)
        },
    }
