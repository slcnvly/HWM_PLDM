# HWM_PLDM 원본 대비 추가된 실험 코드 (fork: slcnvly/HWM_PLDM, branch: adaptive-waypoints).
# See DESIGN.md and RESULTS.md SS8 for the full rationale.
#
# Post-hoc variance head: an approximation to "degree of surprise" for
# changepoint scoring, standing in for ensemble variance / MC-dropout
# variance -- both unavailable without retraining frozen level1
# (ensemble_size=1, dropout=0.0 in every checkpoint this project has; see
# RESULTS.md SS8 for the code-level confirmation of why those paths are
# closed). Trains a small MLP to predict, from the frozen level1 latent
# state, the expected magnitude of compute_changepoints.py's existing raw
# one-step prediction error at that state -- entirely offline, no env/GPU
# beyond a handful of small forward passes through the already-frozen model.
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


class VarianceHead(nn.Module):
    """2-hidden-layer MLP, softplus output (always positive). Predicts
    sigma^2(latent state) -- see module docstring / RESULTS.md SS8 for what
    this approximates and why."""

    def __init__(self, input_dim: int, hidden: int = 128, eps: float = 1e-3):
        super().__init__()
        self.eps = eps
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        raw = self.net(x).squeeze(-1)
        return F.softplus(raw) + self.eps


def train_variance_head(
    train_errs: torch.Tensor,
    train_encs: torch.Tensor,
    val_errs: torch.Tensor,
    val_encs: torch.Tensor,
    device: str,
    hidden: int = 128,
    eps: float = 1e-3,
    epochs: int = 30,
    batch_size: int = 1024,
    lr: float = 1e-3,
):
    """Trains VarianceHead via Gaussian NLL: treats each err value as a
    zero-mean Gaussian draw whose variance is sigma^2(state), i.e.
    F.gaussian_nll_loss(input=0, target=err, var=sigma_sq) ==
    0.5*(log(sigma_sq) + err**2/sigma_sq) -- so sigma_sq is trained to
    approximate E[err^2 | state], matching the "expected squared error"
    target the score in step 2 (err^2 / sigma_sq) assumes.

    Returns the trained model plus a per-epoch log of train/val NLL, for
    the overfitting check RESULTS.md SS8 reports.
    """
    input_dim = train_encs.shape[1]
    model = VarianceHead(input_dim, hidden=hidden, eps=eps).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_encs = train_encs.to(device)
    train_errs = train_errs.to(device)
    val_encs = val_encs.to(device)
    val_errs = val_errs.to(device)
    n_train = train_encs.shape[0]
    zeros_val = torch.zeros_like(val_errs)

    log = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            x, y = train_encs[idx], train_errs[idx]
            var = model(x)
            loss = F.gaussian_nll_loss(torch.zeros_like(y), y, var, full=False)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.shape[0]
        train_nll = total_loss / n_train

        model.eval()
        with torch.no_grad():
            val_var = model(val_encs)
            val_nll = F.gaussian_nll_loss(zeros_val, val_errs, val_var, full=False).item()

        log.append({"epoch": epoch, "train_nll": train_nll, "val_nll": val_nll})
        print(f"[variance head] epoch {epoch}: train_nll={train_nll:.4f} val_nll={val_nll:.4f}", flush=True)

    return model, log


def correlation_stats(model, encs: torch.Tensor, errs: torch.Tensor, device: str):
    """Pearson correlation between predicted variance and (a) the raw error
    (b) the squared error the model was actually trained to match, plus
    quantile summary of the predicted variance itself -- for the "is this
    just reproducing the actual error" overfitting check."""
    model.eval()
    with torch.no_grad():
        var = model(encs.to(device)).cpu().numpy()
    err = errs.numpy()

    def pearson(a, b):
        a = a - a.mean()
        b = b - b.mean()
        denom = np.sqrt((a**2).sum() * (b**2).sum())
        return float((a * b).sum() / denom) if denom > 0 else float("nan")

    quantiles = np.percentile(var, [0, 25, 50, 75, 100])
    return {
        "corr_var_vs_err": pearson(var, err),
        "corr_var_vs_err_sq": pearson(var, err**2),
        "r2_err_sq": float(
            1 - np.sum((err**2 - var) ** 2) / np.sum((err**2 - (err**2).mean()) ** 2)
        ),
        "var_mean": float(var.mean()),
        "var_quantiles_0_25_50_75_100": [float(q) for q in quantiles],
    }


def load_variance_head_for_l1only(config_path: str, checkpoint_path: str, device: str = "cuda"):
    """Convenience loader: same frozen L1-only model used everywhere else in
    this module family (compute_changepoints.py, error_adaptive_l1.py)."""
    return load_level1(config_path, checkpoint_path, device)


def diagnose_changepoint_overlap(
    model,
    errs: torch.Tensor,
    encs: torch.Tensor,
    ep_boundaries: list,
    device: str,
    n_boundaries: int = 5,
    min_seg: int = 8,
    eps: float = 1e-3,
    near_tol: int = 2,
):
    """Step 3: for every episode, re-run the exact same greedy peak-picking
    (pick_changepoints, unchanged) on two scores -- the existing raw error
    and the new err^2/(sigma_sq+eps) -- and report how much they agree.

    Returns a dict with per-episode overlap counts plus aggregate stats:
    exact-index overlap (literal same step chosen) and near-match overlap
    (within near_tol steps, since a peak can shift by a step or two without
    being a materially different changepoint).
    """
    from pldm_envs.diverse_maze.adaptive_waypoints.segmentation import pick_changepoints

    model.eval()
    with torch.no_grad():
        var = model(encs.to(device)).cpu()

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
