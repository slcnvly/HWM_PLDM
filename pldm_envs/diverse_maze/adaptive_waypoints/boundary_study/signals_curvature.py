# Boundary-signal study, Amendment 3: curvature-based signals (11-13) and
# signal 16 (action-sensitivity divergence, relative comparison -- the one
# predictor-based signal that cancels the Amendment-2/gate-check bias by
# construction, since it compares predictions against EACH OTHER, never
# against the true target). See PREREGISTRATION.md Amendment 3 for the
# full rationale.
import numpy as np
import torch

N_OBS_CHANNELS = 16
H, W = 43, 43
D_OBS = N_OBS_CHANNELS * H * W
D_FUSED = 18 * H * W

# Empirical action distribution (main split, all 60*1250 real actions,
# computed once -- see PROGRESS.md Part A ad-hoc check): mean~[0.024,
# 0.045], std~[0.366, 0.449]. Used to sample signal 16's K=16 candidate
# actions (parametric Gaussian approximation of "the dataset's action
# distribution", not a bootstrap resample of literal action vectors --
# documented explicitly since the user's wording allows either reading).
ACTION_MEAN = np.array([0.0243, 0.0451])
ACTION_STD = np.array([0.3664, 0.4494])


# Indexing convention: matches signals 7/8/10 (signals_predictor_free.py),
# NOT event_labels.py's "index i labels frame i+1" convention -- output
# index i corresponds DIRECTLY to frame i (t=i), since these are properties
# "at" a point in the sequence, not prediction errors "for" the next frame.
# This matters because pick_changepoints' output is fed straight into
# metric_b.score_boundaries, which treats boundary values as literal
# indices into z (0..60) -- signals 7/8/10/11/12/13 all need this same
# direct convention to be scored consistently.


def signal_11_curvature(z):
    """z: (61, D). out[i] = ||z[i+1] - 2*z[i] + z[i-1]||_2, for i=1..59
    (i=0 undefined, needs z[-1] -- set to 0)."""
    out = np.zeros(60)
    for i in range(1, 60):
        out[i] = np.linalg.norm(z[i + 1] - 2 * z[i] + z[i - 1])
    return out


def signal_12_normalized_curvature(z):
    """Signal 11 divided by local displacement (||z[i+1]-z[i]|| + ||z[i]-z[i-1]||)."""
    out = np.zeros(60)
    curv = signal_11_curvature(z)
    for i in range(1, 60):
        disp = np.linalg.norm(z[i + 1] - z[i]) + np.linalg.norm(z[i] - z[i - 1])
        out[i] = curv[i] / disp if disp > 1e-8 else 0.0
    return out


def signal_13_chord_deviation(z, w=5):
    """Distance from z[i] to the line connecting z[i-w] and z[i+w].
    Undefined (0) near the window edges where i-w<0 or i+w>60."""
    out = np.zeros(60)
    for i in range(60):
        if i - w < 0 or i + w > 60:
            continue
        a, b = z[i - w], z[i + w]
        ab = b - a
        ab_norm = np.linalg.norm(ab)
        if ab_norm < 1e-8:
            out[i] = np.linalg.norm(z[i] - a)
            continue
        t_proj = np.dot(z[i] - a, ab) / (ab_norm**2)
        proj_point = a + t_proj * ab
        out[i] = np.linalg.norm(z[i] - proj_point)
    return out


def signal_16_action_sensitivity_divergence(model, encodings_torch, proprio_component, K=16, seed=0):
    """encodings_torch: (61,1,C,H,W) torch tensor (real backbone output,
    NOT flattened). proprio_component: (61,1,Cp,H,W) or None.

    For each t=0..59: K candidate actions sampled from the dataset's
    (parametric-Gaussian-approximated) action distribution, all applied to
    the SAME real z_t; K predictions obtained in one batched call; signal
    = sum of per-dim variance ACROSS the K predictions (compared to their
    own mean, never to the true target z_{t+1}) -- this is what cancels
    the Amendment-2 common-bias term by construction, since the bias is
    identical across all K candidates (same z_t) and cancels when
    computing variance around their own mean.

    Returns (obs_signal, proprio_signal), each (60,).
    """
    rng = np.random.default_rng(seed)
    candidate_actions = rng.normal(
        loc=ACTION_MEAN, scale=ACTION_STD, size=(K, 2)
    ).astype(np.float32)  # fixed pool, reused across all t in this episode
    actions_pool = torch.from_numpy(candidate_actions)  # (K, 2)

    obs_sig = np.zeros(60)
    proprio_sig = np.zeros(60)

    for t in range(60):
        state_encs_b = encodings_torch[t : t + 1].squeeze(1).unsqueeze(0).expand(1, K, -1, -1, -1)  # (1,K,C,H,W)
        actions_b = actions_pool.unsqueeze(0)  # (1, K, 2)
        proprio_b = None
        if proprio_component is not None:
            proprio_b = proprio_component[t : t + 1].squeeze(1).unsqueeze(0).expand(1, K, -1, -1, -1)

        with torch.no_grad():
            pred = model.level1.predictor.forward_multiple(
                state_encs=state_encs_b.contiguous(), actions=actions_b, T=1,
                proprio=proprio_b.contiguous() if proprio_b is not None else None,
                compute_posterior=False,
            ).predictions[1]  # (K, C, H, W)

        pred_flat = pred.flatten(1).cpu().numpy()  # (K, D_fused)
        obs_preds = pred_flat[:, :D_OBS]
        proprio_preds = pred_flat[:, D_OBS:D_FUSED]

        obs_sig[t] = obs_preds.var(axis=0).sum()
        proprio_sig[t] = proprio_preds.var(axis=0).sum()

    return obs_sig, proprio_sig
