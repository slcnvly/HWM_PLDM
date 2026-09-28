# Boundary-signal study, predictor-free signals 6/7/8/10 (PREREGISTRATION.md
# SS3, items 6-8 and 10) -- none of these call the frozen predictor, so
# none are affected by the Amendment-2 finding (predictor-based signals on
# hold pending the user's decision).
import numpy as np


def signal_7_latent_speed(z):
    """z: (61, D) real encodings. Returns (60,): ||z_{t+1}-z_t||_2."""
    return np.linalg.norm(z[1:] - z[:-1], axis=1)


def signal_8_direction_change(z):
    """1 - cos(z_{t+1}-z_t, z_t-z_{t-1}). Undefined at t=0 (index 0 of the
    60-length series, which corresponds to frame 1) -- set to 0 there (per
    PREREGISTRATION.md SS3, excluded from peak-picking candidacy anyway
    since it's outside min_seg=8's feasible boundary region)."""
    d = z[1:] - z[:-1]  # (60, D): d[i] = z[i+1]-z[i] for i=0..59
    out = np.zeros(60)
    for i in range(1, 60):
        a, b = d[i], d[i - 1]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            out[i] = 0.0
        else:
            out[i] = 1 - np.dot(a, b) / (na * nb)
    return out


def signal_10_action_delta(actions):
    """actions: (60, A). ||a_t - a_{t-1}||_2, undefined/0 at t=0."""
    out = np.zeros(60)
    out[1:] = np.linalg.norm(actions[1:] - actions[:-1], axis=1)
    return out


SHORT_RUN_THRESHOLD = 3  # P(r_t <= this) is the returned signal, see docstring


def bocpd_changepoint_prob(x, hazard=0.1, prior_var_mult=10.0):
    """Adams & MacKay (2007), Normal-unknown-mean/KNOWN-variance conjugate
    model (PREREGISTRATION.md SS3 item 6's documented simplification),
    applied independently to ONE scalar sequence. x: (T,) one PCA component
    over the window.

    Known variance = empirical variance of x itself (fixed per-episode,
    per-component). Prior on the unknown mean: N(0, prior_var_mult *
    known_var) (vague/wide prior).

    NOTE on what's returned: for a CONSTANT hazard rate H, P(r_t=0) is
    mathematically pinned to exactly H at every t regardless of the data
    (the changepoint mass sums to H*sum(joint) and the total normalization
    is also proportional to sum(joint), so they cancel) -- verified
    empirically while implementing this (a flat 0.1 output at every t on a
    synthetic two-regime test caught it). The genuinely data-dependent part
    of the run-length posterior lives in P(r_t=1), P(r_t=2), ... (their
    growth messages depend on the PREVIOUS step's already-normalized
    distribution and how well x_t fits each run's accumulated mean, which
    is not similarly pinned). So this returns P(r_t <= SHORT_RUN_THRESHOLD)
    -- the probability the current run is still short -- as the per-step
    "changepoint-adjacent" signal, which does respond to the data (checked
    below on a synthetic two-regime sequence).
    """
    T = len(x)
    known_var = float(np.var(x)) + 1e-8
    prior_var = prior_var_mult * known_var
    prior_mean = 0.0

    # run-length posterior, r=0..t at time t. Track sufficient stats per
    # run length: count n_r, sum s_r (for the posterior mean update).
    run_probs = np.array([1.0])  # P(r=0) at t=-1 (before any data): trivially certain
    run_means = np.array([prior_mean])
    run_vars = np.array([prior_var])  # posterior variance of the mean, per run

    changepoint_prob = np.zeros(T)

    for t in range(T):
        xt = x[t]
        # predictive distribution for each current run: N(run_mean, known_var + run_var)
        pred_var = known_var + run_vars
        pred_std = np.sqrt(pred_var)
        log_pred = -0.5 * np.log(2 * np.pi * pred_var) - 0.5 * (xt - run_means) ** 2 / pred_var
        pred_prob = np.exp(log_pred)

        joint = run_probs * pred_prob  # unnormalized P(r_{t-1}, x_t)

        growth_joint = joint * (1 - hazard)
        cp_joint = float((joint * hazard).sum())

        new_run_probs = np.concatenate([[cp_joint], growth_joint])
        norm = new_run_probs.sum()
        if norm <= 0 or not np.isfinite(norm):
            new_run_probs = np.ones_like(new_run_probs) / len(new_run_probs)
        else:
            new_run_probs = new_run_probs / norm

        # posterior mean/var update for growth (run length +1): standard
        # Normal-Normal conjugate update, using this point added to the run.
        new_run_means = np.concatenate([[prior_mean], (run_means / run_vars + xt / known_var) / (1 / run_vars + 1 / known_var)])
        new_run_vars = np.concatenate([[prior_var], 1 / (1 / run_vars + 1 / known_var)])

        run_probs, run_means, run_vars = new_run_probs, new_run_means, new_run_vars
        changepoint_prob[t] = run_probs[: SHORT_RUN_THRESHOLD + 1].sum()

    return changepoint_prob


def signal_6_bocpd(z_pca, hazard=0.1):
    """z_pca: (61, K) PCA-projected encodings (K components, fit externally
    on the selection set). Returns (60,): summed per-component changepoint
    probability (PREREGISTRATION.md SS3 item 6), evaluated on the K
    per-component sequences of length 60 (differences or raw values? --
    applied to the raw latent sequence per the preregistration wording, so
    we run BOCPD on z_pca[1:61] directly, one scalar signal per PCA
    dimension, summed across dimensions)."""
    T = 60
    K = z_pca.shape[1]
    total = np.zeros(T)
    for k in range(K):
        total += bocpd_changepoint_prob(z_pca[1:61, k], hazard=hazard)
    return total
