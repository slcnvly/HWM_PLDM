# Boundary-signal study, Metric B (PREREGISTRATION.md SS6): piecewise-linear
# latent reconstruction error. Given a 60-step window's latent trajectory
# z[0..60] (61 points, fused 18-channel encoding space -- doesn't involve
# the predictor at all, so unaffected by the Amendment-2 predictor
# degeneracy) and a set of 5 internal boundaries, linearly interpolate
# between the 7 anchor points (0, b1..b5, 60) and score the mean squared
# reconstruction error of the interior (non-anchor) points.
import numpy as np


def segment_cost(z, i, j):
    """Sum of squared reconstruction errors for interior points strictly
    between anchors i and j (exclusive), linearly interpolating z[i]->z[j].
    z: (T+1, D). Returns a scalar (summed over interior points AND dims,
    then the caller divides by count for a mean -- kept as a sum here so
    segment costs can be added directly for a multi-segment total)."""
    if j - i <= 1:
        return 0.0, 0
    interior = np.arange(i + 1, j)
    alpha = (interior - i) / (j - i)  # (n_interior,)
    lerp = z[i][None, :] + alpha[:, None] * (z[j] - z[i])[None, :]
    resid = z[interior] - lerp
    return float((resid**2).sum()), len(interior)


def score_boundaries(z, boundaries):
    """z: (T+1, D). boundaries: sorted list of 5 internal indices in (0, T).
    Returns mean squared reconstruction error over all interior points
    (pooling across all 6 segments), matching PREREGISTRATION.md SS6."""
    anchors = [0] + list(boundaries) + [z.shape[0] - 1]
    total_sq, total_n = 0.0, 0
    for a, b in zip(anchors[:-1], anchors[1:]):
        sq, n = segment_cost(z, a, b)
        total_sq += sq
        total_n += n
    if total_n == 0:
        return 0.0
    return total_sq / (total_n * z.shape[1])  # mean over interior points AND feature dims


def oracle_boundaries(z, n_boundaries=5, min_seg=8):
    """Exact DP: the n_boundaries internal split points (respecting min_seg
    spacing) that minimize total squared reconstruction error. O(T^2) segment
    costs + O(T^2 * n_boundaries) DP.
    z: (T+1, D). Returns (boundaries, mean_sq_error)."""
    T = z.shape[0] - 1  # 60

    # all pairwise segment costs (sum of squared error, not yet divided by n)
    cost = np.full((T + 1, T + 1), np.inf)
    npts = np.zeros((T + 1, T + 1), dtype=int)
    for i in range(T + 1):
        for j in range(i + 1, T + 1):
            sq, n = segment_cost(z, i, j)
            cost[i, j] = sq
            npts[i, j] = n

    # dp[k][j] = (best total sq error using k segments ending at anchor j, list of prior anchors)
    NEG = float("inf")
    dp = np.full((n_boundaries + 2, T + 1), NEG)  # dp[k][j]: k segments so far, last anchor at j
    parent = [[-1] * (T + 1) for _ in range(n_boundaries + 2)]
    dp[0][0] = 0.0
    for k in range(1, n_boundaries + 2):  # total segments = n_boundaries+1
        for j in range(1, T + 1):
            best = NEG
            best_i = -1
            for i in range(0, j):
                if dp[k - 1][i] == NEG:
                    continue
                # min_seg: every segment (including the first/last, against
                # the window's own start/end) must be >= min_seg long,
                # matching segmentation.py's pick_changepoints semantics.
                if (j - i) < min_seg:
                    continue
                val = dp[k - 1][i] + cost[i, j]
                if best_i == -1 or val < best:
                    best = val
                    best_i = i
            dp[k][j] = best
            parent[k][j] = best_i

    k_final = n_boundaries + 1
    j = T
    if dp[k_final][T] == NEG:
        raise RuntimeError("oracle DP: no feasible segmentation found (min_seg too large for T/n_boundaries)")
    total_sq = dp[k_final][T]
    boundaries = []
    cur_k, cur_j = k_final, T
    while cur_k > 0:
        i = parent[cur_k][cur_j]
        if cur_k < k_final:  # don't include the final endpoint T or the start 0
            boundaries.append(cur_j)
        cur_j = i
        cur_k -= 1
    boundaries = sorted(boundaries)

    total_n = sum(npts[a, b] for a, b in zip([0] + boundaries, boundaries + [T]))
    mean_sq = total_sq / (total_n * z.shape[1]) if total_n > 0 else 0.0
    return boundaries, mean_sq


def random_boundaries(T, n_boundaries, min_seg, rng):
    """One uniform-random valid segmentation respecting min_seg spacing
    (against 0/T too, matching oracle_boundaries' constraint)."""
    max_tries = 200
    for _ in range(max_tries):
        cand = sorted(rng.choice(range(1, T), size=n_boundaries, replace=False).tolist())
        anchors = [0] + cand + [T]
        ok = all(anchors[i + 1] - anchors[i] >= min_seg for i in range(len(anchors) - 1))
        if ok:
            return cand
    # fallback: uniform spacing
    return [round(T * (i + 1) / (n_boundaries + 1)) for i in range(n_boundaries)]


def fixed_boundaries():
    return [10, 20, 30, 40, 50]
