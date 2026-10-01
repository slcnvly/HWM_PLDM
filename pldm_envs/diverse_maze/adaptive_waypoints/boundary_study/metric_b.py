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


def _point_to_chord_dist(z, a, b, k):
    """Perpendicular distance from z[k] to the line through z[a], z[b]."""
    va, vb, vk = z[a], z[b], z[k]
    ab = vb - va
    ab_norm = np.linalg.norm(ab)
    if ab_norm < 1e-8:
        return np.linalg.norm(vk - va)
    t = np.dot(vk - va, ab) / (ab_norm**2)
    proj = va + t * ab
    return np.linalg.norm(vk - proj)


def top_down_split(z, n_boundaries=5, min_seg=None):
    """Algorithm 14 (Douglas-Peucker style): repeatedly split the segment
    containing the point of maximum perpendicular deviation from its own
    chord, until n_boundaries splits are made. min_seg=None -> unconstrained
    (candidate points can be anywhere in a segment); min_seg=int ->
    candidates within min_seg of either segment endpoint are excluded."""
    T = z.shape[0] - 1
    segments = [(0, T)]
    boundaries = []
    while len(boundaries) < n_boundaries:
        best_seg, best_point, best_dist = None, None, -1.0
        for (a, b) in segments:
            if b - a <= 1:
                continue
            lo = a + (min_seg if min_seg else 1)
            hi = b - (min_seg if min_seg else 1) + 1
            for k in range(max(a + 1, lo), min(b, hi)):
                d = _point_to_chord_dist(z, a, b, k)
                if d > best_dist:
                    best_dist, best_seg, best_point = d, (a, b), k
        if best_point is None:
            break  # no feasible split left (min_seg exhausted the room)
        boundaries.append(best_point)
        segments.remove(best_seg)
        segments.append((best_seg[0], best_point))
        segments.append((best_point, best_seg[1]))

    # min_seg can starve greedy top-down of feasible splits before reaching
    # n_boundaries (no lookahead) -- fall back to uniform spacing for any
    # still-missing slots, same convention as segmentation.py's
    # pick_changepoints, so every algorithm returns exactly n_boundaries
    # and Metric B comparisons stay apples-to-apples. Reported alongside
    # results whenever this fallback actually triggers.
    if len(boundaries) < n_boundaries:
        uniform = [round(T * (i + 1) / (n_boundaries + 1)) for i in range(n_boundaries)]
        for u in uniform:
            if len(boundaries) >= n_boundaries:
                break
            if all(abs(u - c) >= (min_seg or 1) for c in boundaries) and 0 < u < T:
                boundaries.append(u)
        while len(boundaries) < n_boundaries:
            boundaries.append(min(T - 1, (max(boundaries) if boundaries else 0) + 1))
    return sorted(boundaries[:n_boundaries])


def bottom_up_merge(z, n_boundaries=5, min_seg=None):
    """Algorithm 15: repeatedly remove the boundary whose removal costs
    least (smallest resulting increase in squared reconstruction error),
    until n_boundaries remain.

    BUG FOUND AND FIXED (2026-09-30, see PROGRESS.md "bottom-up cliff"
    investigation): the original version always started from EVERY
    interior point as a boundary (ignoring min_seg entirely), merged down
    to exactly n_boundaries via pure unconstrained cost-minimization, and
    only THEN tried to patch min_seg violations post-hoc (force-merge the
    shortest-segment's cheaper neighbor, re-split the largest remaining
    segment at its midpoint). Since merging only ever GROWS segments, that
    first phase produced the IDENTICAL unconstrained-optimal 5 boundaries
    regardless of min_seg's value -- the only thing min_seg changed was how
    much ad-hoc, low-quality post-hoc repair got layered on top, which is
    exactly why min_seg=1 (trivially no violations, no repair) gave the
    true result while min_seg=2..8 were each degraded by a differing
    amount of patching -- a discontinuous artifact of the implementation,
    not a real property of the algorithm or the data.

    Fixed by building min_seg into the merge from the start: initialize
    with min_seg-SIZED atomic segments (boundaries at every min_seg-th
    point), not every single point. Since merging only grows segments,
    every segment stays >= min_seg throughout the entire process -- no
    post-hoc repair needed, and the result varies smoothly with min_seg
    (more, finer atomic chunks to start from as min_seg shrinks), matching
    top_down_split's already-smooth behavior. min_seg=None keeps the
    original every-point start (true unconstrained baseline, unaffected by
    this bug since it never entered the buggy branch)."""
    T = z.shape[0] - 1
    if min_seg:
        # stop early enough that the FINAL segment (last boundary -> T)
        # also satisfies min_seg -- T isn't generally a multiple of
        # min_seg, so a naive range(min_seg, T, min_seg) can leave a
        # too-short leftover last segment (caught by an assertion while
        # testing this fix: min_seg=8 on T=60 gives boundaries ending at
        # 56, leaving a length-4 final segment).
        boundaries = list(range(min_seg, T - min_seg + 1, min_seg))
    else:
        boundaries = list(range(1, T))

    while len(boundaries) > n_boundaries:
        anchors = [0] + boundaries + [T]
        best_idx, best_cost = None, float("inf")
        for i in range(1, len(anchors) - 1):
            a, b = anchors[i - 1], anchors[i + 1]
            cost, _ = segment_cost(z, a, b)
            if cost < best_cost:
                best_cost, best_idx = cost, i
        del boundaries[best_idx - 1]

    # only reachable if min_seg was large enough that fewer than
    # n_boundaries atomic chunks existed to begin with -- pad via
    # top_down-style splitting of the largest remaining segment, same
    # min_seg-respecting feasibility check as top_down_split.
    while len(boundaries) < n_boundaries:
        anchors = [0] + boundaries + [T]
        seg_lens = [(anchors[i + 1] - anchors[i], i) for i in range(len(anchors) - 1)]
        seg_lens.sort(reverse=True)
        _, i = seg_lens[0]
        a, b = anchors[i], anchors[i + 1]
        if min_seg and b - a < 2 * min_seg:
            break  # can't split further without violating min_seg
        mid = (a + b) // 2
        if a < mid < b and mid not in boundaries:
            boundaries.append(mid)
            boundaries.sort()
        else:
            break
    return sorted(boundaries)
