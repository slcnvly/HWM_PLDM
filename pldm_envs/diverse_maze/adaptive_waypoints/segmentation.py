# HWM_PLDM 원본 대비 추가된 실험 코드 (fork: slcnvly/HWM_PLDM, branch: adaptive-waypoints).
# See DESIGN.md for the full rationale. Pure tensor/array math, no model or
# dataset dependency, so it's testable in isolation.
from typing import List

import torch


def pick_changepoints(
    error_series: torch.Tensor, n_boundaries: int, min_seg: int
) -> List[int]:
    """Greedy peak-picking with a minimum-spacing constraint.

    Args:
        error_series: (T,) per-step prediction error for one window.
        n_boundaries: how many internal boundaries to pick (n_waypoints - 1).
        min_seg: minimum spacing (in raw steps) enforced between any two
            chosen boundaries, and between a boundary and the window's own
            start/end (0 and T).

    Returns:
        Sorted list of `n_boundaries` boundary indices in (0, T), each a
        valid split point (segments are `error_series[prev:b]`). Falls back
        to uniform spacing for any boundary that can't be placed without
        violating min_seg (only happens for pathologically short/flat
        series) so the caller always gets exactly n_boundaries indices.
    """
    T = error_series.shape[0]
    assert 0 < min_seg, "min_seg must be positive"

    candidates = error_series.clone()
    # boundaries must leave room for a real segment on both sides
    if min_seg > 1:
        candidates[: min_seg - 1] = -float("inf")
        candidates[T - (min_seg - 1) :] = -float("inf")

    chosen: List[int] = []
    for _ in range(n_boundaries):
        if not torch.isfinite(candidates).any():
            break
        idx = int(torch.argmax(candidates).item())
        chosen.append(idx)
        lo = max(0, idx - min_seg + 1)
        hi = min(T, idx + min_seg)
        candidates[lo:hi] = -float("inf")

    chosen.sort()

    if len(chosen) < n_boundaries:
        # Fallback: fill remaining slots with uniform spacing, skipping any
        # index too close to an already-chosen one.
        n_missing = n_boundaries - len(chosen)
        uniform = [
            round(T * (i + 1) / (n_boundaries + 1)) for i in range(n_boundaries)
        ]
        for u in uniform:
            if len(chosen) >= n_boundaries:
                break
            if all(abs(u - c) >= min_seg for c in chosen) and 0 < u < T:
                chosen.append(u)
                n_missing -= 1
        chosen = sorted(set(chosen))[:n_boundaries]
        # last resort, if still short (degenerate tiny T): pad arbitrarily
        while len(chosen) < n_boundaries:
            chosen.append(min(T - 1, (chosen[-1] if chosen else 0) + 1))
        chosen = sorted(chosen)

    return chosen


def boundaries_to_segments(boundaries: List[int], total_len: int) -> List[tuple]:
    """[b0, b1, ...] -> [(0,b0), (b0,b1), ..., (bk,total_len)]."""
    edges = [0] + list(boundaries) + [total_len]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def resample_to_fixed_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Linearly resample a (seg_len, D) sequence to (target_len, D) along dim 0.

    Used to feed variable-length action segments into the *unchanged* fixed-
    size action encoder (see DESIGN.md decision 2). seg_len==1 is handled by
    repeating (interpolate needs >=2 points).
    """
    assert x.dim() == 2, f"expected (seg_len, D), got shape {tuple(x.shape)}"
    seg_len = x.shape[0]
    if seg_len == target_len:
        return x
    if seg_len == 1:
        return x.repeat(target_len, 1)
    # (seg_len, D) -> (1, D, seg_len) for F.interpolate's expected layout
    x_t = x.t().unsqueeze(0)
    resampled = torch.nn.functional.interpolate(
        x_t, size=target_len, mode="linear", align_corners=True
    )
    return resampled.squeeze(0).t()
