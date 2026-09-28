# Small-scale (20-trajectory) pre-registered sanity check requested before
# running event labeling on the full corpus (see PROGRESS.md "Stage 1
# checkpoint" note). Checks, on the BUGGY (pre-fix) constants:
#   1. violation count broken down by direction (+i, -i, +j, -j)
#   2. whether violation "depth" (how far into the wrongly-assigned wall
#      cell the point falls) scales with |coordinate - origin| -- the
#      signature of a SCALE error, as opposed to a constant OFFSET error
#      (which would show violation depth roughly independent of position).
# Then repeats the direction-count check on the FIXED constants and asserts
# it's ~0 in all four directions.
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

# Buggy (pre-fix) constants, exactly as in
# pldm_envs/diverse_maze/data_generation/maze_stats.py:124-125
BUGGY_OBS_MIN_TOTAL = -0.636125
BUGGY_OBS_RANGE_TOTAL = 10.068675
N = 10
BUGGY_GRID_SIZE = BUGGY_OBS_RANGE_TOTAL / N  # 1.0068675, matches utils.py:238's grid_size = obs_range_total / n

from pldm_envs.diverse_maze.adaptive_waypoints.boundary_study.grid_utils import (
    GRID_SIZE as FIXED_GRID_SIZE,
    OBS_MIN_TOTAL as FIXED_OBS_MIN_TOTAL,
    obs_to_ij as fixed_obs_to_ij,
)

HERE = os.path.dirname(__file__)
DATA_ROOT = os.path.join(HERE, "..", "..", "datasets", "r50_local", "r50_dataset")
N_TRAJECTORIES = 20


def buggy_obs_to_ij(xy):
    return np.floor((xy - BUGGY_OBS_MIN_TOTAL) / BUGGY_GRID_SIZE).astype(int)


def direction_counts_and_depths(xy, ij, grid_size, obs_min_total, layout, rows, cols):
    """For each point whose ij lands on a wall cell, find the nearest
    wall-adjacent edge direction and the fractional depth (in raw units)
    into that cell from that edge (0 = right at the edge, grid_size/2 =
    cell center)."""
    counts = {"+i": 0, "-i": 0, "+j": 0, "-j": 0}
    depths = []  # (direction, coord_value, depth)
    for k in range(len(xy)):
        i, j = ij[k]
        if not (0 <= i < rows and 0 <= j < cols) or layout[i][j] != "#":
            continue
        cell_x0 = obs_min_total + i * grid_size
        cell_y0 = obs_min_total + j * grid_size
        frac_x = (xy[k, 0] - cell_x0) / grid_size
        frac_y = (xy[k, 1] - cell_y0) / grid_size
        d = {"-i": frac_x, "+i": 1 - frac_x, "-j": frac_y, "+j": 1 - frac_y}
        nearest = min(d, key=d.get)
        counts[nearest] += 1
        coord_value = xy[k, 0] if nearest in ("-i", "+i") else xy[k, 1]
        depth_raw = d[nearest] * grid_size  # distance to that edge, raw units
        depths.append((nearest, coord_value, depth_raw))
    return counts, depths


def main():
    splits = torch.load(os.path.join(DATA_ROOT, "main", "data.p"), weights_only=False)
    maps = torch.load(os.path.join(DATA_ROOT, "main", "train_maps.pt"), weights_only=False)

    # r50 episodes are grouped sequentially by map (50 consecutive episodes
    # per map, confirmed via map_idx ordering) -- the first N_TRAJECTORIES
    # would all land on map 0. Pick one episode per map instead, for a
    # sample spanning multiple distinct layouts.
    seen_maps = {}
    for ep_idx, ep in enumerate(splits):
        m = int(ep["map_idx"])
        if m not in seen_maps:
            seen_maps[m] = ep_idx
        if len(seen_maps) >= N_TRAJECTORIES:
            break
    ep_indices = sorted(seen_maps.values())

    all_xy = []
    all_layouts = []
    for ep_idx in ep_indices:
        ep = splits[ep_idx]
        obs = ep["observations"][:, :2].astype(np.float64)
        layout = maps[int(ep["map_idx"])].split("\\")
        all_xy.append(obs)
        all_layouts.append(layout)
        print(f"episode {ep_idx}: map_idx={ep['map_idx']}, {len(obs)} frames")

    print("\n=== BUGGY constants (pre-fix): violations by direction ===")
    buggy_total_counts = {"+i": 0, "-i": 0, "+j": 0, "-j": 0}
    all_depths = []
    total_pts = 0
    for xy, layout in zip(all_xy, all_layouts):
        rows, cols = len(layout), len(layout[0])
        ij = buggy_obs_to_ij(xy)
        counts, depths = direction_counts_and_depths(
            xy, ij, BUGGY_GRID_SIZE, BUGGY_OBS_MIN_TOTAL, layout, rows, cols
        )
        for k in buggy_total_counts:
            buggy_total_counts[k] += counts[k]
        all_depths.extend(depths)
        total_pts += len(xy)

    n_viol = sum(buggy_total_counts.values())
    print(f"total points: {total_pts}, violations: {n_viol} ({n_viol/total_pts*100:.2f}%)")
    for k, v in buggy_total_counts.items():
        print(f"  {k}: {v}")

    print("\n=== Proportionality check: violation depth vs |coord - origin| ===")
    if len(all_depths) >= 3:
        coord_vals = np.array([d[1] - BUGGY_OBS_MIN_TOTAL for d in all_depths])  # distance from origin
        depth_vals = np.array([d[2] for d in all_depths])
        corr = np.corrcoef(coord_vals, depth_vals)[0, 1]
        # linear fit depth = a * coord + b
        a, b = np.polyfit(coord_vals, depth_vals, 1)
        print(f"n_violations_with_depth={len(all_depths)}")
        print(f"Pearson correlation(depth, coord-from-origin) = {corr:.4f}")
        print(f"linear fit: depth = {a:.6f} * (coord-origin) + {b:.6f}")
        print(
            "(scale-error hypothesis predicts depth ~ (coord-origin) * "
            f"(BUGGY_GRID_SIZE-FIXED_GRID_SIZE)/FIXED_GRID_SIZE-ish slope; "
            f"BUGGY_GRID_SIZE-FIXED_GRID_SIZE={BUGGY_GRID_SIZE-FIXED_GRID_SIZE:.6f})"
        )
    else:
        print("too few violations to fit a trend")

    print("\n=== Direct index-drift check (all points, not just violations) ===")
    print("buggy_index - fixed_index, as a function of |coord - origin| -- a")
    print("pure scale error should show a monotonically increasing staircase;")
    print("a pure constant offset would show a roughly flat drift.")
    drift_i, coord_i = [], []
    drift_j, coord_j = [], []
    for xy in all_xy:
        buggy_ij = buggy_obs_to_ij(xy)
        fixed_ij = fixed_obs_to_ij(xy)
        drift_i.extend((buggy_ij[:, 0] - fixed_ij[:, 0]).tolist())
        coord_i.extend((xy[:, 0] - BUGGY_OBS_MIN_TOTAL).tolist())
        drift_j.extend((buggy_ij[:, 1] - fixed_ij[:, 1]).tolist())
        coord_j.extend((xy[:, 1] - BUGGY_OBS_MIN_TOTAL).tolist())
    drift_i, coord_i = np.array(drift_i), np.array(coord_i)
    drift_j, coord_j = np.array(drift_j), np.array(coord_j)
    all_drift = np.concatenate([drift_i, drift_j])
    all_coord = np.concatenate([coord_i, coord_j])
    corr2 = np.corrcoef(all_coord, all_drift)[0, 1]
    print(f"drift range: min={all_drift.min()}, max={all_drift.max()}")
    print(f"Pearson correlation(index_drift, coord-from-origin) = {corr2:.4f}")
    # bucket by coordinate decile to show the staircase explicitly
    order = np.argsort(all_coord)
    n = len(order)
    print("coord-from-origin decile -> mean index drift:")
    for d in range(10):
        idx = order[int(d * n / 10) : int((d + 1) * n / 10)]
        print(f"  decile {d}: coord~{all_coord[idx].mean():.3f} -> mean drift {all_drift[idx].mean():.3f}")

    print("\n=== FIXED constants (post-fix): violations by direction ===")
    fixed_total_counts = {"+i": 0, "-i": 0, "+j": 0, "-j": 0}
    fixed_total_pts = 0
    for xy, layout in zip(all_xy, all_layouts):
        rows, cols = len(layout), len(layout[0])
        ij = fixed_obs_to_ij(xy)
        counts, depths = direction_counts_and_depths(
            xy, ij, FIXED_GRID_SIZE, FIXED_OBS_MIN_TOTAL, layout, rows, cols
        )
        for k in fixed_total_counts:
            fixed_total_counts[k] += counts[k]
        fixed_total_pts += len(xy)

    n_viol_fixed = sum(fixed_total_counts.values())
    print(f"total points: {fixed_total_pts}, violations: {n_viol_fixed} ({n_viol_fixed/fixed_total_pts*100:.4f}%)")
    for k, v in fixed_total_counts.items():
        print(f"  {k}: {v}")

    assert n_viol_fixed == 0, f"expected 0 violations post-fix on this 20-trajectory sample, got {n_viol_fixed}"
    print("\nPASS: 0 violations in all 4 directions on the fixed constants (20-trajectory sample).")


if __name__ == "__main__":
    main()
