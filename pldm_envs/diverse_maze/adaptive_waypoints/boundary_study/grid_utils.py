# Boundary-signal study (see PREREGISTRATION.md). Corrected grid-coordinate
# conversion for maze2d_large_diverse -- see PREREGISTRATION.md SS2 for the
# root-cause derivation (obs_range_total's divisor mismatch, 10.2 vs n=10).
# Deliberately NOT importing pldm_envs.diverse_maze.utils.obs_to_ij / the
# shared RENDER_STATS entry: this local override exists so the fix doesn't
# touch shared code paths other stages/results depend on.
from typing import List, Tuple

import numpy as np

OBS_MIN_SPACE = 0.351
OBS_MAX_SPACE = 8.248
EMPTY_BLOCKS = 8  # open (non-border-wall) cells per axis, large_diverse 25-map layouts
N = 10  # total grid cells per axis (confirmed via train_maps.pt layout strings)

GRID_SIZE = (OBS_MAX_SPACE - OBS_MIN_SPACE) / EMPTY_BLOCKS  # 0.987125
OBS_RANGE_TOTAL = GRID_SIZE * N  # 9.87125 (corrected; was 10.068675)
OBS_MIN_TOTAL = OBS_MIN_SPACE - GRID_SIZE  # -0.636125 (unchanged)


def obs_to_ij(xy: np.ndarray) -> np.ndarray:
    """xy: (..., 2) array of [x, y]. Returns (..., 2) int array of [i, j],
    matching pldm_envs.diverse_maze.utils.obs_to_ij's orientation (i=row,
    derived from x=obs[...,0]; j=col, derived from y=obs[...,1];
    map_layout[i][j] indexing, no transpose -- see
    analysis/event_alignment/stage2_grid_validation.py's docstring for the
    orientation proof via wrappers.py::find_shortest_path)."""
    return np.floor((xy - OBS_MIN_TOTAL) / GRID_SIZE).astype(int)


def open_neighbor_count(layout: List[str], i: int, j: int) -> int:
    """Count of 4-connected open ('O') neighbors of cell (i, j). Out-of-bounds
    neighbors don't count as open."""
    rows, cols = len(layout), len(layout[0])
    count = 0
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ni, nj = i + di, j + dj
        if 0 <= ni < rows and 0 <= nj < cols and layout[ni][nj] == "O":
            count += 1
    return count


def wall_adjacent_edge_distance(layout: List[str], x: float, y: float, i: int, j: int) -> float:
    """Distance (raw units) from continuous position (x, y) to the nearest
    edge of cell (i, j) that borders a wall ('#') cell. Returns +inf if no
    neighbor of (i, j) is a wall."""
    rows, cols = len(layout), len(layout[0])
    cell_x0 = OBS_MIN_TOTAL + i * GRID_SIZE
    cell_y0 = OBS_MIN_TOTAL + j * GRID_SIZE
    frac_x = (x - cell_x0) / GRID_SIZE  # in [0, 1) within this cell
    frac_y = (y - cell_y0) / GRID_SIZE

    def is_wall(ni, nj):
        return not (0 <= ni < rows and 0 <= nj < cols) or layout[ni][nj] == "#"

    dists = []
    if is_wall(i - 1, j):
        dists.append(frac_x * GRID_SIZE)  # distance to low-i edge
    if is_wall(i + 1, j):
        dists.append((1 - frac_x) * GRID_SIZE)  # distance to high-i edge
    if is_wall(i, j - 1):
        dists.append(frac_y * GRID_SIZE)
    if is_wall(i, j + 1):
        dists.append((1 - frac_y) * GRID_SIZE)
    return min(dists) if dists else float("inf")


def validate_zero_violations(xy: np.ndarray, layouts_per_point: List[List[str]]) -> Tuple[int, int]:
    """Sanity check used in PREREGISTRATION.md SS2. xy: (N, 2). layouts_per_point:
    length-N list of that point's map layout. Returns (wall_count, total)."""
    ij = obs_to_ij(xy)
    wall = 0
    for k in range(len(xy)):
        i, j = ij[k]
        layout = layouts_per_point[k]
        rows, cols = len(layout), len(layout[0])
        if 0 <= i < rows and 0 <= j < cols and layout[i][j] == "#":
            wall += 1
    return wall, len(xy)
