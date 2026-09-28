# Boundary-signal study (see PREREGISTRATION.md SS4). Ground-truth physical
# event labels, computed from (x, y, vx, vy) + map layout only -- independent
# of all 10 candidate signals in SS3.
#
# Indexing convention: a 60-step window has frames 0..60 (61 total, matching
# compute_changepoints.py's `window = l2_n_steps*l2_step_skip + 1`). Every
# event array below has length 60, where entry `i` labels frame `f = i + 1`
# -- the same frame the raw-error series' `err[i]` is computed for
# (`compute_error_series` drops frame 0 via `encodings[1:]`). This lines up
# events and all SS3 signals on one shared per-step index with no off-by-one
# translation needed later.
from typing import Dict, List

import numpy as np

from pldm_envs.diverse_maze.adaptive_waypoints.boundary_study.grid_utils import (
    GRID_SIZE,
    obs_to_ij,
    open_neighbor_count,
    wall_adjacent_edge_distance,
)

WALL_CONTACT_FRAC = 0.15  # PREREGISTRATION.md SS4, fraction of one grid cell
WALL_CONTACT_THRESHOLD = WALL_CONTACT_FRAC * GRID_SIZE  # raw units, ~0.148
TURN_DEGREES = 45.0
EPS_STATIONARY = 1e-6


def label_episode(obs: np.ndarray, layout: List[str]) -> Dict[str, np.ndarray]:
    """obs: (>=61, 4) [x, y, vx, vy]. Returns dict of four (60,) bool arrays:
    wall_contact, direction_turn, corridor_change, junction_arrival."""
    obs = obs[:61]
    assert obs.shape[0] == 61, f"expected >=61 frames, got {obs.shape[0]}"
    xy = obs[:, :2].astype(np.float64)
    v = obs[:, 2:4].astype(np.float64)
    ij = obs_to_ij(xy)  # (61, 2)
    speed = np.linalg.norm(v, axis=1)  # (61,)

    n_open_neighbors = np.array(
        [open_neighbor_count(layout, int(i), int(j)) for i, j in ij]
    )  # (61,)

    wall_contact = np.zeros(60, dtype=bool)
    direction_turn = np.zeros(60, dtype=bool)
    corridor_change = np.zeros(60, dtype=bool)
    junction_arrival = np.zeros(60, dtype=bool)

    for i in range(60):
        f = i + 1  # frame this entry labels

        # wall contact: needs f-1 (always available, f>=1)
        dist = wall_adjacent_edge_distance(layout, xy[f, 0], xy[f, 1], int(ij[f, 0]), int(ij[f, 1]))
        decelerating = speed[f] < speed[f - 1]
        wall_contact[i] = np.isfinite(dist) and dist <= WALL_CONTACT_THRESHOLD and decelerating

        # corridor entry/exit: needs f-1
        corridor_change[i] = n_open_neighbors[f] != n_open_neighbors[f - 1]

        # junction arrival: needs f-1
        cell_changed = tuple(ij[f]) != tuple(ij[f - 1])
        junction_arrival[i] = n_open_neighbors[f] >= 3 and cell_changed

        # direction turn: needs f-3..f-1
        if f >= 3:
            v_f = v[f]
            v_prev_mean = v[f - 3 : f].mean(axis=0)
            n1, n2 = np.linalg.norm(v_f), np.linalg.norm(v_prev_mean)
            if n1 >= EPS_STATIONARY and n2 >= EPS_STATIONARY:
                cos_angle = np.clip(np.dot(v_f, v_prev_mean) / (n1 * n2), -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(cos_angle))
                direction_turn[i] = angle_deg >= TURN_DEGREES

    return {
        "wall_contact": wall_contact,
        "direction_turn": direction_turn,
        "corridor_change": corridor_change,
        "junction_arrival": junction_arrival,
    }
