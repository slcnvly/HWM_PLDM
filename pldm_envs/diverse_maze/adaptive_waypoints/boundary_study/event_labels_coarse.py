# Label-noise check for Metric A: coarser re-definitions of the two
# high-rate event types (wall_contact 16.7%, direction_turn 18.7% of steps
# under event_labels.py). corridor_change / junction_arrival are discrete
# grid-cell events and are kept unchanged.
#
# Same index convention as event_labels.py: entry i labels frame f = i + 1.
#
#   direction_turn (coarse): angle between v[f-10] and v[f] >= 90 deg, i.e. net
#     heading change across a 10-step window. The flag is placed at the window
#     centre (f-5), the time the turn actually happens.
#   wall_contact (coarse): near a wall (same 0.15-cell threshold) at frame f
#     AND speed strictly decreasing for >= 3 consecutive steps ending at f.
#
# A physical turn/contact makes several consecutive frames qualify, so each
# run of consecutive flags is collapsed to ONE event at its midpoint
# ("collapsed" variant, the primary one). The per-frame variant is also
# returned for the event-rate comparison.
from typing import Dict, List

import numpy as np

from pldm_envs.diverse_maze.adaptive_waypoints.boundary_study.event_labels import (
    WALL_CONTACT_THRESHOLD,
    EPS_STATIONARY,
    label_episode,
)
from pldm_envs.diverse_maze.adaptive_waypoints.boundary_study.grid_utils import (
    obs_to_ij,
    wall_adjacent_edge_distance,
)

TURN_WINDOW = 10
TURN_DEGREES_COARSE = 90.0
DECEL_RUN = 3


def collapse_runs(flags: np.ndarray) -> np.ndarray:
    out = np.zeros_like(flags)
    i, n = 0, len(flags)
    while i < n:
        if flags[i]:
            j = i
            while j + 1 < n and flags[j + 1]:
                j += 1
            out[(i + j) // 2] = True
            i = j + 1
        else:
            i += 1
    return out


def label_episode_coarse(obs: np.ndarray, layout: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """Returns {"per_frame": events, "collapsed": events}, each a dict of the
    four (60,) bool arrays (corridor/junction identical to event_labels.py)."""
    orig = label_episode(obs, layout)
    obs = obs[:61]
    xy = obs[:, :2].astype(np.float64)
    v = obs[:, 2:4].astype(np.float64)
    ij = obs_to_ij(xy)
    speed = np.linalg.norm(v, axis=1)

    turn = np.zeros(60, dtype=bool)
    wall = np.zeros(60, dtype=bool)
    for i in range(60):
        f = i + 1
        if f >= TURN_WINDOW:
            a, b = v[f - TURN_WINDOW], v[f]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na >= EPS_STATIONARY and nb >= EPS_STATIONARY:
                ang = np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)))
                if ang >= TURN_DEGREES_COARSE:
                    turn[f - TURN_WINDOW // 2 - 1] = True  # centre frame f-5 -> index f-6
        if f >= DECEL_RUN:
            decel = all(speed[f - k] < speed[f - k - 1] for k in range(DECEL_RUN))
            dist = wall_adjacent_edge_distance(layout, xy[f, 0], xy[f, 1], int(ij[f, 0]), int(ij[f, 1]))
            wall[i] = decel and np.isfinite(dist) and dist <= WALL_CONTACT_THRESHOLD

    per_frame = {
        "wall_contact": wall,
        "direction_turn": turn,
        "corridor_change": orig["corridor_change"],
        "junction_arrival": orig["junction_arrival"],
    }
    collapsed = dict(per_frame)
    collapsed["wall_contact"] = collapse_runs(wall)
    collapsed["direction_turn"] = collapse_runs(turn)
    return {"per_frame": per_frame, "collapsed": collapsed, "original": orig}
