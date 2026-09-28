# Boundary-signal study, item 4: how far does the obs_to_ij scale bug
# (PREREGISTRATION.md SS2 / Amendment 1 context) reach into this project's
# actual evaluation results?
#
# Callers of obs_to_ij / sample_nearby_grid_location_v2 in the repo (grep
# confirmed, see PROGRESS.md):
#   1. pldm_envs/diverse_maze/wrappers.py:104-110 (xy_to_ij) -- used by
#      find_shortest_path (BFS shortest-path/turn-counting), itself used
#      inside DiverseMazeWrapper's step-level bookkeeping.
#   2. pldm_envs/diverse_maze/evaluation/maze2d_envs_generator.py:53
#      (Maze2DEnvsGenerator._sample_nearby_location) -- calls
#      sample_nearby_grid_location_v2 with RENDER_STATS's BUGGY
#      obs_range_total, and its output feeds directly into env.set_target().
#
# BUT: for S5-S8's actual hard-difficulty (D13-16) instances, the icml yaml
# (large_diverse_25maps_l2.yaml, `hard:` block) sets
# `set_start_target_path=".../maze2d_large_diverse_probe/starts_targets_13_16.pt"`
# and `override_config: true` -- meaning these 40 instances were NOT
# generated live via Maze2DEnvsGenerator at eval time at all; they were
# pre-saved (recovered locally, byte-identical copy, see PROGRESS.md) and
# just loaded. So the LIVE bug-affected code path (finding #2 above) is
# NOT what produced S5-S8's real starts/targets for hard difficulty -- this
# script checks the pre-saved file's actual validity directly instead.
#
# This does NOT modify any original file -- read-only analysis.
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from grid_utils import obs_to_ij, open_neighbor_count  # noqa: E402

HERE = os.path.dirname(__file__)

STARTS_TARGETS_PATH = (
    "/home/goodwon01/hwm-experiment/experiments/kaggle_run_pipeline_test/output/"
    "HWM_PLDM/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_probe/starts_targets_13_16.pt"
)

# The buggy constants (pre-fix), exactly as maze2d_large_diverse's
# RENDER_STATS entry has them -- used to check what the ORIGINAL code would
# have believed about validity, for comparison against the corrected check.
BUGGY_OBS_MIN_TOTAL = -0.636125
BUGGY_OBS_RANGE_TOTAL = 10.068675
N = 10


def buggy_obs_to_ij(xy):
    grid_size = BUGGY_OBS_RANGE_TOTAL / N
    return np.floor((xy - BUGGY_OBS_MIN_TOTAL) / grid_size).astype(int)


def classify(xy, layout):
    """Returns (on_wall, adjacent_to_wall) using the given conversion's ij."""
    rows, cols = len(layout), len(layout[0])
    i, j = xy
    if not (0 <= i < rows and 0 <= j < cols):
        return True, True  # out of bounds counts as "on wall" (invalid either way)
    if layout[i][j] == "#":
        return True, False
    n_open = open_neighbor_count(layout, i, j)
    adjacent = n_open < 4
    return False, adjacent


def main():
    if not os.path.exists(STARTS_TARGETS_PATH):
        print(f"NOT FOUND: {STARTS_TARGETS_PATH}")
        return
    data = torch.load(STARTS_TARGETS_PATH, weights_only=False)
    n = len(data["starts"])
    print(f"loaded {n} start/target pairs from starts_targets_13_16.pt (S5-S8's real hard-difficulty D13-16 set)")

    rows = []
    for idx in range(n):
        start_xy = np.array(data["starts"][idx][:2], dtype=np.float64)
        target_xy = np.array(data["targets"][idx][:2], dtype=np.float64)
        layout = data["map_layouts"][idx].split("\\")

        for label, xy in (("start", start_xy), ("target", target_xy)):
            ij_fixed = tuple(int(v) for v in obs_to_ij(xy))
            ij_buggy = tuple(int(v) for v in buggy_obs_to_ij(xy))
            on_wall_fixed, adj_fixed = classify(ij_fixed, layout)
            on_wall_buggy, adj_buggy = classify(ij_buggy, layout)
            rows.append({
                "idx": idx,
                "point": label,
                "xy": xy.tolist(),
                "ij_fixed": ij_fixed,
                "ij_buggy": ij_buggy,
                "on_wall_fixed": on_wall_fixed,
                "adjacent_to_wall_fixed": adj_fixed,
                "on_wall_buggy": on_wall_buggy,
                "adjacent_to_wall_buggy": adj_buggy,
                "ij_differs": ij_fixed != ij_buggy,
            })

    n_total = len(rows)
    n_on_wall_fixed = sum(r["on_wall_fixed"] for r in rows)
    n_adj_fixed = sum(r["adjacent_to_wall_fixed"] and not r["on_wall_fixed"] for r in rows)
    n_either_fixed = sum(r["on_wall_fixed"] or r["adjacent_to_wall_fixed"] for r in rows)
    n_ij_differs = sum(r["ij_differs"] for r in rows)

    n_starts = sum(1 for r in rows if r["point"] == "start")
    n_targets = sum(1 for r in rows if r["point"] == "target")
    starts_either = sum(1 for r in rows if r["point"] == "start" and (r["on_wall_fixed"] or r["adjacent_to_wall_fixed"]))
    targets_either = sum(1 for r in rows if r["point"] == "target" and (r["on_wall_fixed"] or r["adjacent_to_wall_fixed"]))

    # The headline "adjacent to wall" number above is expected to be ~100%
    # regardless of the bug -- maze corridors are narrow, so nearly every
    # open cell touches >=1 wall cell by construction (verified separately:
    # 36/36 = 100% of map 0's own open cells qualify). The metric that
    # actually isolates the bug's effect is whether buggy vs fixed DISAGREE
    # on validity for the same real point.
    n_buggy_false_positive_wall = sum(
        1 for r in rows if r["on_wall_buggy"] and not r["on_wall_fixed"]
    )  # bug would have wrongly rejected a real, valid point as "inside a wall"
    n_buggy_false_negative_wall = sum(
        1 for r in rows if r["on_wall_fixed"] and not r["on_wall_buggy"]
    )  # bug would have wrongly accepted an actually-invalid point

    summary = {
        "n_instances": n,
        "n_points_total": n_total,
        "on_wall_fixed_count": n_on_wall_fixed,
        "adjacent_to_wall_only_fixed_count": n_adj_fixed,
        "either_fixed_count": n_either_fixed,
        "either_fixed_fraction": n_either_fixed / n_total,
        "ij_differs_buggy_vs_fixed_count": n_ij_differs,
        "ij_differs_fraction": n_ij_differs / n_total,
        "starts": {"n": n_starts, "either_count": starts_either, "either_fraction": starts_either / n_starts},
        "targets": {"n": n_targets, "either_count": targets_either, "either_fraction": targets_either / n_targets},
        "adjacent_to_wall_metric_caveat": (
            "either_fixed_fraction=1.0 (100%) is NOT evidence of a problem -- "
            "verified separately that 36/36 (100%) of map 0's own open cells "
            "touch >=1 wall cell, i.e. this maze's corridors are narrow enough "
            "that nearly every open cell qualifies as 'adjacent to a wall' "
            "regardless of the bug. The metric below is the one that actually "
            "isolates the bug's effect."
        ),
        "buggy_vs_fixed_disagreement": {
            "ij_differs_count": n_ij_differs,
            "ij_differs_fraction": n_ij_differs / n_total,
            "buggy_false_positive_wall_count": n_buggy_false_positive_wall,
            "buggy_false_positive_wall_note": (
                "count of real, physically-valid points (on_wall_fixed=False) "
                "that the BUGGY conversion would have wrongly flagged as "
                "inside a wall (on_wall_buggy=True) -- i.e. if anything "
                "downstream used the buggy obs_to_ij to validate these exact "
                "points, it would incorrectly reject them."
            ),
            "buggy_false_negative_wall_count": n_buggy_false_negative_wall,
        },
        "note": (
            "These 40 instances came from a pre-saved file "
            "(starts_targets_13_16.pt, set_start_target_path in the icml yaml's "
            "hard: block, override_config=true) -- Maze2DEnvsGenerator's live "
            "sample_nearby_grid_location_v2 call (which DOES use the buggy "
            "obs_range_total) was NOT invoked to produce these specific "
            "instances. This checks the pre-saved file's actual physical "
            "validity directly, using the corrected conversion, and separately "
            "reports what the buggy conversion would have concluded, for "
            "reference. The 80 extra seed=20260910 instances (kaggle_hard_n120) "
            "were NOT recoverable locally (not in any locally-cached _output_.zip; "
            "reproducing them would need Maze2DEnvsGenerator -> ant_draw -> gym, "
            "none of which are installed in this lightweight analysis venv) -- "
            "not included in this check."
        ),
    }

    out_path = os.path.join(HERE, "results_bug_impact_scope.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_point": rows}, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
