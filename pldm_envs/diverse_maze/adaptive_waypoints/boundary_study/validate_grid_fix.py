# One-off validation script (see PREREGISTRATION.md SS2): confirm the
# corrected grid conversion produces zero wall violations on the FULL
# main+probe corpus (not just the 200-episode alignment subsample the
# original diagnosis used).
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from pldm_envs.diverse_maze.adaptive_waypoints.boundary_study.grid_utils import obs_to_ij

BASE = os.path.join(
    os.path.dirname(__file__), "..", "..", "datasets", "r50_local", "r50_dataset"
)


def main():
    total_wall = 0
    total_pts = 0
    for split in ["main", "probe"]:
        splits = torch.load(os.path.join(BASE, split, "data.p"), weights_only=False)
        maps = torch.load(os.path.join(BASE, split, "train_maps.pt"), weights_only=False)
        wall = 0
        pts = 0
        for ep in splits:
            obs = ep["observations"][:, :2].astype(np.float64)
            layout = maps[int(ep["map_idx"])].split("\\")
            rows, cols = len(layout), len(layout[0])
            ij = obs_to_ij(obs)
            for i, j in ij:
                pts += 1
                if 0 <= i < rows and 0 <= j < cols and layout[i][j] == "#":
                    wall += 1
        print(f"{split}: {wall}/{pts} wall violations, open_fraction={1 - wall / pts:.6f}")
        total_wall += wall
        total_pts += pts
    print(f"TOTAL: {total_wall}/{total_pts}, open_fraction={1 - total_wall / total_pts:.6f}")


if __name__ == "__main__":
    main()
