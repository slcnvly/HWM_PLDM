# Boundary-signal study (see PREREGISTRATION.md SS7, corrected below).
# Map-based selection/report split.
#
# CORRECTION (discovered when first actually running this script): SS7
# assumed r50's main and probe splits draw episodes from the SAME 25-map
# pool, so a single map-level permutation could be applied uniformly to
# both. That assumption is wrong -- checked directly: main has 25 distinct
# maps (map_idx 0-24), probe has 20 DIFFERENT distinct maps (map_idx 0-19,
# but `main_maps[k] != probe_maps[k]` for every shared key k -- the map_idx
# numbering is independent per split, not a shared pool with different
# episode counts). This is standard for D4RL-style "diverse" datasets:
# train (main) and validation (probe) intentionally use disjoint maze
# layouts to test generalization to unseen maps.
#
# Given that, main and probe are ALREADY exactly the kind of disjoint,
# no-map-overlap split SS7 was trying to construct by splitting one pool in
# half -- so the corrected design is simpler and arguably more faithful to
# the original intent (avoid overfitting signal *selection* to the same
# maps used for the final *report*): **main = selection set, probe = report
# set**, no further per-half split needed. No map ever appears in both
# (verified: main's 25 map layouts and probe's 20 are entirely distinct
# mazes, confirmed by comparing layout strings, not just indices).
import json
import os

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DATA_ROOT = os.path.join(
    REPO_ROOT, "pldm_envs", "diverse_maze", "datasets", "r50_local", "r50_dataset"
)
OUT_PATH = os.path.join(os.path.dirname(__file__), "split.json")


def build_split():
    main_maps = torch.load(os.path.join(DATA_ROOT, "main", "train_maps.pt"), weights_only=False)
    probe_maps = torch.load(os.path.join(DATA_ROOT, "probe", "train_maps.pt"), weights_only=False)

    # Confirm the layouts themselves are disjoint (not just that the index
    # sets happen to look similar) -- this is the actual guarantee we need.
    main_layouts = set(main_maps.values())
    probe_layouts = set(probe_maps.values())
    overlap = main_layouts & probe_layouts
    assert not overlap, f"main/probe share {len(overlap)} identical map layouts -- selection/report would leak"

    main_splits = torch.load(os.path.join(DATA_ROOT, "main", "data.p"), weights_only=False)
    probe_splits = torch.load(os.path.join(DATA_ROOT, "probe", "data.p"), weights_only=False)

    result = {
        "design": "main=selection (25 maps), probe=report (20 different maps) -- see file header for why this replaces SS7's original half-split-one-pool plan",
        "n_selection_maps": len(main_maps),
        "n_report_maps": len(probe_maps),
        "main": {"selection_episodes": list(range(len(main_splits))), "report_episodes": []},
        "probe": {"selection_episodes": [], "report_episodes": list(range(len(probe_splits)))},
    }
    print(f"selection (main): {len(main_splits)} episodes, {len(main_maps)} maps")
    print(f"report (probe): {len(probe_splits)} episodes, {len(probe_maps)} maps")
    print(f"confirmed disjoint: 0 shared map layouts between selection and report")

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {OUT_PATH}")
    return result


if __name__ == "__main__":
    build_split()
