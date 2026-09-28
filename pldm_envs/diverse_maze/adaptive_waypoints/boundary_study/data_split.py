# Boundary-signal study (see PREREGISTRATION.md SS7). Map-based
# selection/report split: r50's 25 maps split in half so no map's layout
# appears in both halves. Persisted once so Stages 2-4 never re-derive it
# differently.
import json
import os
from collections import defaultdict

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DATA_ROOT = os.path.join(
    REPO_ROOT, "pldm_envs", "diverse_maze", "datasets", "r50_local", "r50_dataset"
)
OUT_PATH = os.path.join(os.path.dirname(__file__), "split.json")


def build_split(seed: int = 0):
    rng = torch.Generator().manual_seed(seed)

    # Collect (split_name, episode_idx, map_idx) for both main and probe;
    # main's map_idx values and probe's are drawn from the same 25-map pool
    # (both train_maps.pt copies are identical -- verified below).
    main_maps = torch.load(os.path.join(DATA_ROOT, "main", "train_maps.pt"), weights_only=False)
    probe_maps = torch.load(os.path.join(DATA_ROOT, "probe", "train_maps.pt"), weights_only=False)
    assert main_maps == probe_maps, "main/probe map pools differ -- split logic assumes a shared pool"
    n_maps = len(main_maps)

    all_map_ids = list(range(n_maps))
    perm = torch.randperm(n_maps, generator=rng).tolist()
    half = n_maps // 2
    selection_maps = set(perm[:half])
    report_maps = set(perm[half:])
    assert selection_maps.isdisjoint(report_maps)
    assert selection_maps | report_maps == set(all_map_ids)

    result = {"n_maps": n_maps, "selection_maps": sorted(selection_maps), "report_maps": sorted(report_maps)}

    for split_name in ("main", "probe"):
        splits = torch.load(os.path.join(DATA_ROOT, split_name, "data.p"), weights_only=False)
        sel_eps, rep_eps = [], []
        counts = defaultdict(int)
        for ep_idx, ep in enumerate(splits):
            m = int(ep["map_idx"])
            counts[m] += 1
            if m in selection_maps:
                sel_eps.append(ep_idx)
            else:
                rep_eps.append(ep_idx)
        result[split_name] = {"selection_episodes": sel_eps, "report_episodes": rep_eps}
        print(
            f"{split_name}: {len(sel_eps)} selection eps, {len(rep_eps)} report eps "
            f"(total {len(splits)}, {n_maps} maps, {len(selection_maps)}/{len(report_maps)} sel/report maps)"
        )

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {OUT_PATH}")
    return result


if __name__ == "__main__":
    build_split()
