# Boundary-signal study, Stage 1 driver (see PREREGISTRATION.md SS4).
# Computes the 4 ground-truth event labels for every episode in main+probe,
# logs per-event-type frequency stats, and renders 10 random trajectories
# with events overlaid on their maze for visual sanity-checking.
#
# Run with: /home/goodwon01/.venvs/pldm_boundary/bin/python run_stage1_events.py
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from pldm_envs.diverse_maze.adaptive_waypoints.boundary_study.event_labels import label_episode
from pldm_envs.diverse_maze.adaptive_waypoints.boundary_study.grid_utils import obs_to_ij

os.environ.setdefault("WANDB_MODE", "offline")  # not logged in yet -- see PROGRESS.md

HERE = os.path.dirname(__file__)
DATA_ROOT = os.path.join(HERE, "..", "..", "datasets", "r50_local", "r50_dataset")
PLOTS_DIR = os.path.join(HERE, "plots")
EVENT_TYPES = ["wall_contact", "direction_turn", "corridor_change", "junction_arrival"]


def compute_all_labels():
    """Returns {split: [{'events': {type: (60,) bool}, 'map_idx': int}, ...]}."""
    out = {}
    for split in ("main", "probe"):
        splits = torch.load(os.path.join(DATA_ROOT, split, "data.p"), weights_only=False)
        maps = torch.load(os.path.join(DATA_ROOT, split, "train_maps.pt"), weights_only=False)
        ep_results = []
        for ep in splits:
            layout = maps[int(ep["map_idx"])].split("\\")
            events = label_episode(ep["observations"], layout)
            ep_results.append({"events": events, "map_idx": int(ep["map_idx"])})
        out[split] = ep_results
        print(f"{split}: labeled {len(ep_results)} episodes", flush=True)
    return out


def summarize(all_labels):
    """Per-event-type: total count, mean count per trajectory, fraction of
    steps positive -- pooled across main+probe."""
    summary = {}
    n_eps_total = 0
    for etype in EVENT_TYPES:
        total = 0
        per_traj_counts = []
        for split, eps in all_labels.items():
            for ep in eps:
                c = int(ep["events"][etype].sum())
                total += c
                per_traj_counts.append(c)
        n_eps_total = len(per_traj_counts)
        summary[etype] = {
            "total_count": total,
            "mean_per_trajectory": float(np.mean(per_traj_counts)),
            "std_per_trajectory": float(np.std(per_traj_counts)),
            "frac_steps_positive": total / (n_eps_total * 60),
        }
    summary["_n_episodes"] = n_eps_total
    return summary


def render_examples(all_labels, n_examples=10, seed=0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pldm_envs.diverse_maze.adaptive_waypoints.boundary_study.grid_utils import (
        GRID_SIZE,
        OBS_MIN_TOTAL,
    )

    os.makedirs(PLOTS_DIR, exist_ok=True)
    rng = random.Random(seed)

    main_splits = torch.load(os.path.join(DATA_ROOT, "main", "data.p"), weights_only=False)
    main_maps = torch.load(os.path.join(DATA_ROOT, "main", "train_maps.pt"), weights_only=False)
    ep_indices = rng.sample(range(len(main_splits)), n_examples)

    markers = {
        "wall_contact": ("x", "red"),
        "direction_turn": ("^", "orange"),
        "corridor_change": ("s", "blue"),
        "junction_arrival": ("*", "green"),
    }

    saved_paths = []
    for ep_idx in ep_indices:
        ep = main_splits[ep_idx]
        layout = main_maps[int(ep["map_idx"])].split("\\")
        rows, cols = len(layout), len(layout[0])
        obs = ep["observations"][:61]
        xy = obs[:, :2]
        events = all_labels["main"][ep_idx]["events"]

        fig, ax = plt.subplots(figsize=(6, 6))
        # draw maze as a grid of cells: layout[i][j], i=row (x-axis in obs_to_ij's
        # convention), j=col (y-axis) -- plot with i on the vertical axis to match.
        grid_img = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                grid_img[i, j] = 0.0 if layout[i][j] == "O" else 1.0
        ax.imshow(grid_img, cmap="gray_r", origin="lower", extent=[0, cols, 0, rows], alpha=0.3)

        # trajectory in grid-cell-continuous coordinates
        plot_x = (xy[:, 1] - OBS_MIN_TOTAL) / GRID_SIZE  # j (column) -> plot x-axis
        plot_y = (xy[:, 0] - OBS_MIN_TOTAL) / GRID_SIZE  # i (row) -> plot y-axis
        ax.plot(plot_x, plot_y, "-", color="black", linewidth=1, alpha=0.6)

        for etype, (marker, color) in markers.items():
            idx = np.where(events[etype])[0]
            if len(idx) == 0:
                continue
            frames = idx + 1  # event index i labels frame i+1
            ax.scatter(plot_x[frames], plot_y[frames], marker=marker, color=color, s=60, label=etype, zorder=5)

        ax.set_title(f"episode {ep_idx} (map {ep['map_idx']})")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        path = os.path.join(PLOTS_DIR, f"event_overlay_ep{ep_idx}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(path)
        print(f"saved {path}", flush=True)

    return saved_paths


def main():
    all_labels = compute_all_labels()
    summary = summarize(all_labels)
    print(json.dumps(summary, indent=2))

    out_path = os.path.join(HERE, "results_stage1_events.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_path}")

    example_paths = render_examples(all_labels)

    try:
        import wandb

        run = wandb.init(project="hwm-boundary-study", job_type="stage1_events", name="stage1-event-labels")
        table = wandb.Table(
            columns=["event_type", "total_count", "mean_per_trajectory", "std_per_trajectory", "frac_steps_positive"]
        )
        for etype in EVENT_TYPES:
            s = summary[etype]
            table.add_data(etype, s["total_count"], s["mean_per_trajectory"], s["std_per_trajectory"], s["frac_steps_positive"])
        run.log({"event_frequency": table})
        run.log({"trajectory_examples": [wandb.Image(p) for p in example_paths]})
        run.finish()
        print("wandb logged (offline)" if os.environ.get("WANDB_MODE") == "offline" else "wandb logged")
    except Exception as e:
        print(f"wandb logging failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
