# Boundary-signal study, item 3: "dp_segmentation" cache for the full r50
# dataset. This is NOT cheating -- it uses exactly the same information
# (the real latent trajectory) available to every signal/algorithm in this
# study; it just finds the Metric-B-exact-minimizing 5 boundaries via
# dynamic programming instead of a scalar-signal heuristic. That's a
# legitimate offline data-preprocessing choice for the ADAPTIVE training
# pipeline (AdaptiveD4RLDataset already supports an arbitrary precomputed
# boundaries-per-episode cache -- this is just a different, better signal
# for producing it).
#
# Output format matches d4rl_adaptive.py:24-28 (AdaptiveD4RLDataset.__init__)
# and compute_changepoints.py's own save convention EXACTLY: a dict
# {episode_idx: [b1..b5]} torch-saved as changepoints_minseg{min_seg}.pt
# inside the same directory as data.p, for EACH split (main, probe).
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from signal1_common import get_model, WINDOW  # noqa: E402
from metric_b import oracle_boundaries  # noqa: E402

HERE = os.path.dirname(__file__)
DATA_ROOT = os.path.join(HERE, "..", "..", "datasets", "r50_local", "r50_dataset")
MIN_SEG = 8
N_BOUNDARIES = 5
CKPT_DIR = HERE


def compute_encodings(model, ep, images, ep_start):
    obs = ep["observations"][:WINDOW]
    proprio_vel = torch.from_numpy(obs[:, 2:4]).float().unsqueeze(1)
    img_seq = torch.from_numpy(np.array(images[ep_start : ep_start + WINDOW])).float().permute(0, 3, 1, 2)
    states = img_seq.unsqueeze(1)
    actions_t = torch.from_numpy(ep["actions"][: WINDOW - 1]).float().unsqueeze(1)
    with torch.no_grad():
        result = model.level1.forward_posterior(states, actions_t, proprio_vel=proprio_vel, encode_only=True)
    return result.backbone_output.encodings.squeeze(1).flatten(1).cpu().numpy()


def process_split(model, split_name):
    data_dir = os.path.join(DATA_ROOT, split_name)
    splits = torch.load(os.path.join(data_dir, "data.p"), weights_only=False)
    images = np.load(os.path.join(data_dir, "images.npy"), mmap_mode="r")
    cum_lengths = np.cumsum([len(s["observations"]) for s in splits])

    ckpt_path = os.path.join(CKPT_DIR, f"dp_segmentation_{split_name}_checkpoint.npz")
    results = {}
    start_ep = 0
    if os.path.exists(ckpt_path):
        ck = np.load(ckpt_path, allow_pickle=True)
        start_ep = int(ck["next_idx"])
        results = ck["results"].item()
        print(f"[{split_name}] resuming from episode {start_ep}", flush=True)

    t0 = time.time()
    n_episodes = len(splits)
    for ep_idx in range(start_ep, n_episodes):
        ep_len = len(splits[ep_idx]["observations"])
        if ep_len < WINDOW:
            continue  # too short, AdaptiveD4RLDataset falls back to baseline for these
        ep_start = 0 if ep_idx == 0 else cum_lengths[ep_idx - 1]
        enc = compute_encodings(model, splits[ep_idx], images, ep_start)
        boundaries, _ = oracle_boundaries(enc, N_BOUNDARIES, MIN_SEG)
        results[ep_idx] = boundaries

        if (ep_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (ep_idx + 1 - start_ep) / elapsed
            eta = (n_episodes - ep_idx - 1) / rate if rate > 0 else float("inf")
            np.savez(ckpt_path, next_idx=ep_idx + 1, results=results)
            print(f"[{split_name}] {ep_idx + 1}/{n_episodes} done, {elapsed:.0f}s elapsed, ETA {eta:.0f}s", flush=True)

    out_path = os.path.join(data_dir, f"changepoints_minseg{MIN_SEG}.pt")
    torch.save(results, out_path)
    print(f"[{split_name}] wrote {len(results)} episodes' boundaries to {out_path}", flush=True)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    return out_path


def main():
    model = get_model()
    t_start = time.time()
    paths = {}
    for split_name in ("main", "probe"):
        paths[split_name] = process_split(model, split_name)
    total_time = time.time() - t_start
    print(f"\nTotal wall-clock time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print("Cache file paths (for AdaptiveD4RLDataset -- copy the one matching your actual training data_path's directory):")
    for split_name, path in paths.items():
        print(f"  {split_name}: {path}")


if __name__ == "__main__":
    main()
