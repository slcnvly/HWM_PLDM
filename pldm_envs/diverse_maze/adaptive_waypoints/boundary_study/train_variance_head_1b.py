# Boundary-signal study, item 3: retrain SS8's post-hoc variance head on
# signal 1b (re-anchored one-step error, Amendment 1) instead of the
# original signal 1. Reuses variance_head.py's VarianceHead/beta_nll_loss/
# train_variance_head machinery unchanged -- only the dataset-building step
# differs (signal 1b's errors + the real z_t each was computed from, instead
# of compute_error_and_encoding_series's signal-1 pairing).
#
# Unlike the original SS8 run, this one SAVES the trained weights to disk
# and commits them -- the original weights were never persisted (only the
# JSON diagnostics summary survived), which is why this had to be retrained
# from scratch at all.
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from signal1_common import get_model, iter_episodes, compute_episode  # noqa: E402
from pldm_envs.diverse_maze.adaptive_waypoints.variance_head import (  # noqa: E402
    constant_baseline_nll,
    train_variance_head,
)

HERE = os.path.dirname(__file__)

# Memory-safe subsample: this machine has ~14GB free RAM (checked via
# `free -h`), well below what Kaggle's kernels had for the original SS8 run
# (which OOM'd twice even there -- see RESULTS.md SS8.3). Full main+probe
# encodings alone would be ~8.7GB in fp16; keeping a wide safety margin.
N_TRAIN_EPISODES = 500  # from main
N_VAL_EPISODES = 400  # from probe


def build_dataset_1b(model, split_name, limit):
    """Returns (err1b: (N,) float32 tensor, encodings_before: (N, D) fp16
    tensor) -- same pairing convention as variance_head.py's build_dataset
    (err at step t paired with the real z_t it was computed FROM), but using
    signal 1b's re-anchored error instead of signal 1's."""
    all_err1b, all_enc_before = [], []
    n_done = 0
    for ep_idx, ep, images, ep_start in iter_episodes(split_name, limit=limit):
        _err1, err1b, encodings = compute_episode(model, ep, images, ep_start)
        all_err1b.append(err1b)  # (60,)
        all_enc_before.append(encodings[:60])  # (60, D) -- z_t for t=0..59
        n_done += 1
        if n_done % 100 == 0:
            print(f"[1b dataset build: {split_name}] episode {n_done}/{limit}", flush=True)

    err1b = torch.from_numpy(np.concatenate(all_err1b)).float()  # (N,)
    encodings_before = torch.from_numpy(np.concatenate(all_enc_before)).half()  # (N, D) fp16
    return err1b, encodings_before


def main():
    model = get_model()

    print("building train (main) dataset...")
    train_err1b, train_enc = build_dataset_1b(model, "main", N_TRAIN_EPISODES)
    print(f"train: {train_err1b.shape[0]} samples, encodings dim {train_enc.shape[1]}")

    print("building val (probe) dataset...")
    val_err1b, val_enc = build_dataset_1b(model, "probe", N_VAL_EPISODES)
    print(f"val: {val_err1b.shape[0]} samples")

    # Step A: constant baseline vs trained MLP (same table shape as RESULTS.md SS8.4)
    const_sigma2, const_val_nll = constant_baseline_nll(train_err1b, val_err1b)
    _, const_train_nll = constant_baseline_nll(train_err1b, train_err1b)
    print(f"constant baseline: sigma^2={const_sigma2:.4f}, val_nll={const_val_nll:.4f}, train_nll={const_train_nll:.4f}")

    trained_model, input_mean, input_std, target_scale, log = train_variance_head(
        train_errs=train_err1b,
        train_encs=train_enc,
        val_errs=val_err1b,
        val_encs=val_enc,
        device="cpu",
    )

    # Save weights + everything needed to reload/predict later -- this is
    # the thing that was missing from the original SS8 run.
    ckpt_path = os.path.join(HERE, "variance_head_1b.pt")
    torch.save(
        {
            "state_dict": trained_model.state_dict(),
            "input_mean": input_mean,
            "input_std": input_std,
            "target_scale": target_scale,
            "hidden": 128,
            "input_dim": train_enc.shape[1],
            "trained_on": "signal_1b (re-anchored one-step error, Amendment 1)",
            "n_train_episodes": N_TRAIN_EPISODES,
            "n_val_episodes": N_VAL_EPISODES,
        },
        ckpt_path,
    )
    print(f"saved {ckpt_path}")

    results = {
        "n_train_episodes": N_TRAIN_EPISODES,
        "n_val_episodes": N_VAL_EPISODES,
        "n_train_samples": int(train_err1b.shape[0]),
        "n_val_samples": int(val_err1b.shape[0]),
        "constant_baseline": {"sigma2": const_sigma2, "val_nll": const_val_nll, "train_nll": const_train_nll},
        "trained_mlp": {"best_epoch": log["best_epoch"], "best_val_nll_raw": log["best_val_nll_raw"]},
        "per_epoch_log": log["per_epoch"],
    }
    out_path = os.path.join(HERE, "results_variance_head_1b.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")
    print(f"\nStep A summary: constant val_nll={const_val_nll:.4f} vs trained MLP val_nll={log['best_val_nll_raw']:.4f} (best epoch {log['best_epoch']})")


if __name__ == "__main__":
    main()
