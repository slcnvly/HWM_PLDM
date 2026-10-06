# Recompute the SS7 error-adaptive-L1 threshold with normalized inputs (2026-10-06 fix).
# The original (raw-input) value stays in error_threshold.json.
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
from pldm_envs.diverse_maze.adaptive_waypoints.error_adaptive_l1 import compute_r50_error_threshold  # noqa: E402

D = os.path.join(HERE, "..", "datasets", "r50_local", "r50_dataset")
mean, std, thr, n = compute_r50_error_threshold(
    data_path=os.path.join(D, "main"),
    config_path=os.path.join(HERE, "..", "..", "..", "pldm/configs/diverse_maze/icml/large_diverse_25maps_l2.yaml"),
    checkpoint_path=os.path.join(D, "3-9-1-seed248_epoch=3_sample_step=15465472.ckpt"),
    device="cpu",
)
json.dump({"mean": mean, "std": std, "threshold": thr, "n_steps": n, "n_sigma": 2.0, "inputs": "normalized (preprocess.py)"},
          open(os.path.join(HERE, "error_threshold_fixed.json"), "w"), indent=2)
print(mean, std, thr, n)
