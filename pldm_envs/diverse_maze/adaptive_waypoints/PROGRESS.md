# adaptive_waypoints progress / decision log (inference-design study)

The boundary-signal study keeps its own trail in `boundary_study/PROGRESS.md`. This file records the decisions
made while running the inference-design study unattended (user instruction 2026-10-05: follow the stated rules;
anything not covered gets the conservative choice, recorded here).

## Decisions

1. **Location prober for waypoint decoding.** The prober handed to the planner in eval-only mode is randomly
   initialised (`pldm/probing/evaluator.py:272-273`: `train_probers = eval_l1 = false` returns untrained probers).
   A diagnostic prober of the same architecture is trained locally on CPU on frozen pretrained-L1 features from
   the r50 probe split. No model weights are trained (the "no fine-tuning" rule is about the model).
2. **GPU quota accounting.** Remaining = allowed − used − (my estimate of the remaining time of my running
   kernels). Kaggle's own `time_reserved` is recorded alongside but not used, because its semantics are
   undocumented.
3. **Checkpoint.** `pretrained_baseline.ckpt` from dataset `seungwonryoo/hwm-finetune-checkpoints` (the file
   used for the SS6b baseline n=120), not a fresh HF download, so V0 is comparable with SS6b.
