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
4. **Deterministic algorithms are enabled with `warn_only=True`.** A strict `use_deterministic_algorithms(True)`
   raises on any op without a deterministic kernel and would kill a 7h run. warn_only keeps the deterministic
   kernels where they exist and logs the rest. The remaining nondeterminism is measured by the step-1 validation.
5. **CRN seed = 0**, trial id = global index 0..119 in the SS6b order. Stage-1 replans, stage-2 replans and the
   initial nominal sequence use distinct generator keys (`stage` in {init, s1, s2}, `level` in {l1, l2, flat}).
6. **Per-env encoding and nominal rollout under CRN.** Batched conv results can depend on batch size. With CRN on,
   the L1 encoder (current obs and goal) and the nominal-trajectory rollout run per environment, so a trial's
   computation is identical whatever the chunk size. MPPI itself was already per environment.

## FLAG for the user (found while preparing C1; not fixed tonight, by the conservative rule)

**The boundary study and the changepoint-cache scripts feed raw, unnormalized inputs to the model.**
Training normalizes images with `state_mean/std`, plus actions and proprio velocity (`pldm/data/utils.py:97,117` →
`Normalizer.normalize_sample`, `pldm_envs/utils/normalizer.py:344-380`), and so does evaluation
(`maze_draw.py:104-105`, `wrappers.py:297-302`). But `compute_changepoints.py:126-133` and the boundary-study
helpers (`signal1_common.py:36-40`, `run_stage2_predictor_free.py:63-71`, and therefore
`generate_dp_segmentation_cache.py`) pass raw 0–255 images, raw velocity and raw actions. Consequences to
re-check: the signal-1 changepoint cache used for SS5/SS6b adaptive training, the surprise cache (SS8), every
boundary-study encoding (Metric A/B, gate checks), and the dp_segmentation cache used in SS9. Whether this
changed the conclusions is unknown until re-run with normalized inputs. All new inference-study CPU code (prober,
C1) normalizes inputs as evaluation does.

## 2026-10-06 follow-up (IN PROGRESS when the session's usage limit was reached)

- V5 launched on Kaggle (user waived the +2h buffer): kernel `hwm-inference-v5`, ~7.6h; resumable per 20-instance chunk.
- Running locally (nohup, logs next to scripts): corrected dp cache for full r50 (`boundary_study/dp_cache_fixed_{main,probe}.log`;
  old cache preserved as `changepoints_minseg8_dp_v1.pt`), corrected signal-1 cache (`signal1_cache_fixed_*.log` →
  `datasets/r50_local/signal1_fixed/`), Metric A / coarse / 16v2 reruns (`boundary_study/run_metric_a*_fixed.log`; old results `*_v1.json`),
  follow-up 1+2 (`followup/action_sensitivity_and_surprise.py`). Old signal-1 cache download → `experiments/old_signal1_cache/`.
- Kaggle CPU kernel `hwm-render-v0-states` renders V0 replan states for follow-up 3 (V0 part).
- Written, not yet run: `followup/decision_points.py` (task 3; r50 reduced to 100 episodes and V0 to backward replans + equal random
  others, because of CPU cost ~2.2 s/state). Not yet written: task 4 (linear BFS-distance probe), error_adaptive_l1.py / variance_head.py
  code fixes, Metric A comparison table, DECISION_POINTS.md, next-week GPU plan.
