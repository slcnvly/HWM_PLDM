# Causal Waypoint Student -- run summary

Kaggle kernel: `hwm-causal-waypoint-student-pipeline` (2 pushes: v1 hit a
CPU/CUDA device-mismatch bug in stage 3, fixed and re-run as v2; v2 ran
~10h wall clock, mostly stage 5's eval chunks running slower than the
historical ~36min/chunk precedent -- no OOM retries were triggered, GPU
allocation for this session was just slower).

## Stage status (all from v2, the working run)

- Stage 1 (student label generation, full main split, 1250 episodes): **SUCCESS**
- Stage 2 (train causal student): **SUCCESS**
- Stage 3 (generate causal-trigger training labels): **SUCCESS**
- Stage 4 (fine-tune level2 on causal-student labels): **SUCCESS**
- Stage 5 (eval hard n=120): **SUCCESS**

The kernel's own final "assemble eval_comparison.json/png" step failed with
`FileNotFoundError` due to a path bug in `run.py` (`FINAL_EVAL_PATH` was
computed relative to the kernel's cwd `/kaggle/working` instead of
`REPO_DIR=/kaggle/working/HWM_PLDM`, so stage 5's real output landed one
directory away from where `eval_comparison.py` looked). **The eval itself
completed correctly** -- this was purely a path bug in the orchestration
script, recovered from the kernel's own stdout log (every stage logs its
result via `log_progress`, which prints to stdout too) and fixed in
`experiments/kaggle_causal_waypoint_student/run.py` for any future re-run.

## Student model

- Winning target parameterization: **linear** (direct Delta-tau regression beat log1p)
- linear: best val MAE = **2.79 steps** (epoch 14)
- log1p: best val MAE = 2.83 steps (epoch 7)

A ~2.8-step average error in predicting steps-to-next-waypoint, from purely
causal features (e_t, e_{t-1}, a_{t-1}, 8-step surprise history), against a
typical waypoint spacing of ~8-20 steps (see `outputs/waypoint_intervals.png`
from the earlier diagnostics branch).

## Eval comparison (hard difficulty, n=120)

| Condition | Success rate | Avg steps (successes) |
|---|---|---|
| (a) baseline (fixed stride, no fine-tune) | 80.0% (96/120) | 169.6 |
| (b) oracle (pick_changepoints, non-causal) | 92.5% (111/120) | 155.1 |
| (c) causal student (online trigger) | **90.8%** (109/120) | 163.1 |

**(c) beats (a) by +10.8pp** and is only **1.7pp behind (b)** -- the fully
causal, real-time-deployable trigger closes **86.7%** of the gap between
the fixed-stride baseline and the non-causal offline oracle, while itself
requiring no future information (see the causality analysis earlier in this
conversation: `pick_changepoints` needs whole-window future access and
can't run online at all; this result shows a causal approximation gets most
of the way to matching it anyway).

No significance test was run on (c) vs (a)/(b) -- per-trial success/steps
extraction failed for all 6 chunks with `No module named 'pldm'` (same
pre-existing, already-documented bug as RESULTS.md SS8.5's run: the
extraction helper runs under Kaggle's system Python, which doesn't have
`pldm` installed, only the conda `pldm` env does). Only aggregate k/n and
mean-steps survived, same limitation SS8.5 already flagged as unfixed.

## Artifacts

- `outputs/causal_student_eval_final.json` -- stage 5's aggregate result (k=109, n=120, avg_steps=163.1)
- `outputs/eval_comparison.json` / `outputs/eval_comparison.png` -- the 3-way comparison
- `checkpoints/student/student_final.pt` -- trained causal Delta-tau regressor (not committed to git -- large binary artifact, stays on Kaggle; re-generatable via `pldm_envs/diverse_maze/adaptive_waypoints/student_model.py`)
- `main_causal/changepoints_minseg8.pt` (on Kaggle only) -- the causal-trigger-derived training labels used for stage 4's fine-tune
