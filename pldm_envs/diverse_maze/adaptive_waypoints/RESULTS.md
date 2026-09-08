# Results: Changepoint-based Adaptive Waypoints

Branch: `adaptive-waypoints`. See [DESIGN.md](DESIGN.md) for the hypothesis and
implementation decisions. This document covers what was actually run and found.

## 1. Summary

Adaptive (prediction-error-driven) waypoint placement, fine-tuned for 2 epochs
from the paper's own pretrained checkpoint, **matches or beats** the fixed-stride
baseline on real hierarchical MPPI planning in D4RL Diverse Maze, evaluated at
the paper's own scale (n=40 environments per difficulty):

| | Medium (D 9-12) | Hard (D 13-16) |
|---|---|---|
| **Paper (Table 3, PLDM hierarchical)** | 95% | 83% |
| **Our baseline** (paper's pretrained checkpoint, unmodified) | 90% (36/40), avg 130.2 steps | 82.5% (33/40), avg 161.2 steps |
| **Our adaptive, min_seg=8** (2-epoch fine-tune from the same checkpoint) | 90% (36/40), avg **101.9** steps | **92.5%** (37/40), avg 145.9 steps |

Our baseline reproduction (90%/82.5%) lands close to the paper's own numbers
(95%/83%), which is itself a useful sanity check on the whole pipeline. Against
that baseline, adaptive segmentation:
- ties on medium success rate but reaches the goal in **~22% fewer steps**
  (130.2 → 101.9 avg)
- **beats baseline on hard by +10 points** (82.5% → 92.5%) and is also faster
  (161.2 → 145.9 avg steps)

n=40 per condition per difficulty (matching the paper's own protocol) is a real
sample size, not the n=4 pilot this project started Stage 2 with -- see §4 for
how that pilot result (baseline 75%, adaptive 100%, both on n=4) motivated
re-running at full scale before treating anything as evidence.

## 2. Experiment design

**Stage 1 -- training-loss / held-out-validation screening across min_seg.**
Adaptive waypoint placement has one real hyperparameter, `min_seg` (minimum
spacing enforced between changepoints -- see DESIGN.md §4). Rather than guess a
value, trained baseline + 3 adaptive variants (min_seg in {3, 5, 8}) from
scratch for 6 epochs on a stratified 50-episodes/map subset ("r50": 1250
episodes / 25 maps main, 1000 episodes / 20 maps probe) and compared both
final training loss and a held-out validation metric (an L2 location-prediction
probe the training config runs automatically each epoch):

| min_seg | final train loss | held-out probe val loss (lower=better) |
|---|---|---|
| baseline (fixed stride) | 3.036 | 0.152 |
| 8 | 3.081 | 0.207 |
| 5 | 4.082 | 0.236 |
| 3 | 4.000 | 0.255 |

min_seg=8 was the clear best of the 3 adaptive variants on both metrics and
was carried forward to Stage 2. (min_seg=3/5 spacing is small enough that
changepoints cluster close together -- segments get very short and need
aggressive interpolation up to the fixed 10-step encoder input, which is the
likely explanation, though not independently verified.)

**Stage 2 -- real planning eval.** The first attempt trained baseline and the
adaptive variant from scratch on the same small r50 subset (6 epochs) and
evaluated live hierarchical MPPI planning -- both conditions scored an
identical, uninformative 0% success rate. Root cause, found by inspecting the
codebase rather than guessing: `load_l1_only: true` in the training config
deletes every level-2 weight from *any* checkpoint before loading it, even one
that already has a fully pretrained level2. The HF repo
(`kevinghst/pldm-maze2d-large-diverse`) hosts exactly such a checkpoint --
`load_from_l1248-seed248_epoch=5_sample_step=10789632.ckpt`, already fetched by
`download_ckpt_from_hf.py` by default but never actually pointed at by
`load_checkpoint_path` with `load_l1_only=false` anywhere in the existing
configs. This is presumably the checkpoint the paper's own Table 3 numbers came
from (`sample_step` ~10.8M vs the from-scratch run's ~300K -- roughly 36x more
training).

Once found, Stage 2 switched to: (a) evaluate that pretrained checkpoint
as-is (baseline, no training at all), (b) fine-tune *from* that same
checkpoint on the adaptive_min_seg=8 dataset for 2 epochs at reduced LR
(`base_lr` x0.1) rather than training a fresh level2 from scratch, (c)
evaluate the fine-tuned checkpoint the same way. This tests "does adaptive
segmentation help or hurt an already-good model" rather than "can two
undertrained models be told apart" -- a fine-tune budget was chosen
specifically because retraining at the paper's full scale was not feasible on
free-tier Kaggle compute.

**Evaluation settings** (both conditions, both stages of Stage 2): D4RL Diverse
Maze, hierarchical (L2) MPPI planning, `n_envs=40` (medium and hard difficulty
separately, matching the paper's own protocol), `n_steps=500`,
`level2.mppi.num_samples=200` (reduced from the paper's 2000-4000 for compute
budget -- this is the one place our protocol still diverges from the paper's).

## 3. Infra notes (for whoever touches this next)

Getting Stage 2 working end-to-end took ~20 debugging iterations. The two
worth knowing before re-running anything:

- **`data.num_workers=0` is required.** The base config sets
  `num_workers=12`, `prefetch_factor=6` for every D4RL dataset (5 get built
  per run: main + 4 probing splits, the latter built unconditionally
  regardless of eval flags). On Kaggle's free-tier RAM this reliably OOM-kills
  (`Killed`, exit 137 -- the Linux OOM killer, not a catchable CUDA-OOM) once
  total resident memory climbs past ~15GB, which happens well before any
  single "expensive" step. Setting `data.num_workers=0` /
  `data.d4rl_config.num_workers=0` fixed every OOM this project hit, in both
  Stage 1 and Stage 2 (training and eval-only alike).
- **`eval_l2=true` is already the config default** (not something this
  project's changes turned on) -- every training run also trains+validates an
  L2 location-prediction probe each epoch whether you want it or not, unless
  a new `eval_cfg.probe_l2` flag (added on this branch, in
  `pldm/evaluation/evaluator.py`) is explicitly set to `false`. This is a nice
  bonus (a real held-out metric for free) but worth knowing since it's what
  produced the Stage 1 validation-loss numbers above without ever being asked
  for.

Kaggle kernels used (all in `hwm-experiment/experiments/`):
`kaggle_render_shards` (dataset rendering), `kaggle_package_r50` +
`kaggle_compute_changepoints` (dataset packaging), `kaggle_adaptive_train`
(Stage 1), `kaggle_finetune_compare` (Stage 2 pilot, n=4 -- superseded),
`kaggle_finetune_full_eval` (Stage 2 real run, n=40, this document's numbers).

## 4. On the n=4 pilot (why it's not reported above)

An earlier n=4 run (8 envs, only 4 fell into a reportable turn-count bucket)
gave baseline 75% / adaptive 100% on medium only. That's the same direction as
the n=40 result above but far too small a sample to treat as evidence on its
own -- it's reported here only as the reason the n=40 re-run happened, not as
a second data point.
