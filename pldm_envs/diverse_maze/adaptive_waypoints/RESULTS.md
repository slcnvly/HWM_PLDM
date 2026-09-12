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

**Two important qualifications, added after a follow-up control experiment and
significance analysis (§5-6) -- read before citing the "+10 points" figure
above:**
1. **About half of the hard-difficulty gain is just from fine-tuning, not
   from adaptive placement.** A control that fine-tunes on *fixed*-stride
   data, identically otherwise, lands at 87.5% (35/40) -- almost exactly
   halfway between baseline (82.5%) and adaptive (92.5%). Adaptive placement's
   own marginal contribution on top of fine-tuning is closer to +5 points,
   not +10.
2. **None of the pairwise differences (baseline vs. control, control vs.
   adaptive, or baseline vs. adaptive) are statistically significant at
   n=40** (95% CI always includes 0; permutation p in 0.31-0.76). The
   monotonic staircase across all three conditions is a real, promising
   pattern, but not yet a settled result at this sample size.

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
- **`Evaluator._create_l2_planning_evaluator` (pldm/evaluation/evaluator.py)
  dropped `level=level` from its `dataclasses.replace(self.h_planning_config,
  ...)` call** (its L1 sibling, `_create_l1_planning_evaluator`, already
  included it correctly) -- found while building the n=120 re-eval (SS6b)
  when `hard.n_envs` was set to something other than 40 for the first time.
  `MazeMPCEvaluator.__init__` does `level_cfg = getattr(config, config.level)`
  to pick the per-difficulty settings; with `config.level` stuck at its class
  default (`"medium"`), every *hierarchical (L2)* planning eval in this
  project -- regardless of which difficulty was requested -- was silently
  reading `n_envs`/`min_block_radius`/`max_block_radius` from **medium's**
  settings instead of the requested level's. Confirmed harmless for every
  number already in this document: this yaml sets identical
  `min_block_radius=4, max_block_radius=9999` across easy/medium/hard, and
  every prior hard-difficulty run in this project happened to set
  `hard.n_envs=40` -- identical to medium's default -- so the misdirected
  read produced the same result it would have anyway. `set_start_target_path`
  (the actual trial file) and `n_steps`/MPPI sub-configs are separate
  top-level `mpc_config` fields set correctly by the same call, so those were
  never affected. Fixed by adding `level=level` to match the L1 version.

Kaggle kernels used (all in `hwm-experiment/experiments/`):
`kaggle_render_shards` (dataset rendering), `kaggle_package_r50` +
`kaggle_compute_changepoints` (dataset packaging), `kaggle_adaptive_train`
(Stage 1), `kaggle_finetune_compare` (Stage 2 pilot, n=4 -- superseded),
`kaggle_finetune_full_eval` (Stage 2 real run, n=40, this document's numbers),
`kaggle_finetune_fixed_control` (§5 control, fixed-stride 2-epoch fine-tune,
hard-only n=40).

## 4. On the n=4 pilot (why it's not reported above)

An earlier n=4 run (8 envs, only 4 fell into a reportable turn-count bucket)
gave baseline 75% / adaptive 100% on medium only. That's the same direction as
the n=40 result above but far too small a sample to treat as evidence on its
own -- it's reported here only as the reason the n=40 re-run happened, not as
a second data point.

## 5. Fine-tuning control (isolating placement from "more training")

Section 1's adaptive result (33/40 -> 37/40 on hard) confounds two things:
adaptive waypoint *placement*, and simply receiving 2 epochs of gradient
updates that the untrained baseline never got. A third condition isolates
this: fine-tuned from the identical pretrained checkpoint, on the identical
r50 dataset, for the identical 2 epochs at the identical reduced LR
(`base_lr` x0.1) as the adaptive run -- the only change is the waypoint
dataset itself, plain `D4RLDataset` (fixed `l2_step_skip=10`,
`l2_n_steps=6`) instead of `AdaptiveD4RLDataset(min_seg=8)`. Evaluated on
hard difficulty only (medium already ties baseline/adaptive, so it isn't
informative for this question), same `n_envs=40, n_steps=500,
level2.mppi.num_samples=200` as the other two hard numbers.

| Condition | Hard success | Avg steps to goal |
|---|---|---|
| Baseline (pretrained, no fine-tune) | 82.5% (33/40) | 161.2 |
| **Fixed-stride, 2-epoch fine-tune (control)** | **87.5%** (35/40) | **151.1** |
| Adaptive min_seg=8, 2-epoch fine-tune | 92.5% (37/40) | 145.9 |

The control lands almost exactly halfway between baseline and adaptive on both
metrics (success rate: 82.5 &rarr; 87.5 &rarr; 92.5, i.e. +5pp then +5pp again;
steps: 161.2 &rarr; 151.1 &rarr; 145.9). Point estimates split the 10-point
baseline-to-adaptive gain roughly evenly:
- **~half (5 of 10 points) is attributable to fine-tuning itself** -- 2 epochs
  on r50 helps even without changing waypoint placement, which isn't
  surprising (the pretrained checkpoint has never seen this specific
  main-dataset subset).
- **~the other half (5 of 10 points) is attributable to adaptive placement
  specifically** -- switching only the waypoint dataset, with identical
  compute/LR/schedule, adds a further +5pp and ~5 fewer average steps on top
  of what fine-tuning alone gets.

This is a real finding in its own right and a substantially more honest claim
than Section 1's original "adaptive beats baseline by 10 points" -- about
half of that gap would have shown up from fine-tuning on *any* waypoint
schedule, fixed or adaptive. See Section 6 for whether any of these three
point estimates are actually distinguishable from each other at n=40.

## 6. Statistical significance (hard difficulty, n=40 per condition)

Two independent checks per condition: a Wilson score interval (better-behaved
than the normal approximation at n=40) and a percentile bootstrap (10,000
resamples) on the binomial success rate. Pairwise comparisons use a Newcombe
interval on the difference in proportions plus a two-sided permutation test
(10,000 permutations) on the pooled pass/fail counts. Script:
`confidence_intervals.py` in this directory; raw output:
`confidence_intervals_results.json`.

| Condition | k/n | Rate | Wilson 95% CI | Bootstrap 95% CI |
|---|---|---|---|---|
| Baseline | 33/40 | 0.825 | [0.681, 0.913] | [0.700, 0.925] |
| Fixed-stride control | 35/40 | 0.875 | [0.739, 0.945] | [0.775, 0.975] |
| Adaptive min_seg=8 | 37/40 | 0.925 | [0.801, 0.974] | [0.825, 1.000] |

| Pair | Diff | Newcombe 95% CI | Permutation p (2-sided) | Significant @ .05 |
|---|---|---|---|---|
| Baseline vs. adaptive min_seg=8 | -0.100 | [-0.253, +0.051] | 0.308 | **No** |
| Baseline vs. fixed-stride control | -0.050 | [-0.211, +0.112] | 0.755 | **No** |
| Adaptive min_seg=8 vs. fixed-stride control | +0.050 | [-0.092, +0.195] | 0.721 | **No** |

**None of the three pairwise gaps reach significance at n=40** -- every 95%
CI on a difference includes 0, and every permutation p-value is well above
.05 (0.31-0.76). This holds even though the point estimates move in a clean,
monotonic staircase (baseline < control < adaptive, both on success rate and
steps-to-goal, Section 5) -- a pattern that's suggestive but, at this sample
size, not statistically distinguishable from three draws off the same
underlying rate. Concretely: the 95% CI on baseline vs. adaptive alone spans
roughly [-25pp, +5pp], wide enough to be consistent with adaptive being
worse, the same, or better than baseline.

A meaningfully tighter CI on the ~5pp step-wise effects above would need on
the order of 150-200+ trials per condition at this effect size and base
rate (rough two-proportion power calculation, 80% power, alpha=.05
two-sided) -- well beyond a single free-tier Kaggle GPU session's budget for
a 500-step, 40-env MPPI eval (the hard-only eval above alone took ~1h11m of
planning time on top of ~2h of fine-tuning). **Bottom line at n=40: the
staircase pattern (fine-tuning helps; adaptive placement adds a further,
similarly-sized increment on top) is the best point estimate available, but
should be read as a promising, non-definitive signal pending a larger run,
not a settled result.** Section 6b below is that larger run.

### 6b. Scaling to n=120 (hard difficulty)

The existing 40 trials per condition (`starts_targets_13_16.pt`, seed=42) were
kept as-is and combined with 80 new trials per condition (same map pool,
disjoint start/target instances, seed=20260910), evaluated in three
checkpointed/resumable chunks (30/30/20) using the identical three already-
saved checkpoints from Section 5 -- no re-training, evaluation only, same
`n_steps=500, level2.mppi.num_samples=200`. Chunked new-80 results (chunks of
30/30/20): baseline 22+24+17=63/80, fixed-stride control 28+27+20=75/80,
adaptive min_seg=8 29+28+17=74/80 (full breakdown in
`results_hard_n120_progress.json`/`results_hard_n120_final.json`); combined
n=120 totals below.

| Condition | n=40 rate | n=40 avg steps | n=120 rate | n=120 avg steps |
|---|---|---|---|---|
| Baseline | 82.5% (33/40) | 161.2 | **80.0%** (96/120) | 169.6 |
| Fixed-stride control | 87.5% (35/40) | 151.1 | **91.7%** (110/120) | 169.5 |
| Adaptive min_seg=8 | 92.5% (37/40) | 145.9 | **92.5%** (111/120) | 155.1 |

The n=40 "staircase" (baseline < control < adaptive, evenly spaced) does
**not** hold up at n=120. Two things move:
- **Fixed-stride control jumps from 87.5% to 91.7%** and lands within 1
  success of adaptive (110/120 vs 111/120) -- at n=40 the control's extra 8
  trials happened to undershoot fine-tuning's true effect. With more data,
  fine-tuning alone (no adaptive placement) accounts for nearly all of the
  gain over baseline.
- **Baseline itself drops slightly, from 82.5% to 80.0%**, widening the
  fine-tuning-vs-no-fine-tuning gap rather than narrowing it.
- **Adaptive's success rate is flat (92.5% at both scales)**, but its avg
  steps-to-goal advantage persists and sharpens relative to the control:
  155.1 vs 169.5, whereas at n=40 the gap was smaller (145.9 vs 151.1) and
  baseline's n=120 avg steps (169.6) is now essentially tied with the
  control's (169.5), not worse.

| Pair (n=120) | Diff | Newcombe 95% CI | Permutation p (2-sided) | Significant @ .05 |
|---|---|---|---|---|
| Baseline vs. adaptive min_seg=8 | -0.125 | [-0.213, -0.038] | 0.0083 | **Yes** |
| Baseline vs. fixed-stride control | -0.117 | [-0.205, -0.028] | 0.0149 | **Yes** |
| Adaptive min_seg=8 vs. fixed-stride control | +0.008 | [-0.064, +0.081] | 1.000 | **No** |

At n=120, both fine-tuned conditions are now significantly better than
baseline on success rate (this wasn't true at n=40). But **adaptive vs. the
fixed-stride control is not distinguishable on success rate** (diff
+0.8pp, p=1.0) -- the n=40 result's "~half the gain is adaptive-placement-
specific" claim does not survive the larger sample. What does survive:
adaptive's steps-to-goal advantage over the control (155.1 vs 169.5, ~8%
fewer steps among successful trials) persists at both scales, though no
significance test is reported for this continuous metric here (only success
rate was tested; a proper comparison would need the per-trial steps-to-goal
distribution, not just the aggregate mean, which isn't retained from this
run's summary.json outputs).

**Revised bottom line: fine-tuning on r50 -- regardless of whether waypoints
are placed adaptively or at fixed stride -- is the effect that reliably beats
baseline at this sample size. Adaptive placement's distinct contribution, if
any, shows up in steps-to-goal efficiency among successful trials, not in
whether the agent succeeds at all.** This is a materially different (and more
conservative) conclusion than Section 6's n=40 read, and supersedes it.

## 7. Error-adaptive L1 resource allocation (eval-only)

Sections 5-6b vary *what's in the level2 waypoint dataset* (fixed stride vs.
adaptive segmentation) but always give the level1 MPPI planner the same fixed
compute budget every real step. This section asks a different, orthogonal
question: at fixed model weights (no re-training, same two checkpoints as
SS6b -- baseline and adaptive min_seg=8), can *spending more level1 planning
compute only when the model is visibly surprised* improve on always spending
the same amount?

### 7.1 Experiment design

**Online surprise signal.** After every real MPC step, compute a one-step
latent prediction error: encode the actual post-step observation and compare
it to the one-step-ahead prediction, squared-error-averaged over the encoding
feature dims -- exactly `compute_changepoints.py`'s `compute_error_series`
formula (DESIGN.md SS3), just evaluated online instead of once offline over
stored trajectories. Two choices made here, checked with the user before
running (both approved) since they affect whether the online number and the
threshold below are even on the same scale:
- **Latent-space MSE, not raw xy/pixel distance.** The changepoint error
  distribution this threshold is drawn from is latent-space; a location-space
  error would be a different metric on a different scale.
- **Computed with the same separate frozen L1-only checkpoint**
  `compute_changepoints.py` used (`3-9-1-seed248_epoch=3_sample_step=15465472.ckpt`,
  from `kevinghst/pldm-maze2d-large-diverse`), not the L1 weights bundled
  inside whichever hierarchical checkpoint (baseline/adaptive) is being
  planned with -- those are two different pretrained models in this repo.
  This monitor model has nothing to do with the model actually doing the
  planning; it only produces the surprise signal.

**Threshold.** Mean + 2*std of the identical error computed over every
step of every episode in the r50 **main** dataset (1250 episodes x 60 steps
= 75,000 steps -- the same data/model `compute_changepoints.py` swept for
changepoints, just aggregated into a scalar mean/std here instead of used for
peak-picking): mean=4.6338, std=1.0756, **threshold=6.7851**. Computed once
per kernel run (~1 minute), before any of the 12 eval chunks.

**Resource boost.** Whenever a real step's error exceeds the threshold, the
*next* level1 planning call uses 2x `num_samples` and 1.5x planning horizon;
otherwise the checkpoint's normal fixed settings (hard: `num_samples=1000`,
horizon=`l2_step_skip=10`) are used. Implementation detail worth stating
plainly: `num_samples` boosts **per-env** (MPPI already runs its per-sample
rollout in a per-env Python loop, so `ctrl.K` is trivially settable per env),
but planning **horizon** boosts at the **whole-chunk** level (any env in the
20-env chunk tripping the threshold boosts the horizon for the entire batch
that step) -- `MPPIPlanner.plan()`'s batched dynamics rollout requires one
shared plan length per call, so per-env horizon isn't available without
restructuring the batched planner itself, which was out of scope for an
eval-only change. Level2 (waypoint planning) is completely untouched --
`num_samples=200`, no horizon change -- matching the user's request.

Implementation lives entirely in new eval-side files
(`error_adaptive_l1.py`'s `compute_r50_error_threshold`, `L1ErrorMonitor`,
`ErrorAdaptiveL1Planner`) plus one small additive hook on the existing
`MPCEvaluator` (`post_l1_step_hook`, default `None`, called once per real
step in the bilevel branch of `pldm/planning/mpc.py`) and a new
`ErrorAdaptiveHierarchicalD4RLMPCEvaluator` subclass
(`pldm/planning/d4rl/hmpc.py`) wired in behind an opt-in config flag
(`eval_cfg.h_d4rl_planning.error_adaptive_l1`, default `false`). No training
code (`pldm/train.py`, objectives, model definitions) was touched.

**Trials.** Same 120 hard-difficulty (D 13-16) trials as SS6b: the original
40 (`starts_targets_13_16.pt`, seed=42) + the same 80 new ones (seed=20260910)
-- reused directly from SS6b's own saved `extra80_full.pt` rather than
regenerated, so the trial set is byte-identical, not just seed-identical.
Evaluated fresh in full (not just the new 80) for the two new conditions,
since fixed-vs-error-adaptive is a new axis that SS6b never ran. Same
`n_steps=500`, same `level2.mppi.num_samples=200`. Chunked 6x20 (vs. SS6b's
30/30/20) with progress written to
`results_hard_l1adaptive_progress.json` after every chunk, same
resumable-skip-already-done-chunks pattern as SS6b's kernel. (In practice
this run completed end-to-end in one ~9h Kaggle session, so resumption was
never exercised for real -- see SS7.3.)

### 7.2 Results

| Checkpoint | L1 resource allocation | Hard success (n=120) | Avg steps to goal |
|---|---|---|---|
| baseline | fixed (SS6b) | 80.0% (96/120) | 169.6 |
| baseline | error-adaptive (SS7) | 80.0% (96/120) | 163.9 |
| adaptive min_seg=8 | fixed (SS6b) | 92.5% (111/120) | 155.1 |
| adaptive min_seg=8 | error-adaptive (SS7) | **95.8%** (115/120) | **152.6** |

Error-adaptive allocation moves in the same direction for both checkpoints
(flat-or-better success rate, fewer average steps), but the effect is small
and, per SS7.3 below, not statistically distinguishable from zero at this
sample size:
- **Baseline: identical success rate (96/120 both ways), ~3.3% fewer average
  steps** (169.6 -> 163.9). Boosting compute on surprising steps didn't turn
  any failure into a success here, but successful trials finished slightly
  faster.
- **Adaptive min_seg=8: +4 successes (111 -> 115, 92.5% -> 95.8%), ~1.6%
  fewer average steps** (155.1 -> 152.6). The larger of the two effects, and
  the only one where the success count actually moved -- but see the
  significance test below before reading much into +4/120.

### 7.3 Statistical significance

Same three checks as SS6 (Wilson CI, Newcombe CI on the difference, two-sided
permutation test, 10,000 resamples/permutations each), extended to all 6
pairs among the resulting 4 conditions (`confidence_intervals.py`, scale
`n120_l1adaptive`). The two pairs this experiment is actually about --
same checkpoint, fixed vs. error-adaptive -- first:

| Pair | Diff | Newcombe 95% CI | Permutation p (2-sided) | Significant @ .05 |
|---|---|---|---|---|
| Baseline: fixed vs. error-adaptive | +0.000 | [-0.101, +0.101] | 1.0000 | **No** |
| Adaptive min_seg=8: fixed vs. error-adaptive | -0.033 | [-0.099, +0.030] | 0.4016 | **No** |

Neither same-checkpoint pair is significant -- baseline's is exactly zero
(identical 96/120 both ways) and adaptive min_seg=8's +4-success gain has a
95% CI that comfortably includes 0. The remaining 4 (cross-checkpoint) pairs,
for completeness -- these reproduce SS6b's already-established
baseline-vs-fine-tuned result and aren't new information:

| Pair | Diff | Newcombe 95% CI | Permutation p (2-sided) | Significant @ .05 |
|---|---|---|---|---|
| Baseline (fixed) vs. adaptive min_seg=8 (fixed) | -0.125 | [-0.213, -0.038] | 0.0073 | **Yes** |
| Baseline (fixed) vs. adaptive min_seg=8 (error-adaptive) | -0.158 | [-0.242, -0.077] | 0.0001 | **Yes** |
| Baseline (error-adaptive) vs. adaptive min_seg=8 (fixed) | -0.125 | [-0.213, -0.038] | 0.0079 | **Yes** |
| Baseline (error-adaptive) vs. adaptive min_seg=8 (error-adaptive) | -0.158 | [-0.242, -0.077] | 0.0005 | **Yes** |

**Bottom line: at n=120, error-adaptive L1 resource allocation is not
distinguishable from fixed allocation for either checkpoint on success rate**
-- both point estimates move in the helpful direction (flat-or-better,
never worse) and average steps-to-goal improves modestly for both, but
none of that clears the bar a Wilson/permutation test needs at this sample
size. This doesn't rule out a real effect (the adaptive-checkpoint pair's
point estimate, +3.3pp, is the same rough magnitude as SS6's n=40-scale
effects that later *did* firm up at n=120 in SS6b) -- it says a larger run
would be needed to tell a small real effect apart from noise here, same
caveat SS6 raised about its own n=40 numbers.

### 7.4 Infra notes and timing

Three bugs surfaced and were fixed before this run's numbers landed (kept
here rather than only in commit messages, per this project's practice of
documenting what actually went wrong):
- Two attempts (kernel versions 1-2) crashed during setup on issues specific
  to this kernel script, not the feature code: a nested-duplicate-path glob
  on the mounted L1-only checkpoint (same root cause as the
  main/probe-directory gotcha already in kaggle-infra-gotchas, just not
  applied consistently to every glob in this script), and then the
  checkpoint being deleted (as part of the usual extraction-dir cleanup)
  before threshold calibration got to load it.
- One more fix applied in review, before it could cause a third failure: the
  calibrated threshold is a small float that can render in Python's
  scientific notation, which risked being misparsed when interpolated into
  the OmegaConf CLI override string -- fixed by formatting it fixed-point.

Version 3 ran clean end-to-end. Threshold calibration: ~1 minute (75,000
steps over the small L1-only model). All 12 eval chunks (2 checkpoints x 6
chunks of 20): **36-40 minutes each**, ~7.4 hours total planning time, both
checkpoints landing within a 4-minute band of each other per chunk despite
one having a materially higher boost-trigger rate in principle -- the
per-step boost-decision overhead wasn't separately instrumented in this run,
so the actual boosted-step fraction for each condition isn't available to
report (a gap worth closing before running this again). Total session
length ~9 hours including environment setup (miniconda + mujoco-py +
dependencies, ~10-15 min) and the two failed/retried versions' own setup
time.
