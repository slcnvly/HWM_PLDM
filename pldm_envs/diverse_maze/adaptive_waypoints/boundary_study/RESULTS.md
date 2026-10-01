# Boundary-signal study: results

Branch `adaptive-waypoints`. Full process/derivation trail in
`PROGRESS.md`; this file is the condensed, decision-relevant summary,
judged against `PREREGISTRATION.md`'s stop condition (§8) and its
Amendments 1-3.

## tl;dr

1. **No post-hoc one-step-prediction-error signal is usable on this
   checkpoint.** The frozen level1 model was trained with pure rollout
   loss (no teacher forcing -- `z_dim=0` makes the mechanism
   architecturally absent, not just unused) over a 15-frame window. Its
   fresh one-step prediction from a real re-anchored state is ~95% a
   fixed, input-independent bias with ~zero correlation to real movement
   direction. This rules out signals 1b/2b/3b/4/5/9b as currently defined.
2. **Every scalar signal we tried (6, 7, 8, 10, 11, 12, 13, 16, 16v2)
   loses to fixed-interval placement on Metric B**, at every min_seg
   value tested (relaxing min_seg makes scalar signals WORSE, not
   better).
3. **A trajectory-level dynamic-programming split (same information every
   other method has, just exactly optimized for Metric B) beats
   fixed-interval placement by 30.7% even under the ORIGINAL min_seg=8
   constraint.** A much simpler, fast, deterministic bottom-up merge
   algorithm captures most of that gain (+9.6% at min_seg=8, up to
   +27.9% at min_seg=3) -- after fixing a real implementation bug that
   initially hid this (see below).
4. **min_seg is not justified by compression loss** -- short segments
   compress/decompress through the fixed-10-step scheme with near-zero
   loss; long segments are where the real cost is, and min_seg only
   bounds the minimum, never the maximum.
5. **Metric A and Metric B substantially disagree** (Spearman rho -- see
   below) -- "physically/humanly meaningful moment" and "good anchor for
   piecewise-linear latent reconstruction" are measuring different
   things, not proxies for each other.
6. **Open question, explicitly not answered here**: does the Metric B
   gain from DP/bottom-up segmentation translate into better *planning
   success*? That requires fine-tuning + eval rollouts, out of scope for
   this CPU-only study (per the original instruction and the most recent
   one) -- a `dp_segmentation` boundary cache for the full r50 dataset has
   been generated for exactly this follow-up (see below).

## 0. Scope and what this document does not cover

Per repeated instruction across this study: no `pldm.train` fine-tuning,
no `pldm.planning`/MPC eval rollouts, no GPU. Everything here is frozen-
level1 forward passes (or, for Metric B/dp_segmentation, not even that --
pure CPU array math on already-computed real latent trajectories) plus
CPU-side statistics. The one exception, by design: a small post-hoc
variance-head MLP (§8-style, CPU, seconds) -- not the frozen world model.

## 1. Stage 0-1: infrastructure, grid-conversion bug, event labeling

- Full r50 dataset (main=1250 eps/25 maps, probe=1000 eps/20 *different*
  maps) and the frozen L1-only checkpoint were already available locally
  (inside a prior Kaggle kernel's cached output) -- this entire study ran
  on local CPU, no Kaggle kernel needed.
- Found and fixed a real wall-violation bug in the maze coordinate
  conversion (`maze_stats.py`'s `obs_range_total` for `maze2d_large_diverse`
  used an inconsistent divisor, 10.2 vs the `n=10` actually used at lookup
  time) -- a scale error, not an offset, confirmed by a direction-skew
  signature (100% of violations on the `+i`/`+j` edge) and a +/-0.5-cell
  shift making it worse. Fixed locally (`grid_utils.py`), not patched into
  the shared file. Verified 315->0 violations at full scale.
- Checked the bug's actual blast radius: S5-S8's real hard-difficulty
  instances bypassed the buggy live code path entirely (loaded from a
  pre-saved file) -- 0/80 real points were actually in a wall, though 2/80
  would have been wrongly flagged as walls by the buggy conversion if
  anything downstream had used it.
- Event labels (wall contact, direction turn, corridor change, junction
  arrival) computed for all 2250 episodes -- all four rates plausible and
  non-degenerate (2.4%-18.7% of steps positive).
- The data split originally planned (halve one shared 25-map pool) turned
  out to not apply -- main and probe are ALREADY disjoint map pools (25 vs
  20 completely different maps, verified by layout string comparison, not
  just index). Corrected design: main = selection set, probe = report set
  (Metric A/B scoring in this document all uses the selection set, main;
  Metric A has not yet been run on the report set, probe -- noted as
  future work, not done here).

## 2. Amendment 1: signal 1 (the original pipeline's signal) is confounded with window position

Spearman(signal 1, window index t) = 0.573 (N=2250 episodes x 60 steps,
p~0) -- strong, real structural dependence on elapsed window position, not
purely local dynamics. Signal 1b (re-anchored): rho=-0.029, negligible.
Diagnosed via a rollout-step-size check: NOT a magnitude-collapse ("the
rollout stops") -- the very first predicted step overshoots real movement
by 11x, then stabilizes near the real step size for the rest of the
window. Pearson(signal 1, distance-from-window-start) = 0.436 -- moderate,
real confound.

## 3. Amendment 2: the gate check -- predictor-based re-anchoring is ~95% bias

Before building any Stage 2 signal on a fresh single-step predictor call,
gate-checked whether "1b" (re-anchored one-step error) measures local
dynamics or something else. Full corpus (N=135,000 samples), three spaces
(fused "state", obs-only [matches the real training objective AND the
MPPI planner's cost], proprio-only):

| | fused | obs-only | proprio-only |
|---|---|---|---|
| ratio (pred-error / "nothing moved" baseline) | 8.83x | 20.37x | 1.01x |
| bias as % of prediction error | 94.0% | 94.7% | 31.7% |
| direction cosine (pred step vs. real step) | 0.020 | 0.0007 | 0.444 |

Redefining to match the planner's own cost space (obs-only) made it
WORSE, not better (20.4x vs 8.8x) -- ruled out "wrong space" as the
explanation.

## 4. The teacher-forcing correction (the real mechanism)

Initial hypothesis ("the model never sees a real state at t>0, only ever-
compounding self-generated states") was checked and REFUTED --
`d4rl.py:236-237` confirms training windows start at arbitrary in-episode
positions, so a real observation at any t is exactly what some training
window's own first frame looks like. **The user pushed back with the
correct mechanism**, verified directly against the paper's own
appendix-described loss weighting (`gamma_tf=0, gamma_roll=1` for the
maze sub-model):

- `large_diverse_25maps.yaml:64`: `z_dim: 0` for level1's ACTUAL
  pretraining config (`train_l1: true`, confirmed -- not the `_l2` variant
  this whole study loads checkpoints via). With `z_dim=0`,
  `sequence_predictor.py:44-45` never instantiates `prior_model`
  (`self.prior_model = None`, line 78) -- the ONLY teacher-forcing-capable
  branch in `forward_multiple` (the `term_states` path, ~line 260-268) is
  **architecturally unreachable, by explicit design**. `PredictorConfig.
  use_teacher_forcing`/`transformer_teacher_forcing_ratio`
  (`enums.py:37,39`) are dead config, consumed nowhere in this codebase.
  `PredictionObs`/`PredictionProprio` are both plain, pure-rollout
  `PredictionObjective` instances -- there is no teacher-forcing loss
  code path to weight against, for this architecture, at all.
- `prediction.py:83`: squared error (MSE), not L1 -- a paper-vs-code
  discrepancy if the appendix literally states L1 for this sub-model.
- `large_diverse_25maps.yaml:1,40`: training rollout length is **15
  frames (14 prediction steps)**, not 60 -- this entire study has
  evaluated the frozen model 4x past its trained horizon.

**Precise mechanism**: with no teacher-forcing path, no gradient signal
ever isolates and corrects a SINGLE fresh one-step prediction specifically
-- it's diluted, unweighted, across a 14-step rollout mean loss. Window
position was never the relevant variable (it does start anywhere); the
relevant fact is that "receiving a real state and being scored purely on
the immediate next step" never happens as its own isolated training
signal. Signal 16 was redefined to match the model's real 15-frame
rollout horizon (14-step, not 1-step) -- this measurably helped (obs:
-19.9% -> -13.2% vs. fixed; proprio: -16.0% -> -7.8%) but didn't flip the
sign; still a net loss.

**Part A, resolved in passing**: the action-connectivity paradox (1-step
real/shuffled-action ratio = 1.0000, yet the planner gets ~80-92% success
on hard difficulty) is not a contradiction. Two genuinely different real
actions on the same `z_t` DO change the obs-only prediction by a real
amount (L2=6.53) -- just ~2.5% of the ~256 L2 scale the common bias
dominates. MPPI only needs *relative* ranking among candidates sharing the
same start state/target, where the common bias cancels -- the planner can
exploit exactly the part an absolute-error-ratio metric can't see.

## 5. Stage 2 + Amendment 3: every scalar signal loses to fixed-interval, at every min_seg

300-episode sample (main, stratified across all 25 maps), Metric B
(piecewise-linear latent reconstruction error) at min_seg=8:

| | mean Metric B | vs. fixed |
|---|---|---|
| **dp_segmentation** (exact DP optimum for Metric B -- see §6) | 0.0581 | **+31.3%** |
| bottom_up (merge algorithm, min_seg=8, AFTER bugfix -- §6) | ~0.072 | **~+9-12%** |
| top_down (min_seg=8) | 0.0880 | -4.0% |
| **fixed (10,20,30,40,50)** | **0.0846** | 0% (reference) |
| random (20 seeds) | 0.0853 | -0.7% |
| signal_13 (local chord deviation) | 0.0897 | -6.0% |
| signal_16v2 proprio (14-step rollout-matched) | 0.0827 | -7.8% |
| signal_16v2 obs (14-step rollout-matched) | 0.0868 | -13.2% |
| signal_10 (action delta) | 0.0966 | -14.2% |
| signal_6 (BOCPD, 10-component PCA) | 0.0983 | -16.1% |
| signal_8 (direction change) | 0.1044 | -23.3% |
| signal_7 (latent speed) | 0.1073 | -26.8% |
| signal_12 (normalized curvature) | 0.1118 | -32.1% |
| signal_11 (2nd-difference curvature) | 0.1174 | -38.7% |

**Every single scalar signal loses.** The motivating hypothesis for 11-13
(Metric B rewards curvature/bends, not speed) was checked directly and
was WRONG -- all three curvature signals lose too, and signal 11 is the
single worst signal in this entire study.

## 6. Correcting the "bottleneck is min_seg" conclusion -- it was a real implementation bug

An earlier pass through this study concluded "the bottleneck is min_seg=8,
not signal quality," based on bottom-up merging jumping from -2.1% (not
beating fixed) at min_seg=2 to +24.7% at min_seg=1. **That comparison, and
that conclusion, were both wrong, for two separate reasons the user caught
directly:**

**(a) Mismatched comparison.** "Bottom-up reaches 86% of oracle" compared
*unconstrained* bottom-up (+26.9%, no min_seg at all) against *min_seg=8-
constrained* oracle (+31.3%) -- different constraints, not a fair ratio.
The correct same-min_seg reachability (bottom_up improvement / oracle
improvement, both at the identical min_seg):

| min_seg | oracle | bottom_up (fixed) | reachability |
|---|---|---|---|
| 8 | 30.7% | 9.6% | 31.3% |
| 6 | 38.6% | 21.2% | 54.9% |
| 5 | 40.2% | 22.3% | 55.5% |
| 4 | 41.4% | 26.0% | 62.8% |
| 3 | 42.4% | 27.9% | **65.8%** |
| 2 | 42.7% | 26.9% | 63.0% |
| 1 | 43.9% | 24.7% | 56.3% |

**(b) A genuine implementation bug in `bottom_up_merge`** (`metric_b.py`),
found and fixed 2026-09-30: the original version always started merging
from EVERY interior point (ignoring `min_seg` entirely during the whole
merge-down phase), reached exactly 5 boundaries via pure unconstrained
cost-minimization, and only THEN patched `min_seg` violations with a
crude post-hoc force-merge-and-resplit heuristic. Since merging only ever
GROWS segments, that first phase produced the IDENTICAL unconstrained-
optimal 5 boundaries regardless of `min_seg`'s value -- the only thing
`min_seg` changed was how much low-quality post-hoc patching got layered
on top. `min_seg=1` (trivially no violations, no patching) happened to be
the only case showing the TRUE result; every other `min_seg` value was
degraded by an inconsistent amount of ad-hoc repair, producing the
artificial "cliff."

**Fixed** by building `min_seg` into the merge from the start: initialize
with `min_seg`-sized atomic segments (not length-1 segments), so every
segment stays >= `min_seg` throughout the ENTIRE merge process by
construction -- no post-hoc repair needed. (A second edge-case bug caught
while testing the fix: `range(min_seg, T, min_seg)` can leave a too-short
final segment when `T` isn't a multiple of `min_seg`; fixed to
`range(min_seg, T-min_seg+1, min_seg)`.)

**Corrected sweep, same 100 episodes, bootstrap-confirmed (paired, 95% CI,
n=10000):**

| min_seg | oracle | bottom_up | top_down | bottom_up vs fixed, 95% CI |
|---|---|---|---|---|
| 8 | 30.7% | **+9.6%** | -2.6% | [-0.01132, -0.00405] (significant, better) |
| 6 | 38.6% | +21.2% | 2.2% | [-0.02048, -0.01323] (significant, better) |
| 5 | 40.2% | +22.3% | 3.6% | [-0.02185, -0.01381] (significant, better) |
| 4 | 41.4% | +26.0% | 6.0% | [-0.02454, -0.01684] (significant, better) |
| 3 | 42.4% | **+27.9%** | 6.8% | [-0.02633, -0.01822] (significant, better) |
| 2 | 42.7% | +26.9% | 7.7% | [-0.02587, -0.01713] (significant, better) |
| 1 | 43.9% | +24.7% | 8.8% | [-0.02416, -0.01532] (significant, better) |

**Bottom-up significantly beats fixed-interval at EVERY min_seg tested,
including the original min_seg=8 (+9.6%, 95% CI excludes 0).** The
original "bottleneck is min_seg" framing is itself corrected: **the real
finding is "signal quality is uniformly bad across every scalar signal
tried, AND, independently, min_seg does still cap how much of the
available headroom a well-implemented direct-optimization algorithm can
reach"** -- both true, neither explains the other away. Scalar signals
(13/10/6) get monotonically WORSE as min_seg relaxes (opposite direction
from oracle/bottom-up/top-down) -- naive peak-picking has no mechanism to
avoid pathological clustering once spacing is unconstrained; only
algorithms that directly optimize Metric B benefit from relaxing it.

## 7. min_seg is not justified by compression loss

Round-trip MSE from the actual 10-step linear-interpolation compression
scheme (`resample_to_fixed_length`, the real mechanism DESIGN.md decision
2 uses for the action encoder), as a function of segment length (100
episodes, independent of any particular segmentation): grows smoothly and
consistently FASTER than linearly, from ~0.00009 at length 2 (near-zero --
resampling a short segment to 10 points is upsampling, not lossy
compression) to ~0.039 at length 58 (~440x higher for a 29x longer
segment). **This argues against `min_seg` being justified by compression-
loss concerns at all -- if anything, long segments (which `min_seg`
doesn't bound) are the expensive ones, not short ones.**

Applying this lookup to bottom-up's own actual chosen segment-length
distribution at each min_seg value (rerun with the fixed `bottom_up_merge`):

| min_seg | length: min/p10/median/p90/max | mean expected encoding loss |
|---|---|---|
| 8 | 8/8/8/16/20 | 0.00297 |
| 6 | 6/6/12/18/24 | 0.00312 |
| 5 | 5/5/10/15/25 | 0.00318 |
| 4 | 4/4/8/16/24 | 0.00320 |
| 3 | 3/3/9/15/27 | 0.00323 |
| 2 | 2/2/10/16/26 | 0.00331 |
| 1 | 1/2/10/17/27 | 0.00338 |

Same conclusion as before, now on the CORRECTED (bug-fixed) bottom-up
boundaries: encoding loss barely moves (0.00297 -> 0.00338, +14%
relative) across the whole sweep, while bottom-up's own Metric B
improvement moves from +9.6% to +27.9% (peak at min_seg=3) over the same
range -- the Metric B upside dominates the encoding-loss cost at every
point tested.

## 8. Metric A vs Metric B

[PENDING -- run_metric_a.py rerun with the fixed bottom_up_merge still
running; this section will report: step-level AUROC/AP for the 9 scalar
signals (+16v2 obs/proprio), boundary-level precision/recall/F1 overall
and per event type for all 13 candidates (renaming "oracle" to
"dp_segmentation" throughout per the user's framing -- it is not cheating,
it uses the same real latent trajectory every other method sees, just
finds the Metric-B-exact optimum via DP instead of a heuristic), bootstrap
95% CIs, and the Metric A-vs-B rank comparison with Spearman correlation.]

## 9. dp_segmentation cache for the full r50 dataset

Per the explicit instruction to prepare this (and nothing requiring GPU)
for a follow-up fine-tuning/eval run on Kaggle: generated a
`changepoints_minseg8.pt` cache for ALL of r50 (main: 1250 episodes,
probe: 1000 episodes) using exact DP optimization of Metric B (the
`dp_segmentation` method -- identical format to `compute_changepoints.py`'s
own output, directly loadable by `AdaptiveD4RLDataset` unchanged).

[PENDING -- `generate_dp_segmentation_cache.py` still running at time of
writing, ETA ~5 hours from launch (oracle DP's per-episode cost, ~8.4s/ep x
2250 episodes, is the dominant cost -- a known, pure-CPU, no-GPU-needed
batch job). Will fill in final wall-clock time and confirm both output
file paths once complete:
`pldm_envs/diverse_maze/datasets/r50_local/r50_dataset/main/
changepoints_minseg8.pt` and the `probe/` equivalent.]

## 10. Per the stop condition (PREREGISTRATION.md §8)

The stop condition asked: does any signal beat fixed-interval on both
metrics? Answer for Metric B, scalar signals: **no, not one, at any
min_seg value tested.** But the stop condition's framing (implicitly,
"is adaptive placement worth anything on this data") now needs a more
precise answer than a flat no: **adaptive placement IS worth something --
a direct Metric-B-optimizing split (dp_segmentation, or its fast
approximation bottom-up merging) beats fixed by 30.7% / ~10-28% even
under the original constraint. What doesn't work is finding that split
via any of the 9 scalar signals tried.** This is a meaningfully different
conclusion from "adaptive segmentation doesn't help" -- it's "the specific
signal-based approach to adaptive segmentation tried in signals 1-13/16
doesn't help; a direct optimization approach does, on this metric."

**What remains open, not answered by anything in this CPU-only study**:
does the Metric B gain from dp_segmentation/bottom-up actually improve
planning success rate, or steps-to-goal, once used to fine-tune and
evaluate a real checkpoint? Metric B is itself a proxy (how well can a
piecewise-linear fit reconstruct the latent trajectory) -- a real test
needs the GPU fine-tuning + MPC eval pipeline this study was explicitly
scoped to avoid. The `dp_segmentation` cache (§9) is prepared specifically
to make that follow-up possible without re-deriving anything.
