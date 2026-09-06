# Adaptive (changepoint-based) waypoints

Branch: `adaptive-waypoints` (fork: slcnvly/HWM_PLDM), based on clean
`upstream/main` — independent of the `lang-align` branch/hypothesis.

## Hypothesis

Baseline HWM uses a fixed waypoint stride (`l2_step_skip=10`) and fixed
waypoint count (`l2_n_steps=6`), uniformly spaced. A fixed schedule like this
is tuned (implicitly or explicitly) to Diverse Maze's specific dynamics and
episode length, and won't generalize across tasks with different
characteristic timescales (e.g. denser vs sparser direction changes). The
model already computes one-step prediction error every step (that's the
`PredictionObs`/`PredictionProprio` training signal) — reusing *that same
signal* to place waypoints at high-error points (where the frozen level-1
world model's simple local dynamics assumption breaks down — hitting a wall,
a sharp turn) should track the trajectory's actual structure instead of an
arbitrary fixed clock, and should generalize better across layouts/tasks
without per-task retuning of stride/count.

## Design decisions (my calls, since I'm proceeding without check-ins)

**1. Keep waypoint count (6) and total window (60 raw steps) fixed, only
change *placement* of the 5 internal boundaries.** This isolates the effect
being tested (adaptive placement vs. uniform spacing) from confounds like
"more/fewer waypoints" or "longer/shorter horizon", and keeps the comparison
against baseline apples-to-apples (same model capacity, same effective
receptive field). A version that also varies count/horizon is a reasonable
follow-up but conflates two hypotheses in one experiment.

**2. Keep the action-encoder architecture completely unchanged.** Baseline's
posterior MLP takes a flattened fixed-size `(10, 2)` action chunk. Rather
than redesigning it for variable-length input (RNN, padding+masks — more
moving parts, more that can go subtly wrong, harder to isolate from the
placement hypothesis itself), each variable-length segment's raw actions are
**linearly interpolated (resampled) to exactly 10 steps** before hitting the
existing encoder. This keeps the diff to data construction only — zero model
architecture changes, matching this project's running principle (touch as
little of the frozen/shared machinery as possible; see [[lang-align]]'s same
principle applied to level1).

**3. Changepoint signal: frozen level1's one-step prediction error, computed
in a one-time offline preprocessing pass**, not recomputed live during
training. Live computation would need a level1 forward pass inside
`__getitem__` for every sample every epoch — wasteful, since the frozen
model and raw trajectories never change. Preprocessing produces cached
boundary indices per trajectory, stored alongside the existing `data.p`.

**4. Segmentation algorithm: greedy peak-picking with a minimum-spacing
constraint.** Given a per-step error series over a 60-step window, greedily
take the highest-error step, remove all steps within `min_seg` of it, repeat
until 5 boundaries are chosen (or fall back to uniform spacing for slots
that can't be filled without violating min_seg — happens only for
pathologically flat error series). `min_seg` is the one real hyperparameter
here and is exactly what "제일 좋은 기준점을 실험을 통해 잡는다" means in
practice — swept empirically (see EXPERIMENTS.md once results exist)
rather than guessed.

## What's genuinely being tested vs. what's a simplification

This tests "does non-uniform boundary placement (holding count/horizon/
architecture fixed) beat uniform placement". It does *not* test variable
waypoint *count* per trajectory, which is closer to the strongest version of
the user's stated claim ("fixed count/spacing only suits specific tasks").
Recommending this narrower version first because it's the smallest change
that still directly tests the core mechanism (error-driven placement) without
also changing model capacity — if it shows a real effect, variable count is
the natural next step; if it doesn't, variable count wouldn't be worth
trying either (the placement signal itself would be the confirmed dead end,
not just this particular encoding of it).
