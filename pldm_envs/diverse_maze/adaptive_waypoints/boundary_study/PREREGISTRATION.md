# Boundary-signal study: preregistration

Branch: `adaptive-waypoints`. Written and committed *before* any signal-scoring
code is run (Stage 0, per instructions). Builds directly on the existing
`compute_changepoints.py`/`segmentation.py` infra in this directory. **No
fine-tuning and no planning/eval rollouts are run anywhere in this study** --
only frozen level1 forward passes (`model.level1.forward_posterior`, already
used by `compute_changepoints.py`) and CPU-side array/statistics code.

## 0. Question

Given the existing pipeline (frozen level1, r50 dataset, 60-step window, 6
waypoints = 5 internal boundaries, greedy peak-picking with `min_seg=8`,
10-step linear interpolation for the action encoder -- see `DESIGN.md`),
**which of several training-free boundary-detection signals most clearly
identifies "good" waypoint boundaries**, scored two ways: agreement with
independently-labeled physical events, and how well the resulting boundaries
support piecewise-linear reconstruction of the latent trajectory.

## 1. Data and infra facts (confirmed before writing this doc)

- Dataset: `r50_dataset` (packaged output of `kaggle_package_r50`, already
  present locally as a ~9.9GB extract at
  `pldm_envs/diverse_maze/datasets/r50_local/r50_dataset/`). **main = 1250
  episodes, probe = 1000 episodes**, each with `images.npy` (98x98x3 uint8,
  101 raw frames/episode) + `data.p` (`observations` [x,y,vx,vy], `actions`,
  **`map_idx`** already present per-episode) + `train_maps.pt` (25 maps'
  layout strings for `main`; probe has its own copy).
- Frozen level1-only checkpoint: `3-9-1-seed248_epoch=3_sample_step=15465472.ckpt`
  (same one `compute_changepoints.py` and SS7/SS8 used), config
  `pldm/configs/diverse_maze/icml/large_diverse_25maps_l2.yaml`. Verified
  locally: `load_level1()` + `compute_error_series()` run correctly and
  reproduce a 60-length error series per episode in ~0.125s/episode on CPU
  (whole main+probe corpus: ~5 min). This entire study runs locally, no
  Kaggle kernel needed -- the full dataset and checkpoint were already
  sitting in this repo's `experiments/kaggle_package_r50/output/_output_.zip`.
- **Python version note:** this repo's dataclasses (`ProprioConfig` etc. in
  `pldm/models/encoders/enums.py`) use bare mutable dataclass defaults, which
  Python's `dataclasses` module rejects from ~3.11 onward (same issue
  `kaggle_compute_changepoints.py`'s docstring documents for Kaggle's system
  Python 3.12). This machine only has system Python 3.14. Rather than edit
  shared model code, installed a standalone Python 3.10 via `uv` into
  `~/.venvs/pldm_boundary` (torch 2.14 CPU build + omegaconf/numpy/scipy/
  sklearn/wandb) -- matches this project's existing precedent of solving this
  via environment control, not code changes. All `boundary_study/` scripts
  run under this venv's interpreter.
- **wandb: not yet logged in on this machine** (`wandb.Api()` has no API
  key configured). Runs will be logged in `WANDB_MODE=offline` and synced to
  project `hwm-boundary-study` once credentials are available -- flagged to
  the user, not blocking the rest of this study.

## 2. Wall-violation bug (Stage 1 prerequisite) -- root cause found and fixed

Prior session's grid-cell conversion (`obs_to_ij`-style: `floor((xy -
obs_min_total) / grid_size)`) had a 2.58% wall-violation rate (315/12200 on
a 200-episode probe sample), skewed **100% into the `+i`/`+j` (high) edge
direction, 0% into `-i`/`-j`** (chi2=311 vs uniform), and a +/-0.5-cell
constant shift made it *worse* in every direction tested, not better -- ruling
out a simple additive offset.

**Root cause** (found by reading `pldm_envs/diverse_maze/data_generation/
maze_stats.py`'s own comments for `maze2d_large_diverse`, not by curve-fitting):
`obs_range_total` (10.068675) was computed as `grid_size * 10.2` where
`grid_size = (obs_max_space - obs_min_space) / 8` (8 = empty/open blocks per
axis, 25-map large-diverse layouts have a 1-cell wall border on all sides of
a 10x10 grid). But `obs_to_ij`/`compute_changepoints.py` (utils.py) always
computes `grid_size = obs_range_total / n` with **n=10**, not 10.2. So the
origin (`obs_min_total`) was calibrated against one grid_size
(`obs_range_total/10.2 = 0.987125`) while every actual lookup uses a
*different*, ~2% larger grid_size (`10.068675/10 = 1.0068675`) -- a scale
mismatch, not a constant offset, which is exactly why it compounds toward the
high-index edge and why a constant +/-0.5-cell shift can't fix it.

**Fix, verified:** use a single self-consistent divisor (n=10) throughout:
`grid_size = (obs_max_space - obs_min_space) / 8`, `obs_range_total =
grid_size * 10` (9.87125, not 10.068675), `obs_min_total` unchanged
(-0.636125 -- it happens to already be consistent with the corrected
grid_size). Re-running the exact same 200-episode/12200-coordinate
validation with this pair: **open_fraction 0.97418 -> 1.00000, 315 -> 0
violations, in both directions.** This does not require knowing the maze's
true physical wall thickness/agent radius (unavailable locally -- no mujoco/
d4rl/gym package or XML is present in this environment) -- it's resolved
purely by making the codebase's own two divisors consistent. Implemented as
a small local override in `boundary_study/grid_utils.py` (does not touch the
shared `maze_stats.py`, to avoid any risk to prior stages' results, which
used `obs_to_ij` only for goal-validity/BFS checks at a resolution where this
2% drift apparently never mattered enough to break planning -- worth a note
for whoever next touches core grid-conversion code, but out of scope here).

## 3. Signals under comparison (10, all training-free, all from frozen level1)

For a 60-step window (raw error series indices 0..59, corresponding to
frames 1..60 of the 61-frame window):

1. **Raw one-step error**: `||ẑ_{t+1} - z_{t+1}||_2` -- current pipeline's
   existing signal (`compute_error_series`).
2. **Surprise** (§8 method): `err_t^2 / σ²(z_t)`, using a beta-NLL variance
   head retrained locally from scratch following §8.2's exact recipe (v2
   fix: log-variance output, standardized inputs/normalized target, beta=0.5,
   lr=1e-4, grad clip 1.0, best-val-NLL checkpoint) -- **the original
   trained weights were never persisted to disk** (only
   `variance_head_diagnostics_v2.json`'s summary numbers survived from the
   Kaggle run), so this is a faithful re-run, not a reuse of stale weights.
   Trained on main(train)/probe(val) errors+encodings, same split roles as
   §8.
3. **Locally-normalized error**: `err_t / mean(err_{t-10..t+10, excl. t})`,
   window truncated (not padded) at the 60-step series' own edges.
4. **5-step cumulative error**: roll frozen level1 forward 5 steps from
   `z_t` using the 5 real actions, compare to actual `z_{t+5}`. Defined for
   `t <= 54` (0-indexed); undefined tail (`t in [55,59]`) is set to NaN and
   excluded from peak-picking candidacy for this signal only (min_seg=8
   already excludes indices outside `[7,52]` from being *chosen* as a
   boundary for a 60-length series with 5 boundaries, so this NaN tail never
   actually removes a would-be-selected candidate in practice -- confirmed
   by construction, not assumed).
5. **Action sensitivity**: 16 action sequences = real action + iid Gaussian
   noise (std = 10% of each action dimension's empirical std, computed once
   over main-split actions), each rolled 1 step from `z_t`; signal = sum
   over feature dims of the variance (across the 16 resulting `ẑ_{t+1}`) --
   i.e. trace of the empirical covariance, not full covariance.
6. **BOCPD** (Adams & MacKay 2007), hazard=1/10, Gaussian observation model,
   applied to `z_t`. Running exact multivariate BOCPD on the full
   (spatial, ~33k-dim) latent is computationally impractical at this sample
   size (2250 episodes); **simplification, fixed here before any signal is
   scored, not tuned after seeing results:** PCA to the top 10 principal
   components (fit on main-split latents only), independent per-component
   Normal-unknown-mean/known-variance conjugate BOCPD, summed
   log-predictive-probability across components -> one scalar changepoint
   probability per step. Reported explicitly as an approximation in
   RESULTS.md, not silently as exact BOCPD.
7. **Latent speed**: `||z_{t+1} - z_t||_2`.
8. **Latent direction change**: `1 - cos(z_{t+1}-z_t, z_t-z_{t-1})`, undefined
   at `t=0` (no `z_{t-1}`) -- excluded from peak-picking candidacy at that
   single index (again outside `[7,52]`, so moot in practice).
9. **Error delta**: `err_t - err_{t-1}`, undefined at `t=0`, same as above.
10. **Action delta** (model-free baseline signal): `||a_t - a_{t-1}||_2`.

## 4. Event labels (ground truth for metric A) -- Stage 1

Computed independently of all 10 signals, from `(x,y,vx,vy)` + map layout
only, using the corrected grid conversion (§2). Concrete thresholds fixed
here (not tuned against any signal):

- **Wall contact**: current cell has >=1 open-neighbor that is a wall AND
  distance from the agent's continuous position to the nearest such
  wall-adjacent cell edge is <= 0.15 * grid_size (~0.148 raw units -- a
  deliberately tight "very close to the wall," not merely "in the cell next
  to a wall") AND `||v_t|| < ||v_{t-1}||` (decelerating). Requires `t>=1`.
- **Direction turn**: angle between `v_t` and `mean(v_{t-3:t-1})` >= 45
  degrees. Requires `t>=3`. Skipped (no label either way) if either vector's
  norm is below 1e-6 (effectively stationary -- angle undefined).
- **Corridor entry/exit**: the number of open 4-connected neighbors of the
  agent's current grid cell differs from the previous step's count.
  Requires `t>=1`.
- **Junction arrival**: current cell has >=3 open 4-connected neighbors AND
  the agent's cell index changed this step (or `t=0`, trivially "arrived" at
  the window start).

Per-event-type frequency and per-trajectory mean count logged as a wandb
table; 10 random trajectories rendered with events overlaid on the maze for
visual sanity-check (as instructed), before any signal scoring happens.

## 5. Boundary extraction (identical across all signals + baselines)

Greedy peak-picking (`pick_changepoints`, unchanged), `min_seg=8`, 5
boundaries, applied independently to each 60-step window. Three baselines,
scored identically to the 10 signals:

- **Fixed-interval**: boundaries at raw steps (10,20,30,40,50) -- current
  production baseline.
- **Random**: uniform-random boundaries respecting `min_seg=8`, averaged
  over 20 seeds.
- **Oracle** (metric B only): the 5 boundaries minimizing metric B exactly,
  via DP over `O(T^2 * 6)` segment costs per trajectory (T=60) -- an upper
  bound, not a candidate signal.

## 6. Metrics

**Metric A -- physical-event alignment:**
- Step-level: AUROC + average precision, positive = any step within +/-2 of
  a labeled event of that type (also computed pooled across all event
  types).
- Boundary-level: of the 5 chosen boundaries, precision = fraction within
  +/-2 steps of *some* labeled event; recall = fraction of events within
  +/-2 steps of *some* chosen boundary; F1 = harmonic mean. Computed overall
  and separately per event type.

**Metric B -- piecewise-linear latent reconstruction error:** using the 5
boundaries + window endpoints (7 points), linearly interpolate the latent
state between neighboring anchor points, mean squared distance (of the
in-between steps) to the actual latent state. Reported as: raw value,
improvement over fixed-interval (`(fixed - signal)/fixed`), and
oracle-reachability (`(fixed - signal) / (fixed - oracle)`).

All metrics: trajectory-level bootstrap, **10,000 resamples**, 95% CI
(percentile method, consistent with `confidence_intervals.py`'s existing
convention in this directory).

## 7. Selection / report split

r50's 25 maps split in half by `map_idx` (already present in `data.p`,
confirmed above -- no need for the `train_maps.pt`-index matching
`stage2_grid_validation.py` had to do for the 200-episode alignment subset).
**Split is by map, not by episode**, so no map's layout appears in both
halves. All *signal-selection* work (Stage 2's distributional diagnostics,
Stage 3's per-signal scoring used to rank candidates) uses only the
selection half; the *final* Stage 4 table recomputes only the **top-3**
signals (by selection-half metric-A F1 and metric-B reachability) on the
held-out report half. Expensive signals (cumulative error, action
sensitivity, BOCPD) computed on up to 300 episodes per half, sampled
stratified by map from whichever half is active.

## 8. Stop condition

If, on the report half, **no signal beats the fixed-interval baseline on
both metric A (pooled F1) and metric B (reconstruction error)**, conclude:
adaptive changepoint-based boundary placement offers no measurable benefit
over fixed-interval placement on this data, for any of the 10 signals
tested here -- and say so plainly in `RESULTS.md`, rather than reframing
around whichever sub-metric happens to look best.

## 9. What this study explicitly does not do

No `pldm.train` calls, no fine-tuning, no `pldm.planning`/MPC rollouts, no
GPU. The only "training" anywhere in this study is signal 2's small
post-hoc variance-head MLP (CPU, seconds, mirrors §8.2 exactly) -- not the
frozen level1/level2 world model itself.
