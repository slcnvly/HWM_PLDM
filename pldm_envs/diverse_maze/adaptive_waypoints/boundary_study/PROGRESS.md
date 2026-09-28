# Boundary-signal study: progress tracker

Read this file first if resuming after a session break. Update it at the end
of every stage (or sub-step within a long stage) -- commit + push each time.

## Status: Stage 1 in progress (grid-conversion fix done; event labeling next)

## Infra facts worth knowing before touching anything here

- Full r50 dataset + frozen L1 checkpoint already extracted locally at
  `pldm_envs/diverse_maze/datasets/r50_local/r50_dataset/` (main=1250 eps,
  probe=1000 eps, 25 maps, `map_idx` in `data.p`). Source:
  `experiments/kaggle_package_r50/output/_output_.zip` (already existed on
  disk from a prior session, ~9.9GB unpacked). **No Kaggle kernel needed for
  this study** -- runs entirely local CPU.
- Run all `boundary_study/` scripts with
  `/home/goodwon01/.venvs/pldm_boundary/bin/python` (Python 3.10 + torch-cpu
  2.14 + omegaconf/numpy/scipy/sklearn/wandb) -- system Python is 3.14, which
  fails on this repo's mutable-dataclass-default pattern (see
  PREREGISTRATION.md §1). Do not `pip install` into system Python for this.
- `compute_error_series` (reused from `compute_changepoints.py`) verified:
  ~0.125s/episode on CPU. Full main+probe corpus (~2250 eps) costs ~5 min for
  the base error series alone.
- **wandb not logged in on this machine.** Log with `WANDB_MODE=offline` for
  now; ask the user to either run `wandb login` (suggest they use `!wandb
  login` in the session) or supply an API key before Stage 4's final upload.
  Runs so far: none yet.

## Stage 0: Preregistration -- DONE

`PREREGISTRATION.md` written and committed before any scoring code ran.
Key locked-in decisions: 10 signals (§3), event-label thresholds (§4),
metrics + bootstrap n=10000 (§6), map-based selection/report split (§7),
stop condition (§8).

## Stage 1: Ground-truth event labeling -- IN PROGRESS

- [x] **Wall-violation bug found and fixed.** Root cause: `maze_stats.py`'s
  `obs_range_total` for `maze2d_large_diverse` was computed with divisor
  10.2 but consumed everywhere else (`obs_to_ij`) with divisor n=10 -- a
  ~2% scale mismatch, not a constant offset (explains why it's 100%
  `+i`/`+j`-skewed and why a +/-0.5-cell shift made it worse, not better).
  Fix: single consistent n=10 divisor throughout ->
  `obs_range_total=9.87125` (was 10.068675), `obs_min_total=-0.636125`
  (unchanged). Verified on the existing 200-episode/12200-coordinate
  validation set: 315 violations (97.418% open) -> **0 violations (100.000%
  open)**, confirmed in both directions. Full derivation in
  PREREGISTRATION.md §2. Implemented as a local override in (to be written)
  `boundary_study/grid_utils.py` -- does NOT edit the shared
  `pldm_envs/diverse_maze/data_generation/maze_stats.py` (flagged as a
  separate finding for whoever next touches core grid-conversion code, since
  it may also affect goal-validity checks elsewhere, but that's out of scope
  for this study).
- [ ] Write `boundary_study/grid_utils.py` (corrected `obs_to_ij` + neighbor
  lookup helpers) -- next up.
- [ ] Compute the map-based selection/report split (25 maps, half/half) and
  persist it (so Stage 2-4 never re-derive it differently).
- [ ] Implement the 4 event labelers (wall contact, direction turn, corridor
  entry/exit, junction arrival) over the full main+probe corpus.
- [ ] Log per-event-type frequency table + 10 random trajectory/event
  overlay plots to wandb (`hwm-boundary-study`, offline for now).

## Stage 2: Candidate signal computation -- NOT STARTED

10 signals, §3 of PREREGISTRATION.md. Signal 2 (surprise) needs a fresh
variance-head MLP retrain first (original §8 weights were never saved to
disk -- only the JSON diagnostics summary survived). Expensive signals
(4, 5, 6) capped at 300 episodes/half, stratified by map.

## Stage 3: Scoring -- NOT STARTED

Metric A (event alignment: step AUROC/AP, boundary P/R/F1 overall + per
event type) and Metric B (piecewise-linear latent reconstruction error +
fixed-interval improvement + oracle-reachability), vs. fixed/random/oracle
baselines, bootstrap 95% CI (n=10000), on the selection half.

## Stage 4: Final table -- NOT STARTED

Top-3 signals from Stage 3 re-scored on the held-out report half. Final
wandb table + example trajectory plots. `RESULTS.md` written against the
Stage 0 stop condition.

## Next action if resuming cold

Write `boundary_study/grid_utils.py`, then the map split script, then the
event labelers -- in that order, each committed+pushed separately per the
user's stage-level (not file-level) commit cadence for this task.
