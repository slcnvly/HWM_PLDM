# Boundary-signal study: progress tracker

Read this file first if resuming after a session break. Update it at the end
of every stage (or sub-step within a long stage) -- commit + push each time.

## Status: Stage 1 checkpoints verified (grid fix + rollout API confirmed); full-corpus event labeling next

## Execution-outage report (resolved)

Mid-session, every Bash call that *executed* code (`python3 -c ...`, running
a `.py` file, even via a plain `--version` on the project's own venv
interpreter) failed with exactly:
`The server-side auto mode classifier is temporarily unavailable (error),
so auto mode cannot determine the safety of Bash right now.` -- while
`echo`, `ls`, `grep`, and `python3 --version` (system Python only) kept
succeeding throughout. First failure was immediately after `grid_utils.py`
was written; recovered on its own after ~10 retries over ~10-15 minutes
(the exact commands run are the ones now sitting in this session's tool-call
history: `/home/goodwon01/.venvs/pldm_boundary/bin/python -c "print('ok')"`
and running `validate_grid_fix.py`/`verify_rollout_api.py` under that same
interpreter). **This was a tool-level outage (this session's Bash/code-exec
path), not a missing-data or environment problem** -- confirmed by checking
paths *during* the outage (`ls`/`grep` still worked) and re-confirmed
immediately after recovery:
- `pldm_envs/diverse_maze/datasets/r50_local/r50_dataset/main/{data.p,
  images.npy (3,637,515,128 bytes), metadata.pt, train_maps.pt}` -- present.
- `.../probe/{data.p, images.npy (2,910,012,128 bytes), metadata.pt,
  train_maps.pt}` -- present.
- `.../3-9-1-seed248_epoch=3_sample_step=15465472.ckpt` (225,058 bytes) and
  `.../load_from_l1248-seed248_epoch=5_sample_step=10789632.ckpt`
  (1,920,656 bytes) -- present.
- `pldm/configs/diverse_maze/icml/large_diverse_25maps_l2.yaml` -- present.
- `/home/goodwon01/.venvs/pldm_boundary/bin/python` -> resolves to the uv-
  installed CPython 3.10.21 -- present and, once the outage cleared, runs
  fine (`python -c "print('exec-check-ok')"` succeeded on first retry after
  recovery).
No environment/data problem was ever found; nothing needed fixing here.

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
- **wandb: logged in** (`/home/goodwon01/.venvs/pldm_boundary/bin/wandb
  login <key>`, key supplied by the user directly). Credentials cached in
  `~/.netrc`. `wandb.Api().default_entity` = `goodwon01-chung-ang-university`.
  `run_stage1_events.py`'s earlier forced `WANDB_MODE=offline` default has
  been removed -- runs now log online to project `hwm-boundary-study`
  directly. Runs so far: none yet (Stage 1 event labeling hasn't been run
  at full scale yet).

## Stage 0: Preregistration -- DONE

`PREREGISTRATION.md` written and committed before any scoring code ran.
Key locked-in decisions: 10 signals (§3), event-label thresholds (§4),
metrics + bootstrap n=10000 (§6), map-based selection/report split (§7),
stop condition (§8).

## Stage 1: Ground-truth event labeling -- IN PROGRESS

- [x] **Wall-violation bug: root cause, with exact file:line citations.**

  **Where n=10 actually comes from (the value used at lookup time):**
  `pldm_envs/diverse_maze/utils.py:310`, inside
  `sample_nearby_grid_location_v2`: `ij = obs_to_ij(anchor, obs_min_total,
  obs_range_total, n=len(map_layout))` -- `n` is the number of rows in the
  maze's own layout string (`map_key.split("\\")`), confirmed = 10 for every
  `maze2d_large_diverse` map via `train_maps.pt`. This `n` is what
  `obs_to_ij` (`utils.py:224-243`) actually divides by:
  `utils.py:238`: `grid_size = obs_range_total / n`.

  **Where 10.2 comes from (a bare, undocumented literal):**
  `pldm_envs/diverse_maze/data_generation/maze_stats.py:123-125`
  (`maze2d_large_diverse`'s `RENDER_STATS` entry):
  ```
  123:  # "obs_range_total": 9.87125,  # (obs_max_space - obs_min_space) / empty_blocks * total_blocks
  124:  "obs_range_total": 10.068675,  # total_blocks = 10.2
  125:  "obs_min_total": -0.636125,  # obs_min_space - obs_range_total / total_blocks
  ```
  Line 123 (commented out) is the original formula with `total_blocks=10`
  (the real grid dimension): `(8.248-0.351)/8*10 = 9.87125`. Line 124's live
  value instead used `total_blocks=10.2`: `(8.248-0.351)/8*10.2 =
  10.068675` -- the comment gives no justification for 10.2 over 10; it is
  not derived from maze geometry, agent radius, or wall thickness anywhere
  in this file or `data_generation/estimate_state_ranges.py` (which is only
  the empirical min/max sampler that produces `obs_min_space`/
  `obs_max_space`, not a source for 10.2 either). **10.2 appears to be a
  typo or an old manual tweak that was never reconciled with `utils.py:238`'s
  `n=10`.**

  **The mismatch, precisely:** line 125's `obs_min_total` formula divides by
  `total_blocks` too -- using the SAME 10.2, it reproduces
  `obs_range_total/10.2 = 0.987125` (the ORIGINAL, "n=10" grid_size) --  so
  `obs_min_total=-0.636125` is self-consistent with grid_size=0.987125.
  But `utils.py:238` always computes `grid_size = obs_range_total/n` with
  `n=10` (not 10.2), giving a DIFFERENT grid_size (1.0068675, ~2% larger)
  at actual lookup time. So `obs_min_total` (anchored to one grid_size) and
  every real `obs_to_ij` call (using a different, larger grid_size) disagree
  by a scale factor, not a constant -- **confirming the user's hypothesis
  exactly: this is a scale error, not an offset.**

  **User's hypothesis explicitly checked -- both parts confirmed:**
  1. *"위반이 한 방향에만 몰렸다" (violations skew to one direction only):*
     confirmed on a fresh, map-diverse 20-trajectory sample (one episode
     from each of maps 0-19, `validate_grid_fix_small.py`): pre-fix, 90/2020
     points (4.46%) violate, split `+i: 56, -i: 1, +j: 33, -j: 0` -- 99%
     (89/90) in the `+i`/`+j` direction, matching the original 200-episode
     finding (`+i` 49.5%, `+j` 50.2%, `-i`/`-j` ~0%).
  2. *"반 칸 이동으로 안 고쳐진다" / scale not offset:* two independent
     checks, both consistent with a scale error:
     - Violation "depth" (how far into the wrongly-assigned wall cell a
       point falls) vs. distance from the axis origin: Pearson r=0.53 over
       the 90 violations (positive, as a scale error predicts -- depth
       should grow with distance from `obs_min_total`; a pure offset would
       show no such trend).
     - Direct index-drift check (buggy index minus fixed index, ALL 2020
       points, not just violations), bucketed by coordinate-from-origin
       decile: drift magnitude increases roughly monotonically from decile 0
       (mean drift -0.002, near origin) through decile 8 (mean drift
       -0.181, far from origin) -- a staircase, not a flat line. (Decile 9
       drops back to 0, likely a boundary/sparse-sample artifact at the
       extreme edge of the maze -- noted, not hidden.) A pure additive
       offset would show a flat drift independent of position; this doesn't.
  3. **Full-scale confirmation (all 2250 episodes, both main+probe, not
     just the 20/200-episode samples):** 0 violations, 100.000% open,
     both directions -- `validate_grid_fix.py`.

  Scripts: `validate_grid_fix.py` (full corpus), `validate_grid_fix_small.py`
  (20-trajectory direction/proportionality breakdown, both pre- and
  post-fix). Both committed.

- [x] **`grid_utils.py` written and verified** (corrected `obs_to_ij` +
  `open_neighbor_count` + `wall_adjacent_edge_distance`).
- [x] **Rollout API for Stage 2 signals 4/5, verified on a real episode**
  (`verify_rollout_api.py`, episode 0, t=10): `model.level1.predictor.
  prior_model is None` **and** `posterior_model is None` (both -- not what
  was assumed when this note was first drafted; `z_stochastic=True` is a
  red herring here since neither network exists to use it against).
  **Consequence, confirmed by reading `sequence_predictor.py`'s
  `forward_multiple` (~line 231-305):** with both networks None, EVERY step
  falls through to the final `else` branch (`~line 303-305`,
  `predictor_input.append(actions[i])`) regardless of the `compute_posterior`
  flag's value -- so `compute_posterior` is a no-op for this checkpoint.
  What actually makes `forward_posterior`'s full-60-step call "one-step
  error" is NOT teacher-forcing/re-anchoring on real intermediate frames
  (there is no such mechanism here) -- it's a **single continuous open-loop
  autoregressive rollout from frame 0**, using real actions, where
  `current_state` is always the PREVIOUS step's own prediction
  (`current_state = pred_output.prediction`, line 321) except at i=0.
  Empirically confirmed: a fresh 1-step call re-anchored at the REAL t=10
  encoding gives `err1b[10]=2.04`, vs. `err1[10]=5.70` (signal 1, part of
  the 60-step chain that has already compounded drift since frame 0) for
  the exact same episode/t -- a real, mechanism-driven gap. **Correction to
  this note's first draft:** the original `_smoke_rollout.py` one-off script
  computed its own `err_direct` comparison with an off-by-one
  (`predictions[t]` instead of `predictions[t+1]`), giving a since-corrected
  wrong number (4.95); `signal1_common.py`'s `err1` (used everywhere from
  here on) was checked line-by-line against `compute_error_series` itself
  and matches it exactly (max abs diff 0.0 on a real episode) -- the
  `_smoke_rollout.py` bug never affected anything beyond that one throwaway
  print statement.
  **This is consistent with training, not a bug:** `pldm/objectives/
  prediction.py`'s `PredictionObjective` consumes whatever `forward_posterior`
  produces, and `jepa.py`'s own internal call hardcodes `compute_posterior=
  True` unconditionally -- since that flag is a no-op for this checkpoint
  either way, training and this eval-time computation are already using the
  identical open-loop-since-window-start mechanism. So `compute_changepoints.
  py`'s existing "one-step error" (signal 1 throughout this whole project) is
  real and matches training -- but it should be understood precisely as
  *"the window-start-anchored open-loop rollout's per-step error,"* not
  *"freshly re-anchored on the real previous frame at every step."*
  **This is exactly why signal 4 (5-step cumulative error, re-anchored fresh
  at each `t` using `state_encs=encodings[t:t+1]`) is a meaningfully
  different quantity from signal 1** for `t` more than a few steps into the
  window, not a redundant near-duplicate -- worth calling out explicitly in
  RESULTS.md later.
  Also confirmed: `forward_multiple(..., T=5)` returns 6 states (index 0 =
  the starting anchor, indices 1-5 = the 5 rolled-forward predictions,
  `predictions[-1]` is the one to compare against the real `z_{t+5}`), and
  the rollout is non-degenerate (step-to-step mean squared deltas:
  `[2.05, 0.18, 0.054, 0.015, 0.005]` -- evolving and decaying, not frozen or
  exploding).
- [x] **Diagnostics (a)-(c) requested before Stage 2 design -- all three run
  at full scale (main+probe, N=2250 episodes for (a)/(c); N=50 for (b)).**
  Scripts: `diagnostic_a_position_dependency.py`, `diagnostic_b_rollout_
  stopping.py`, `diagnostic_c_distance_confound.py`. Raw numbers in
  `results_diagnostic_{a,b,c}.json`, plots in `plots/`.

  **(a) Position dependency -- confirmed, large effect for signal 1, gone
  for signal 1b.** Pooled Spearman correlation between the signal's value
  and the window index `t` itself (N=2250 episodes x 60 steps =135,000
  pairs): **signal 1: rho=0.573 (p~0)** -- a strong, unambiguous structural
  dependence on elapsed window position. **Signal 1b: rho=-0.029** (p is
  tiny only because N is huge; the effect size itself is negligible) -- the
  re-anchored version shows essentially no such dependence, as expected if
  it's a genuine local one-step quantity. The reconstructed boundary-
  position histogram (signal 1, min_seg=8, 5 boundaries/episode, same
  algorithm S5/S6b's original cache used -- that exact cached file was
  never persisted locally, so this is a deterministic same-checkpoint,
  same-data reconstruction, not a different computation) is in
  `plots/diagnostic_a_position_dependency.png` alongside the mean-error-vs-t
  curves for both signals.

  **(b) Rollout "stopping" -- NOT confirmed in the hypothesized form; a
  different, still-important pattern found instead.** Mean open-loop
  predicted step size `||ẑ_{k+1}-ẑ_k||` (N=50 episodes) does *not* decay
  toward zero -- it does the opposite of "freezing": `k=0: 1.46` (a huge,
  one-off overshoot -- ~11x the real trajectory's own step size at k=0,
  `0.13`), then drops sharply and **stabilizes around 0.13-0.16 for k=3
  through k=19** -- almost exactly matching the real trajectory's typical
  step size for the rest of the window (`pred/real` ratio 1.06 at k=19).
  So the rollout is not stalling in latent-*magnitude* terms. **Combined
  with (a) and (c), the likely explanation for signal 1's growth over `t`
  is compounding *directional* error (the open-loop rollout takes
  steps of roughly the right size but the wrong direction once it can no
  longer see the real trajectory, so positional error random-walks upward)
  rather than the rollout's magnitude collapsing.** This changes the
  framing from "detects staleness" to "detects accumulated dead-reckoning
  drift" -- still a confound relative to local dynamics, just a different
  mechanism than hypothesized. Full curve in
  `plots/diagnostic_b_rollout_stopping.png`.

  **(c) Distance-from-start confound -- confirmed, moderate.** Pooled
  Pearson r = Spearman rho = **0.436** (p~0, N=2250 episodes) between
  signal 1 and `||z_{t+1}-z_0||` (latent distance traveled since the window
  start). Moderate, not dominant (r=0.436 -> ~19% of variance) -- real
  evidence for "adaptive was partly following how far the trajectory has
  drifted from the window's start," consistent with (a)'s window-index
  finding (distance and elapsed time are themselves correlated) and (b)'s
  directional-drift explanation, but signal 1 is not *purely* a distance
  proxy either. Scatter in `plots/diagnostic_c_distance_confound.png`.

  **Bottom line for Stage 2 design:** signal 1 (what S5-S8 actually used)
  is confirmed to carry a real, non-trivial structural confound with
  elapsed window position/distance, via compounding directional drift in
  the open-loop rollout -- not a subtle effect (rho=0.57 on t, r=0.44 on
  distance). Signal 1b (and by extension 2b/3b/9b) does not share this
  confound. **This means signal 1 and 1b should be treated as testing
  genuinely different hypotheses in Stage 2-4, not as near-duplicates** --
  and any result where signal 1 "wins" on metric A/B should be checked
  against whether it's just recovering the fixed-interval baseline's own
  behavior indirectly (since elapsed-time correlates with position in a
  60-step episode window too).
- [ ] Retrain §8's variance head on **1b's** errors (not the original
  signal 1's), per the user's item 3 -- save the trained weights to disk
  and commit them this time (the original run's weights were never
  persisted). Not started yet.
- [ ] Bug-impact-scope check (item 4): find every caller of `obs_to_ij`/
  `sample_nearby_grid_location_v2` in the repo, check whether the buggy
  10.2-divisor conversion was used for start/goal sampling in S6b's actual
  eval instances (seed=42, seed=20260910), and if so, what fraction of
  those points land in or adjacent to a wall cell. Not started yet.
- [ ] Run `event_labels.py` at full scale via `run_stage1_events.py` (main+
  probe, all ~2250 episodes) -- not yet run at full scale; only the
  grid-conversion piece it depends on has been validated so far. Per the
  user's item 5, log to wandb in **online** mode this time (see "Infra
  facts" above -- login already succeeded with a real API key; the user's
  "offline for now, I'll sync when I log in" instruction predates knowing
  that, flagged to them, proceeding online since it's strictly better and
  matches their actual goal).

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

Grid-conversion fix and the signal-4/5 rollout API are both verified now
(see Stage 1 above). Next: run `run_stage1_events.py` at full scale (main+
probe, ~2250 episodes), sanity-check the event-frequency numbers aren't
degenerate (0% or 100%), look at a few of the 10 rendered trajectory-overlay
plots by eye, then move to Stage 2 (signal computation) using the confirmed
`forward_multiple(state_encs=encodings[t:t+1], actions=..., T=k,
compute_posterior=False)` pattern for signals 4/5.
