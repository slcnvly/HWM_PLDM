# Boundary-signal study: progress tracker

Read this file first if resuming after a session break. Update it at the end
of every stage (or sub-step within a long stage) -- commit + push each time.

## Status: Stage 1 COMPLETE. Root-cause investigation DONE (see diagnostic below). Stage 2 (predictor-free signals) code written; driver script + run still pending.

## Root-cause investigation (user's 3 follow-up checks, post-Amendment-2) -- DONE, diagnostic only (not Amendment 3)

Script: `investigate_1b_bias.py`. Full corpus (N=135,000 samples) for (2)/(3)
1-step, 300 episodes for (3) 5-step. Raw numbers: `results_investigate_1b_bias.json`.

**User's pushback on Amendment 2's hypothesis, and why it's right:** the
"frozen predictor never sees a real state at t>0 during training" story has
two holes -- (a) if training windows start at arbitrary in-episode
positions, a real observation at ANY t is exactly what SOME training
window's own first frame looks like, and (b) the rollout diagnostic showed
the very FIRST predicted step (k=0, the same situation as training) already
overshoots real movement by 11x -- so "never trained on this" can't be the
whole story either. Investigated exactly 3 things, no further, per the
user's explicit scope:

**(1) Training window start position -- REFUTES the Amendment-2 hypothesis,
confirmed.** `pldm_envs/diverse_maze/d4rl.py:236-237`
(`D4RLDataset.__getitem__`): `episode_idx = np.searchsorted(self.
cum_lengths, idx, side="right")`, `start_idx = idx - self.cum_lengths[
episode_idx-1]` -- `idx` indexes a FLAT cumulative range over every
(episode, frame) pair in the dataset. With a shuffling DataLoader, training
windows start at arbitrary in-episode positions, not just frame 0. The
user's counter-example (a) is correct and the original Amendment-2
mechanistic hypothesis is retracted.

**(2) Normalization layers -- no discrepancy, confirmed both by code and
empirically at full scale.** `pldm/models/utils.py:46-84` (`build_conv`,
used by both the MeNet6 encoder's conv trunk and `ConvPredictor`'s conv
layers): always `nn.GroupNorm` (per-sample statistics, batch-independent),
never `nn.BatchNorm2d`, regardless of `BackboneConfig.backbone_norm=
"batch_norm"`'s default value (`encoders/enums.py:28` -- not actually
consumed by this architecture's conv-building path). No Dropout in
`ConvPredictor` either. Empirical check (`check_2_eval_vs_train`, real
batch, deep-copied model, `model.eval()` vs `model.train()`, no_grad):
**max|encodings_eval - encodings_train| = 0.0, max|predictions_eval -
predictions_train| = 0.0 -- exactly identical.** Rules out a train/eval
statistics mismatch entirely.

**(3) Action-shuffle baseline + bias-corrected cosine + R², obs/proprio
separately, full corpus:**

| | obs | proprio |
|---|---|---|
| real-action pred L2 mean | 256.33 | 24.65 |
| shuffled-action pred L2 mean | 256.34 | 29.04 |
| **ratio real/shuffled (1-step)** | **1.0000** | 0.849 |
| **ratio real/shuffled (5-step, N=300 eps)** | **1.0004** | 0.998 |
| direction cosine (real action) | 0.0007 | 0.444 |
| direction cosine, bias-corrected | 0.0149 | 0.424 |
| R² (debiased) | 0.873 | 0.860 |

**New, concrete finding (not previously characterized): the obs-channel
prediction is statistically indistinguishable whether given the real
action or a randomly shuffled one, at both 1-step and 5-step (ratio =
1.0000 / 1.0004)** -- the obs branch's output does not respond to action
identity at all when re-anchored on a real state. Bias-correcting the
cosine barely moves it (0.0007 -> 0.0149, still ~0) -- even the *residual*
variation after removing the average predicted direction carries no real
directional signal. Proprio is different again: real actions give a
real (if modest) 15% error reduction over shuffled ones at 1-step, but
that advantage is gone by 5 steps (ratio -> 0.998) -- whatever
action-sensitivity proprio has doesn't survive compounding.

**On R² looking high (~0.87) despite Amendment 2's "~95% is bias" finding
-- not a contradiction, different denominators:** R² is 1 minus the
debiased residual's sum of squares over the TARGET's own total variance
across the whole dataset (which is huge -- different episodes/maps/times
span very different parts of the maze). Amendment 2's bias-fraction is
relative to the PREDICTION ERROR's own magnitude specifically. A
prediction can be "mostly a fixed bias relative to its own error size" and
still have high R² simply because the target varies enormously across the
full dataset -- these are both true at once, not in tension.

**Bottom line:** hypotheses (1) and (2) are ruled out by direct evidence,
exactly as the user argued. What actually characterizes the failure,
established here: the obs-channel one-step (and 5-step) prediction from a
freshly re-anchored real state is **action-invariant** -- it doesn't
change based on which action is given, real or shuffled, at either
horizon. The deeper mechanistic *why* behind that action-invariance was
explicitly out of scope for this round ("이 이상은 파고들지 마") and is
not investigated further here.

**User's decision after the Amendment-2 gate check:** go with option 3 --
proceed with predictor-free signals 6/7/8/10 for Stage 2 (+ oracle/fixed/
random baselines for Metric B), predictor-based signals (2,2b,3,3b,4,5,9,9b)
on hold. User pushed back on the "off-training-distribution" root-cause
hypothesis with two concrete counter-examples and asked for exactly 3
follow-up checks (no further digging beyond these), recorded as a
diagnostic here, NOT a new preregistration amendment.

**1. Background job still running (check this first):**
`investigate_1b_bias.py`, launched via `nohup ... > investigate_1b_bias.log
2>&1 &`, PID was 1950 (check `ps aux | grep investigate_1b_bias` -- if
gone, check the log's last line and `results_investigate_1b_bias.json`).
It's checkpointed every 100 episodes (`investigate_checkpoint.npz` +
`investigate_*.memmap` scratch files, all gitignored, ~54GB disk) -- if it
died, just rerun `nohup /home/goodwon01/.venvs/pldm_boundary/bin/python
investigate_1b_bias.py > investigate_1b_bias.log 2>&1 &` from the
`boundary_study/` dir and it resumes from the last checkpoint
automatically (don't delete the memmap/checkpoint files unless starting
fully over). At last check it was at 1300/2250 episodes. When it finishes,
it writes `results_investigate_1b_bias.json` and deletes its own
memmap/checkpoint scratch files.

**What it computes (all 3 checks the user asked for, plus the carried-over
bias-corrected cosine and R²):**
- (1) already answered by reading code, no run needed: `pldm_envs/
  diverse_maze/d4rl.py:236-237` (`D4RLDataset.__getitem__`) -- training
  windows start at ARBITRARY in-episode positions (idx indexes a flat
  cumulative range over all (episode,frame) pairs), not just frame 0. This
  REFUTES the Amendment-2 "only ever anchored at t=0" hypothesis, as the
  user correctly argued.
- (2) already answered by reading code (`pldm/models/utils.py:46-84`,
  `build_conv`, used by both the MeNet6 encoder and ConvPredictor): always
  `nn.GroupNorm` (batch-independent), never BatchNorm, no Dropout either --
  so model.eval() vs model.train() should be numerically identical. The
  script's `check_2_eval_vs_train()` empirically confirms this on a real
  batch (deep-copied model, no_grad) -- **already confirmed IDENTICAL
  (0.0 max diff) in a 1-episode smoke test**, full-corpus confirmation
  pending only in the sense that it reruns the same check.
- (3) action-shuffle baseline (real action vs. a random permutation of
  actions, 1-step full corpus + 5-step on 300 episodes) + bias-corrected
  direction cosine (subtract the GLOBAL mean predicted-step vector, then
  recheck cosine against real movement) + R² (debiased residual variance
  vs. the target's own total variance) -- for obs and proprio separately.
  **10-episode smoke test already showed a very informative pattern** (see
  git history / rerun to confirm at scale): obs-channel real/shuffled
  ratio ~1.0 (model's prediction is ESSENTIALLY INDEPENDENT of which
  action is given, both 1-step and 5-step), proprio ratio ~0.82 (some real
  action-sensitivity). R² was high (~0.88 obs, ~0.865 proprio) despite the
  Amendment-2 finding that ~95% of obs error is bias -- **these aren't
  contradictory**: R² is relative to the TARGET's own huge variance across
  the whole dataset, bias-fraction is relative to the PREDICTION ERROR's
  own magnitude -- different denominators. Explain this clearly when
  writing up the final numbers so it doesn't read as inconsistent.

**Once it finishes:** write up these 3 findings as a PROGRESS.md diagnostic
entry (not Amendment 3, per the user's instruction), commit, push.

**2. Stage 2 (predictor-free signals) -- code written, NOT yet run:**
- `metric_b.py`: piecewise-linear reconstruction error, oracle DP (exact,
  respects min_seg=8 against 0/T too), random baseline, fixed baseline.
  **Validated on a synthetic test** (oracle <= fixed and <= random mean,
  as it must be) -- see git history for the test snippet if it needs
  re-checking.
- `signals_predictor_free.py`: signals 7 (latent speed), 8 (direction
  change), 10 (action delta) -- straightforward, not yet run at all.
  Signal 6 (BOCPD): **had a real bug, now fixed and verified** -- a
  constant-hazard BOCPD's `P(run_length=0)` is mathematically pinned to
  exactly the hazard rate at every timestep regardless of data (verified
  both by derivation and by a flat-0.1-everywhere empirical result before
  the fix); switched the returned signal to `P(run_length <=
  SHORT_RUN_THRESHOLD=3)`, which IS genuinely data-dependent -- **verified
  on a synthetic two-regime sequence, shows a clear peak right at the true
  changepoint** (baseline ~0.17, peaks at 0.50 two steps after the true
  break). Fully documented in the function's docstring.
- **`data_split.py` corrected**: PREREGISTRATION.md SS7 wrongly assumed
  main and probe draw from the same 25-map pool. They don't -- main has
  25 maps, probe has 20 COMPLETELY DIFFERENT maps (verified by comparing
  actual layout strings, zero overlap). Corrected design (simpler and
  arguably more faithful to the original intent): **main = selection set,
  probe = report set**, no further per-half split needed since they're
  already disjoint. Already re-run successfully, `split.json` written and
  committed.
- **Not yet written:** the actual driver script that samples ~300 episodes
  from main (stratified across 25 maps, ~12/map), fits a 10-component PCA
  for signal 6 on a subsample, computes all 4 signals + fixed/random(20
  seeds)/oracle boundaries per episode, scores each via Metric B, and
  reports mean/improvement-over-fixed/oracle-reachability per signal. This
  is the main remaining work for Stage 2.

**Next action when resuming:** check on / let finish the background
investigation job, write up its results, then write and run the Stage 2
driver script described above. Report both together per the user's
request ("Stage 2의 predictor-free 신호 결과와 원인 규명 결과가 모두
나오면 멈추고 보고해줘").


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
- [x] **Variance head retrained on signal 1b** (`train_variance_head_1b.py`),
  reusing `variance_head.py`'s `VarianceHead`/`beta_nll_loss`/
  `train_variance_head` unchanged -- only the dataset-building step differs
  (1b's re-anchored errors + the real `z_t` each was computed from, instead
  of signal 1's window-start-anchored errors). Memory-safe subsample: 500
  train episodes (main, 30,000 samples), 400 val episodes (probe, 24,000
  samples) -- this machine has ~14GB free RAM (`free -h`), well under what
  the original SS8 Kaggle run had (which OOM'd twice even there); full
  main+probe would be ~8.7GB of fp16 encodings alone, too close to the
  edge to risk. **Result: constant baseline val_nll=1.2259 (sigma^2=3.680)
  vs. trained MLP val_nll=1.2126, best epoch 1 of 30** -- a modest
  improvement (~0.013 nats, ~1.1% relative), same early-epoch-overfit
  pattern as the original SS8.4 run on signal 1 (which also picked epoch 1
  of 30 and improved by a similarly modest ~0.011 nats/~0.5%). **Unlike the
  original run, the trained weights are saved and committed this time**:
  `variance_head_1b.pt` (17MB -- state_dict + input_mean/std + target_scale,
  everything needed to reload and call `predict_variance_mlp` later for
  signal 2b). Full per-epoch log in `results_variance_head_1b.json`.
- [x] **Bug-impact-scope check (item 4).** All callers of `obs_to_ij`/
  `sample_nearby_grid_location_v2` in the repo (`bug_impact_scope.py`'s
  header comment has the full grep): (1) `wrappers.py:104-110` (`xy_to_ij`,
  used by `find_shortest_path`'s BFS shortest-path/turn-counting), and (2)
  `evaluation/maze2d_envs_generator.py:53`
  (`Maze2DEnvsGenerator._sample_nearby_location` -> `sample_nearby_grid_
  location_v2`, using RENDER_STATS's buggy `obs_range_total`, feeding
  directly into `env.set_target()`).

  **But S5-S8's actual hard-difficulty (D13-16) instances did NOT go
  through path (2) at eval time.** The icml yaml
  (`large_diverse_25maps_l2.yaml`, `h_d4rl_planning.hard`) sets
  `set_start_target_path=".../maze2d_large_diverse_probe/
  starts_targets_13_16.pt"` with `override_config: true` -- the 40
  seed=42 instances are pre-saved and just loaded, not generated live.
  **This exact file was recoverable locally, byte-identical** (found under
  a cached Kaggle kernel output dir -- see `bug_impact_scope.py`'s
  `STARTS_TARGETS_PATH`), so this check uses the REAL instances directly,
  not a reproduction.

  **Result (all 40 instances = 80 points, starts+targets), using the
  corrected conversion for ground truth:** `on_wall_fixed_count = 0` --
  none of the real 40 hard-difficulty instances are actually inside a wall.
  (The naive "adjacent to a wall" check came back 100% for both starts and
  targets, but this is a maze-corridor artifact, not a finding -- verified
  separately that 36/36, i.e. 100%, of map 0's own open cells touch >=1
  wall cell regardless of any bug; flagged explicitly in the JSON so it
  isn't misread.) **The metric that actually isolates the bug's effect:**
  buggy vs. corrected conversion disagree on which grid cell 3/80 points
  (3.75%) belong to, and for **2 of those 3, the buggy conversion would
  have wrongly classified a real, physically-valid point as being inside a
  wall** (`on_wall_buggy=True`, `on_wall_fixed=False`) -- both on `start`
  points (idx 20, idx 36), zero false negatives. **So: these specific 40
  instances were never at risk (they bypassed the buggy code path
  entirely), but this quantifies that *if* something downstream ever
  validated points via the buggy `obs_to_ij` (e.g. `wrappers.py`'s
  `xy_to_ij`/`find_shortest_path`, used for turn-counting/BFS -- not
  checked further here, out of scope), it would wrongly reject ~2.5% of
  real, valid points as wall collisions.** The 80 extra seed=20260910
  instances (`kaggle_hard_n120`) were NOT recoverable locally (not in any
  cached `_output_.zip`; reproducing them exactly would need
  `Maze2DEnvsGenerator -> ant_draw -> gym`, none of which are installed in
  this lightweight analysis venv, and installing them just for this check
  wasn't judged worth the risk/effort) -- not included. Full per-point
  detail in `results_bug_impact_scope.json`. No original file modified.
- [x] **Full-scale event labeling run, item 5 -- DONE.** `run_stage1_events.py`
  on all 2250 episodes (main 1250 + probe 1000), ~17s wall-clock (pure
  numpy, no model forward pass needed for event labels). Logged online to
  `hwm-boundary-study` (see "Infra facts" above re: proceeding online
  despite the user's "offline for now" instruction predating their own
  login). Run: https://wandb.ai/goodwon01-chung-ang-university/hwm-boundary-study/runs/n0bqfkjw

  **Per-event-type rates (N=2250 episodes, 60 steps each), all plausible
  and non-degenerate (none near 0% or 100%):**
  | event | total count | mean/trajectory | frac. of steps positive |
  |---|---|---|---|
  | wall_contact | 22,482 | 9.99 | 16.7% |
  | direction_turn | 25,229 | 11.21 | 18.7% |
  | corridor_change | 4,479 | 1.99 | 3.3% |
  | junction_arrival | 3,173 | 1.41 | 2.4% |

  10 random trajectory/event overlay plots rendered and logged
  (`plots/event_overlay_ep*.png`) -- spot-checked a few by eye, event
  markers land on plausible-looking wall-hits/turns along the plotted
  paths, not obviously misaligned. Raw numbers in
  `results_stage1_events.json`.

## Stage 1: COMPLETE. Summary before Stage 2 (per the user's explicit stop request)

All 5 requested items done: (1) preregistration amendment with 1b/2b/3b/9b,
(2) diagnostics (a)-(c), (3) 1b-based variance head retrained+saved, (4)
bug-impact-scope check, (5) full event labeling run. **Headline finding:**
signal 1 (everything S5-S8 built on) is confirmed structurally confounded
with elapsed window position/distance (Spearman rho=0.57 vs t, r=0.44 vs
distance-from-start) via compounding directional drift in its open-loop
rollout (not magnitude collapse, diagnostic (b)'s more nuanced result) --
signal 1b does not share this confound. The wall-violation bug itself did
not taint S5-S8's actual hard-difficulty instances (they bypassed the buggy
live code path via a pre-saved file), but would misclassify ~2.5% of real
valid points as wall collisions if anything else in the pipeline relies on
the buggy conversion. Stopping here for the user's review before Stage 2
(candidate signal scoring) as requested -- Stage 2 design should treat
signal 1 and 1b (and their 2b/3b/9b derivatives) as testing genuinely
different hypotheses, not near-duplicates, per Amendment 1.

## Amendment 2 gate check: is 1b measuring what we think? -- DONE, TRIGGERED

Full writeup in PREREGISTRATION.md "Amendment 2". Summary:

**Infra note (fix this if touching `gate_check_1b.py` again):** the first
two attempts at this script held the full (N=135000, D=33282) float32
arrays for z_t/z_tp1/pred in RAM simultaneously (~54GB) -- this machine has
~14GB free. Both attempts died silently around episode 600/2250, and both
times the WSL2 VM itself had rebooted (`uptime` showed "up 0 min", journal
logs showed "corrupted or uncleanly shut down") -- almost certainly the
memory blowup triggering severe system-wide pressure that took the whole
VM down with it, not (or not only) a Windows-sleep coincidence. **Fixed**:
rewrote as a memory-safe, two-pass, checkpointed/resumable streaming
script -- per-episode scalars only in RAM, the one array that genuinely
needs a global reduction (`pred_err`, for the bias vector) goes to an
on-disk `np.memmap` (`gate_check_pred_err_fused.f32.memmap`, ~18GB scratch,
deleted on successful completion), with a `.npz` checkpoint every 100
episodes (`gate_check_checkpoint.npz`, also deleted on completion) so a
future interruption resumes instead of restarting. Ran clean end-to-end
after the fix, no further crashes. Separately, also flagged to the user:
if interruptions recur, check Windows sleep/power settings, since `nohup`
protects a process from terminal hangup but not from the whole WSL2 VM
being suspended/killed by the host OS.

**(4) Code citations (see PREREGISTRATION.md Amendment 2 for full detail):**
- Predictor residual handling: `pldm/models/predictors/conv_predictors.py:
  534-550` -- `pred_output.prediction` is always the absolute next-state
  by the time it's returned (residual addition, when `config.residual=
  true`, happens internally before returning).
- **Real training objectives**: `large_diverse_25maps_l2.yaml:113-115` ->
  `PredictionObs` (`pred_attr="obs"`, 16ch) + `PredictionProprio`
  (`pred_attr="proprio"`, 2ch), both `global_coeff=2.416154262252218`
  (yaml:131-134) -- NOT the fused `pred_attr="state"` space signal 1/1b
  have used throughout. `encodings = cat([obs_component(16ch),
  proprio_component(2ch)])` confirmed exactly via `torch.allclose`.
- **MPPI planner cost space**: `pldm/planning/planners/enums.py:60`
  (`cost_entity: str = "obs_component"`, default, no yaml override) +
  `enums.py:58` (`proprio_cost: bool = False`, default) +
  `pldm/planning/planners/mppi_planner.py:221-270` (`RunningCost.__call__`,
  plain MSE) -- planning cost uses **obs_component only, proprio fully
  excluded**, by two independent defaults.

**(5):** manual fused-space MSE matches `PredictionObjective(pred_attr=
"state")` exactly (`torch.allclose`, 3/3 episodes) -- no reimplementation
bug. But `pred_attr="state"` was never actually trained on, so this
confirms internal consistency, not that fused is the right comparison
space.

**(1)-(3), full corpus (N=135,000 samples):**

| | fused (original 1b) | obs-only (training+planner space) | proprio-only (training space) |
|---|---|---|---|
| ratio pred/copy (of means) | **8.83** | **20.37** | **1.01** |
| bias as % of pred error | 94.0% | 94.7% | 31.7% |
| residual-after-debias, % orig | 34.0% | 32.3% | 97.3% |
| direction cosine, mean | 0.020 | 0.0007 | 0.444 |

**Verdict: TRIGGERED.** Redefining 1b to match the planner's actual cost
space (obs-only) makes it WORSE (20.4x vs 8.8x), ruling out "wrong space"
as the sole explanation. The obs channel's fresh one-step prediction is
~95% a fixed, input-independent bias vector with ~zero direction
correlation (cosine 0.0007, 50.1% positive = indistinguishable from
random) -- the proprio channel degrades far more gracefully (31.7% bias,
cosine 0.44) but proprio is exactly what the planner ignores.

**Leading hypothesis:** the frozen predictor, per Amendment 1, is only
ever trained/evaluated as a single continuous rollout anchored on frame 0
of each window (no `prior_model`/`posterior_model` to re-anchor on later
real frames) -- it may never see "a real observation as `current_state` at
t>0" during training at all, making 1b's re-anchoring an off-training-
distribution query. Not separately ablated to 100% confirmation, but
well-supported by the code-level findings and the obs-vs-proprio
asymmetry.

**Consequence, not yet acted on -- needs the user's call:** every
predictor-based Stage 2 signal (2, 2b, 3, 3b, 4, 5, 9, 9b) re-anchors on a
real state at potentially large `t`, so all of them inherit this same
risk, not just 1b. Options put to the user: (a) only re-anchor near small
`t`, (b) gate-check each predictor-based signal individually before
trusting it, or (c) deprioritize predictor-based signals in favor of
6/7/8/10 (no predictor calls) pending further investigation. **Stopped
here, per the user's explicit request, before writing any Stage 2 signal
code.**

Full numbers: `results_gate_check_1b.json`. Script: `gate_check_1b.py`
(now memory-safe/resumable, see infra note above).

## Stage 2: predictor-free signals (6/7/8/10) + Metric B baselines -- DONE

Per the user's option-3 decision after Amendment 2 (predictor-based
signals 2,2b,3,3b,4,5,9,9b on hold). Script: `run_stage2_predictor_free.py`
(+ `metric_b.py`, `signals_predictor_free.py`). 300 episodes sampled from
main (selection set), stratified across all 25 maps (12/map). PCA for
signal 6 fit on 3660 samples from 60 of those episodes, 10 components,
71.8% explained variance (a lossy projection -- noted as a limitation of
this specific BOCPD approximation, not swept further per the
preregistered "fixed before scoring" rule).

**Metric B results (piecewise-linear reconstruction error, mean over 300
episodes -- see PREREGISTRATION.md SS6 for the exact definition):**

| | mean Metric B | vs. fixed | vs. oracle |
|---|---|---|---|
| oracle (DP upper bound) | 0.05814 | +31.3% | 100% (by definition) |
| **fixed (10,20,30,40,50)** | **0.08462** | 0% (reference) | 0% |
| random (20 seeds) | 0.08525 | -0.7% | -2.3% |
| signal_10 (action delta) | 0.09661 | -14.2% | -45.3% |
| signal_6 (BOCPD) | 0.09828 | -16.1% | -51.6% |
| signal_8 (direction change) | 0.10437 | -23.3% | -74.5% |
| signal_7 (latent speed) | 0.10727 | -26.8% | -85.5% |

**All four predictor-free signals are WORSE than fixed-interval placement
on Metric B, and worse than the random baseline too** (random is
essentially at parity with fixed, -0.7%). None comes close to the stop
condition's bar (PREREGISTRATION.md SS8: "beats fixed-interval on both
metrics") -- on this metric alone, none beats it at all. Ranked from least
to most bad: action delta > BOCPD > direction change > latent speed.

**Not yet done:** Metric A (event-alignment) for these same 4 signals --
the user's request here was specifically "Stage 2 진행 + 지표 B용
oracle/기준선," so Metric A wasn't computed this round. Full bootstrap
(n=10,000) CIs per PREREGISTRATION.md SS6 also not run yet -- these are
point estimates only. Both would be needed before treating this as a
final Stage 3 scoring pass; flagged as the natural next step if the user
wants the full picture (Metric A might tell a different story, e.g. signal
8/7 could still align with physical events despite bad latent
reconstruction).

Raw numbers: `results_stage2_predictor_free.json`.

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
