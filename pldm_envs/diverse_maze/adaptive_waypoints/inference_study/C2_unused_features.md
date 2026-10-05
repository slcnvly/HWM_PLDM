# C2: unused inference features in the original code (git history)

Sources: this repo's history (remote `upstream` = kevinghst/HWM_PLDM). The public history begins with a squashed
`73d0542 first commit` (2026-03-08) that already contains the full internal codebase (module `hjepa/`, renamed
to `pldm/` in `b6d0bd5`), followed by cleanup commits `eb44492 first round delete` (−24,363 lines, 191 files,
mostly the "wall" environment and its configs), `739ed25 delete round 2. code works` (−2,533 lines, including
the SGD planner) and `36801e2 last delete. code runs`. **No commit adds any of these features. They are all present
in the first commit, so the history cannot say when they were added or why they were dropped.** What the
history can say is where they were still used at release time:

| feature | in first commit | used anywhere at release? | trace of intent |
|---|---|---|---|
| `determine_optimal_depths` | `hjepa/planning/d4rl/hmpc.py:152` and a twin in `hjepa/planning/wall/hmpc.py:244` | **No caller in either file or anywhere else** (git grep on 73d0542) | Loops L2 plan depth from `min_plan_length` to `max_plan_length`, plans once per depth, and picks the **smallest depth whose predicted final L2 latent is within `depth_probe_threshold` of the goal** (else the closest one); returns depth × `step_skip` as a step budget. The wall version contains commented-out code probing the predicted locations per depth. **This is an adaptive planning-depth mechanism, implemented and then left uncalled.** |
| `probe_depth` | config field only (`planners/enums.py`), set `true` in the deleted wall config `h_wall_planning` (removed in eb44492), `false` in the maze config | No `.probe_depth` read anywhere in 73d0542 | The wall-environment hierarchical config had `probe_depth: true` with `l2_depth_probe_threshold: 1`. So depth probing was switched on for the wall experiments, but the code that read the flag was already gone from the public snapshot. |
| `depth_probe_threshold` | config field, read only inside `determine_optimal_depths` | No (its only reader is dead) | Same as above. |
| `num_refinement_steps` | `MPPIPlanner.__init__` argument, stored (`mppi_planner.py:282, 352` in 73d0542) | Never read | The comment `# add refinement steps?` above `command()` survives to this day. The deleted wall configs' `mppi` block also had `num_refinement_steps: 1`. Multi-iteration MPPI refinement was planned but never wired in. |
| `z_reg` / `z_reg_coeff` | MPPI computes `z_reg` (`mppi_torch.py:362-363` in 73d0542) | **Not added to the MPPI cost**, then or now | The live version was in the **SGD planner** (`pldm/planning/planners/sgd_planner.py`, deleted in 739ed25). There `z_reg = −log N(z; prior_mu, prior_var)·z_reg_coeff` was added to the objective. MPPI copied the computation but never added the term. The maze config's `z_reg_coeff: 0.1` is a leftover from SGD planning. |
| `final_trans_norm_cutoff` | `d4rl/enums.py` and `wall/enums.py` config fields, plus `hjepa/planning/wall/utils.py:27 determine_proximities` | `determine_proximities` has no caller | `determine_proximities` returns **the first step at which the L1 embedding is within `final_trans_norm_cutoff` of the target**: a distance-based criterion for handing over to the final flat-L1 stage instead of a fixed `final_trans_steps`. Dead at release. |
| `error_threshold` | config field (maze and wall) | Never read in 73d0542 | No surviving reader. (The fork's SS7 error-adaptive L1 uses a separate field, `error_adaptive_l1_threshold`.) |

Related: the authors added one inference-time option *after* release: `69f0222` (2026-03-25, "option to use
empirical l2 latent mean and std to init mppi"), i.e. `l2_use_latent_mean_std`, which this study tests as V1.

**Reading.** Yes, there are clear traces that the original authors tried adaptive inference-time decisions. Both
hierarchy-specific adaptive mechanisms were implemented but uncalled in the released code:
(1) adaptive L2 planning depth by probing which depth reaches the goal (`determine_optimal_depths`, enabled by
`probe_depth: true` in the wall-env config), and (2) a distance-based switch to the final flat stage
(`determine_proximities` with `final_trans_norm_cutoff`). The public history starts after these were already
dead, so whether they were dropped for not helping, for cost, or just left over cannot be determined from git.
