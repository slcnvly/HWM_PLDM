# Overnight log (2026-10-05 → )

Unattended run of the inference-design study (INFERENCE_PREREG.md). Updated at the end of every stage.
Decisions made without the user are listed in `PROGRESS.md` (this directory).

| time (UTC) | stage | status | note |
|---|---|---|---|
| 2026-10-05 | 0 prereg | done | INFERENCE_PREREG.md committed before any code/GPU |
| 2026-10-05 10:05 | 1 CRN implementation | done | `ce3f2c4`: per-(seed,trial,stage,step,level) generators for MPPI noise + initial U, per-env encoding/nominal rollout when CRN is on, `deterministic_algorithms` flag (warn_only), V4 flags, diag recorder. CPU tests: default path bit-identical to original; with CRN, chunk layouts 4 / 2+2 / 1x4 give bit-identical actions and waypoints. |
| 2026-10-05 10:05 | prober | done | diagnostic location prober (frozen encoder, CPU): held-out error 0.026 cells mean, 0.029 on unseen maps |
| 2026-10-05 10:10 | 1 GPU validation | launched | kernel `hwm-inference-validate` (quota: 30h allowed, 0 used) |
| 2026-10-05 10:12 | C1 | done | `inference_study/results_c1_latent_vs_maze_distance.json` |
| 2026-10-05 10:20 | C2 | done | `inference_study/C2_unused_features.md` |
| 2026-10-05 10:30 | 1 GPU validation v1 | failed (env) | Kaggle base image dropped `libgl1-mesa-glx`; apt aborted the whole install so `GL/osmesa.h` was missing and mujoco_py failed to compile. Fixed by installing packages one by one (+`libgl1`, `libglx-mesa0`). ~10 GPU-min used. |
| 2026-10-05 13:09 | 1 GPU validation | **PASS** | instances 0-19, chunk 20 vs 10+10: success 20/20 identical, steps 20/20 identical, **all 20 agent trajectories bit-identical at every step**. 18/20 success. Chunk of 20 = 72.7 min, chunk of 10 = 36.1 min. Data: `inference_study/runs/validate/`. |
| 2026-10-05 13:12 | 2 V0 + 3 V1 | launched (parallel) | quota before V0: 30 allowed, 2.72 used, 27.3 remaining ≥ 9.6; before V1: 19.7 remaining (after V0's 7.6h reservation) ≥ 9.6 |
| 2026-10-05 16:11 | V0 + V1 (1st try) | **never ran** | Both pushed seconds apart at 13:12; both showed RUNNING/QUEUED for ~3h, then ERROR with empty logs, no output and **no GPU quota consumed** (used stayed 2.72h). No failure message from the API. |
| 2026-10-05 16:15 | V0 relaunched alone | running | quota usage confirmed to increase (2:43→2:50 in 4 min) before launching the next |
| 2026-10-05 16:20 | V1 relaunched | running | usage now increases ~2x wall-clock (both consuming). Monitor checks quota growth every 15 min to catch silent stalls. |
| 2026-10-05 23:12 | 2 V0 | done | **94/120 (78.3%)**, mean steps (successes) 184.3, median 155. Chunk 0 is bit-identical to the validation run's cs20 chunk (different Kaggle session): CRN gives cross-session reproducibility. Failure types (n=26, overlapping): wrong path 24, stagnation 5, unreachable carrot 3, oscillation 0, slow progress 0, unclassified 1. |
| 2026-10-05 23:15 | 3 V2 | launched | quota: used 16.36h, V1 remaining ~0.7h → 12.9h remaining ≥ 4.5+2 |
| 2026-10-05 23:31 | 3 V1 | done | **101/120 (84.2%)**, mean steps 157.9. vs V0: flips 16 (V0 fail→V1 success) / 9 (reverse), McNemar p=0.23; steps on 85 both-solved: V1 faster 50 / slower 35, mean −30.0, Wilcoxon p=0.012 (Holm, m=4: 0.046 before V2). L2 clamp fraction rose 0.67→0.86. |
| 2026-10-05 23:33 | 3 V3 | **SKIPPED (quota rule)** | used 16.98h + V2 remaining est 4.19h → 8.83h left < V3 est 14.7h + 2h. (V3 alone would also exceed Kaggle's 12h session limit.) |
| 2026-10-05 23:33 | 3 V4 | **SKIPPED (quota rule)** | rule: a skipped GPU job skips all later GPU jobs. (Also, independently: V4 est ~10h + 2h > 8.83h − V2.) |
