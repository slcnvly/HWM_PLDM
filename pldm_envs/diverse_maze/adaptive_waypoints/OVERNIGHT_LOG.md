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
