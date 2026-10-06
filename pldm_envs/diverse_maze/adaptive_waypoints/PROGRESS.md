# adaptive_waypoints progress / decision log (inference-design study)

The boundary-signal study keeps its own trail in `boundary_study/PROGRESS.md`. This file records the decisions
made while running the inference-design study unattended (user instruction 2026-10-05: follow the stated rules;
anything not covered gets the conservative choice, recorded here).

## Decisions

1. **Location prober for waypoint decoding.** The prober handed to the planner in eval-only mode is randomly
   initialised (`pldm/probing/evaluator.py:272-273`: `train_probers = eval_l1 = false` returns untrained probers).
   A diagnostic prober of the same architecture is trained locally on CPU on frozen pretrained-L1 features from
   the r50 probe split. No model weights are trained (the "no fine-tuning" rule is about the model).
2. **GPU quota accounting.** Remaining = allowed − used − (my estimate of the remaining time of my running
   kernels). Kaggle's own `time_reserved` is recorded alongside but not used, because its semantics are
   undocumented.
3. **Checkpoint.** `pretrained_baseline.ckpt` from dataset `seungwonryoo/hwm-finetune-checkpoints` (the file
   used for the SS6b baseline n=120), not a fresh HF download, so V0 is comparable with SS6b.
4. **Deterministic algorithms are enabled with `warn_only=True`.** A strict `use_deterministic_algorithms(True)`
   raises on any op without a deterministic kernel and would kill a 7h run. warn_only keeps the deterministic
   kernels where they exist and logs the rest. The remaining nondeterminism is measured by the step-1 validation.
5. **CRN seed = 0**, trial id = global index 0..119 in the SS6b order. Stage-1 replans, stage-2 replans and the
   initial nominal sequence use distinct generator keys (`stage` in {init, s1, s2}, `level` in {l1, l2, flat}).
6. **Per-env encoding and nominal rollout under CRN.** Batched conv results can depend on batch size. With CRN on,
   the L1 encoder (current obs and goal) and the nominal-trajectory rollout run per environment, so a trial's
   computation is identical whatever the chunk size. MPPI itself was already per environment.

## FLAG for the user (found while preparing C1; not fixed tonight, by the conservative rule)

**The boundary study and the changepoint-cache scripts feed raw, unnormalized inputs to the model.**
Training normalizes images with `state_mean/std`, plus actions and proprio velocity (`pldm/data/utils.py:97,117` →
`Normalizer.normalize_sample`, `pldm_envs/utils/normalizer.py:344-380`), and so does evaluation
(`maze_draw.py:104-105`, `wrappers.py:297-302`). But `compute_changepoints.py:126-133` and the boundary-study
helpers (`signal1_common.py:36-40`, `run_stage2_predictor_free.py:63-71`, and therefore
`generate_dp_segmentation_cache.py`) pass raw 0–255 images, raw velocity and raw actions. Consequences to
re-check: the signal-1 changepoint cache used for SS5/SS6b adaptive training, the surprise cache (SS8), every
boundary-study encoding (Metric A/B, gate checks), and the dp_segmentation cache used in SS9. Whether this
changed the conclusions is unknown until re-run with normalized inputs. All new inference-study CPU code (prober,
C1) normalizes inputs as evaluation does.

## 2026-10-06 follow-up (IN PROGRESS when the session's usage limit was reached)

- V5 launched on Kaggle (user waived the +2h buffer): kernel `hwm-inference-v5`, ~7.6h; resumable per 20-instance chunk.
- Running locally (nohup, logs next to scripts): corrected dp cache for full r50 (`boundary_study/dp_cache_fixed_{main,probe}.log`;
  old cache preserved as `changepoints_minseg8_dp_v1.pt`), corrected signal-1 cache (`signal1_cache_fixed_*.log` →
  `datasets/r50_local/signal1_fixed/`), Metric A / coarse / 16v2 reruns (`boundary_study/run_metric_a*_fixed.log`; old results `*_v1.json`),
  follow-up 1+2 (`followup/action_sensitivity_and_surprise.py`). Old signal-1 cache download → `experiments/old_signal1_cache/`.
- Kaggle CPU kernel `hwm-render-v0-states` renders V0 replan states for follow-up 3 (V0 part).
- Written, not yet run: `followup/decision_points.py` (task 3; r50 reduced to 100 episodes and V0 to backward replans + equal random
  others, because of CPU cost ~2.2 s/state). Not yet written: task 4 (linear BFS-distance probe), error_adaptive_l1.py / variance_head.py
  code fixes, Metric A comparison table, DECISION_POINTS.md, next-week GPU plan.

## 2026-10-06: r50 데이터의 성격 (CPU, 위치 기반 통계)

스크립트 `followup/trajectory_character.py` → `followup/results_trajectory_character.json`, 그림 `followup/trajectory_straightness_hist.png`.
무작위 행동 시뮬레이션은 Kaggle CPU 커널 `hwm-sim-random-r50`(`experiments/kaggle_sim_random_r50/`)로 만들었고 GPU 할당량은 쓰지 않았다.

### 1. 수집 방식: 순수 무작위 행동 (목표도, 계획기도 없음)
- r50의 `metadata.pt`: `sampling_mode: uniform`, `resample_every: 1`, `action_noise: 0`, `action_repeat: 4`(`id`), `episode_length: 100`,
  `qvel_norm_prior: true`(`uniform`). 맵당 50 에피소드이고 main 25개 맵, probe 20개 맵이다. 원본 설정은 `configs/maze2d_large/25maps.yaml:1-16`과 같고 에피소드 수만 다르다.
- 행동 생성: `data_generator_maze2d.py:349-352`에서 `sampling_mode == "uniform"`이면 `generate_uniform_trajectory`(`:314-327`)를 쓴다.
  `resample_every=1`이므로 **매 스텝** `sample_vector(max_norm=1)`(`pldm_envs/utils/utils.py:4-10`)로 새 행동을 뽑는다.
  크기는 U[0,1], 각도는 U[0, 2π]이고, 이전 행동이나 위치, 목표와 무관하게 i.i.d.로 뽑힌다.
  각 행동은 `ActionRepeatWrapper`로 시뮬레이터 4스텝 동안 반복된다(`:363-368`, `wrappers.py:204-219`).
- 시작 상태: `pick_random_start`(`data_generator.py:84-96`)가 `env.reset()`으로 무작위 위치를 고르고,
  속도는 `sample_vector(max_norm=5)`로 크기 U[0,5]의 무작위 초기 속도를 준다.
- 같은 파일에 목표 속도로 가속하는 구간과 OU 잡음을 섞는 "OU" 모드(`generate_ou_trajectory`, `:249-312`)도 있지만, r50에는 쓰이지 않았다.
  두 모드 모두 목표 위치를 향한 계획은 없다.
- 결론: **r50은 가속도 공간의 순수 무작위 행보다.** 목적 지향적 궤적은 전혀 없다.

### 2–5. 궤적 통계 (1 스텝 = 기록 스텝 = 시뮬레이터 4스텝, 속도 = 스텝당 변위)

| 출처 | n | 직진도 10스텝: 평균 / 중앙값 [q25, q75] | 직진도 50스텝 중앙값 † | 20스텝 내 재방문 (스텝당 / 셀 진입당) | 방향 반전 | 벽 접촉 엄격 / 중간 / 느슨 | 스텝당 이동 (셀) |
|---|---|---|---|---|---|---|---|
| r50 main | 1250 | 0.845 / 0.926 [0.786, 0.978] | 0.519 | 0.3% / 6.5% | 1.7% | 2.1% / 6.2% / 10.4% | 0.069 |
| 무작위 시뮬레이션, 데이터 초기 속도 (a1) | 1250 | 0.850 / 0.927 [0.792, 0.978] | – | 0.4% / 6.8% | 1.7% | 2.1% / 6.1% / 10.3% | 0.070 |
| 무작위 시뮬레이션, 초기 속도 0 (a2) | 1250 | 0.839 / 0.919 [0.780, 0.974] | 0.516 | 0.3% / 6.9% | 1.6% | 1.9% / 5.6% / 9.6% | 0.061 |
| 평가 V0 성공 (목표 도달까지) | 94 | **0.901 / 0.975** [0.891, 0.994] | **0.595** | 0.4% / **4.0%** | 1.2% | **1.1%** / 5.0% / 9.1% | **0.111** |
| 참고: 평가 V0 실패 (전체 485스텝) | 26 | 0.865 / 0.959 | – | 0.8% / 10.5% | 1.6% | 1.2% / 5.5% / 10.9% | 0.092 |

† 50스텝 직진도는 요청 외 추가 분석이다. 미로를 따라가는 경로는 직선이 아니라 꺾일 수밖에 없어서 목적 지향 궤적도 1보다 훨씬 낮다.

정의:
- 직진도: 덩어리 시작과 끝의 직선 거리를 덩어리 안 경로 길이로 나눈 값.
- 재방문: 셀이 바뀐 스텝 중 새 셀이 직전 20스텝 안에 이미 지나간 셀인 경우. 모든 스텝 대비 비율과 셀 진입 대비 비율을 함께 적었다.
- 반전: 연속 변위의 cos < 0. 두 변위가 모두 0.01셀보다 클 때만 센다.
- 벽 접촉: 벽 쪽 모서리까지 거리 ≤ d이고, 직전 스텝에 그 벽 쪽으로 움직이고 있었으며(>0.005셀), 벽 쪽 변위가 일정 비율 이상 줄어든 스텝.
  임계값은 엄격(d ≤ 0.07셀, 80% 감소), 중간(0.12셀, 50%), 느슨(0.20셀, 25%)이다.
  r50에서 벽 모서리까지 거리의 하한은 약 0.05셀이다(질점 반지름 효과, 1% 분위수 0.048).

읽기:
- **r50은 무작위 행동 시뮬레이션과 모든 통계에서 구분되지 않는다** (같은 과정이므로 예상대로다. 시뮬레이션이 데이터를 재현한다는 점검이기도 하다).
  초기 속도를 0으로 바꿔도 직진도와 재방문은 거의 같다. 그래서 무작위 초기 속도가 직진도를 만든 것은 아니다.
- **"무작위 = 제자리 흔들림"은 아니다.** 행동이 가속도이고 크기가 ≤ 1이라 관성이 무작위 가속을 평균내므로, 10스텝 직진도 중앙값은 0.93으로 높고 반전은 1.7%뿐이다.
  r50은 미로 안을 천천히(스텝당 0.07셀) 매끄럽게 표류하는 궤적이다.
- 목적 지향(평가 성공)과의 차이는 다음과 같다.
  - 직진도는 10스텝 0.926→0.975, 50스텝 0.52→0.60으로 높다.
  - 속도는 1.6배다.
  - 셀 진입당 재방문은 6.5%→4.0%로 적다.
  - 엄격 기준 벽 접촉은 2.1%→1.1%로 절반이다.
  데이터는 분포상 "완전 무작위" 끝에 있다. 다만 10스텝 규모의 직진도로는 목적 지향 궤적과 잘 구분되지 않는다.
  구분이 잘 되는 축은 속도, 재방문, 엄격 기준 벽 접촉이다.
- 벽 접촉 비율은 임계값에 **매우 민감하다**. r50 기준 엄격 2.1%, 중간 6.2%, 느슨 10.4%로 5배 차이가 난다.
  방향(r50이 평가 성공보다 많음)은 세 임계값 모두에서 같지만, 느슨한 기준에서는 차이가 12%까지 줄어든다.
  Stage 1의 원래 벽 접촉 라벨(16.7%, 1스텝 감속만 요구)은 느슨한 기준보다도 넓은 정의였다.
- 함의: 모델은 목표 지향 행동을 한 번도 보지 못했다. 그런데도 평가는 13–16셀 떨어진 목표를 향한 직진 위주의 궤적을 요구한다.
  r50 에피소드는 100스텝 동안 대략 7셀까지만 움직인다(C1).
  학습 데이터에 없는 "먼 목표를 향한 장거리 이동"을 플래너가 latent 공간 거리만으로 이어 붙여야 한다는 점은,
  잠재 목표 거리가 실제 미로 거리를 잘 따라가지 못하는 것(C1 ρ≈0.44, follow-up 4의 맵 단위 ρ≈0.13)과 맞물린다.
