# 미로 평가의 계층적 추론 루프 (코드 수준 문서)

브랜치 `adaptive-waypoints`, 2026-10-03 기준 코드. GPU 실행 없이 코드 읽기와 CPU 확인으로 작성했다
(CPU 확인: 모델 텐서 모양, 비용 함수 모양, L2 잠재 행동 경계 추정. 스크립트는 문서 끝 "부록 B").
경로는 모두 `hwm_pldm/` 저장소 루트 기준이다. 별도 표기가 없으면 "현재 값"은 우리 §6b/§9 hard 평가
설정을 뜻한다: `large_diverse_25maps_l2.yaml`과 Kaggle 커널의 override
(`experiments/kaggle_dp_segmentation_finetune/run.py:315-331`: `hard.n_steps=500`,
`hard.level2.mppi.num_samples=200`, `l2_latent_bounds_percentile=2.0`, chunk당 `n_envs=20`).

표기: B = 한 chunk의 환경 수(우리 평가 20), K = MPPI 샘플 수, T = 계획 길이.
"MPC 스텝" = `env.step` 한 번 = 시뮬레이터 스텝 4번(아래 §1.0).

---

## 0. 한눈에 보기

```
매 MPC 스텝 i (Stage 1: i = 0..469)
 ├─ i % 4 == 0 이면 재계획 (L2, L1 모두)
 │   ├─ L1 인코더: 이미지(B,3,98,98) + 속도(B,2) → 잠재 상태 (B,18,43,43)
 │   ├─ L2 MPPI: 8차원 잠재 행동 z를 K=200개 × T_L2 스텝 샘플 → L2 예측기로 굴림
 │   │      비용 = Σ_t MSE(예측 obs_t, 목표 이미지 obs)   [16×43×43 obs 공간]
 │   │      → 웨이포인트 열 pred_obs (T_L2+1, B,16,43,43)
 │   ├─ L1 목표 ← pred_obs[1]  (항상 "첫 번째" 웨이포인트)
 │   └─ L1 MPPI: 2차원 행동을 K=1000개 × 10 스텝 샘플 → L1 예측기로 굴림
 │          비용 = Σ_{t=1..10} MSE(예측 obs_t, 웨이포인트 1)
 │          → 행동 열 (B,10,2)
 ├─ 실행: 마지막 재계획에서 나온 행동 열의 (i % 4)번째 행동
 └─ 환경: 같은 행동을 4번 반복 → 새 이미지, 보상(마지막 서브스텝에서만 판정)
Stage 2 (15 스텝): 계층 없이 L1만으로 목표 이미지를 직접 추적
총 485 MPC 스텝, 조기 종료 없음. 성공 = 보상 1이 처음 나온 스텝.
```

---

## 1. 전체 흐름: 관측에서 실행까지

### 1.0 진입점과 단계 구성

| 단계 | 위치 |
|---|---|
| `pldm.train` main → `eval_only`이면 `trainer.validate()` | `pldm/train.py:722-733` |
| `validate()`는 전체를 `fixed_rng(seed + epoch·1009 + 17)` 안에서 실행 | `pldm/train.py:628-631` |
| L2 잠재 행동 경계 계산(평가 전에 한 번) | `pldm/train.py:635-650` |
| `Evaluator.evaluate()`: probing을 먼저 하고 L2 계획 평가 | `pldm/evaluation/evaluator.py:362-373, 408-455` |
| 난이도별 설정 병합 후 `HierarchicalD4RLMPCEvaluator` 생성 | `pldm/evaluation/evaluator.py:259-343` |
| chunk(=`n_envs_batch_size`)별로 planner를 새로 생성 | `pldm/planning/mpc.py:164-188` |
| **Stage 1**: 계층 계획 `_perform_mpc(h_planner, bilevel_planning=True)` | `pldm/planning/mpc.py:231-236` |
| **Stage 2**: 평면 L1 계획 `final_trans_steps` 스텝 | `pldm/planning/mpc.py:238-244` |
| Stage 1 길이 = min(n_steps − final_trans_steps, max_plan_length_l2 · step_skip) = min(485, 470) = **470** | `pldm/planning/mpc.py:429-436` |
| Stage 2 길이 = `final_trans_steps` = **15** | `pldm/planning/mpc.py:426-428`, `pldm/planning/d4rl/enums.py:21`, yaml `:183` |
| 합계 **485 MPC 스텝** (§9 시행별 데이터에서 실패 시행의 step이 전부 485였던 것과 일치) | `pldm/planning/d4rl/hmpc.py:85-96` |

한 MPC 스텝의 의미: `ActionRepeatWrapper`가 같은 행동을 4번 실행한다(`action_repeat=4`, `mode='id'`,
yaml `:190-191`, `pldm_envs/diverse_maze/wrappers.py:204-219`). 학습 데이터도 같은 설정으로 생성되었으므로
(`pldm_envs/diverse_maze/data_generation/maze_stats.py:33-34`) 데이터의 1 스텝과 평가의 1 MPC 스텝은 같은 단위다.

### 1.1 Stage 1, 한 MPC 스텝 i의 순서 (`pldm/planning/mpc.py:440-600`)

**(a) 재계획 여부 판단**: `if i % self.config.replan_every == 0` (`mpc.py:441`). 현재 값 4.

**(b) 현재 관측 수집** (재계획 시에만)
- 이미지 `obs_t`: (B, 3, 98, 98), 정규화됨. 렌더링은 `render_umaze` → `normalizer.normalize_state`
  (`pldm_envs/diverse_maze/maze_draw.py:81-107`, `wrappers.py:286-295`). `stack_states=1`이므로 프레임 1장(`mpc.py:585-590`).
- 속도 `curr_proprio_vel`: (B, 2), 정규화됨 (`mpc.py:452-458`, `wrappers.py:297-302`). 위치(`proprio_pos`)와 `locations`는 사용하지 않음(None).

**(c) L2 계획 길이**: `plan_size = max((470 − i + 9) // 10, min_plan_length=3)` (`mpc.py:468-470`).
i=0에서 47, 이후 10 스텝마다 1씩 줄어든다.

**(d) `TwoLvlPlanner.plan`** (`pldm/planning/planners/two_lvl_planner.py:25-85`)

| 순서 | 입력 → 출력 | 위치 |
|---|---|---|
| d1. L1 인코더 | 이미지 (B,3,98,98) + 속도 (B,2) → `BackboneOutput`: `encodings` (B,18,43,43) = `obs_component` (B,16,43,43) ⊕ `proprio_component` (B,2,43,43) | `two_lvl_planner.py:38-50` |
| d2. L2 계획 | 위 `BackboneOutput` (L2 인코더를 거치지 않음, `repr_input=True`) → `PlanningResult`: `actions` (B, T_L2, 8), `pred_obs` (T_L2+1, B, 16,43,43) | `two_lvl_planner.py:51-56`, `mppi_planner.py:384-530` |
| d3. L1 목표 설정 | `l2_result.pred_obs[1]` (B,16,43,43) → flatten (B, 29584) | `two_lvl_planner.py:73-75`, `objectives_v2.py:48-56` |
| d4. L1 계획 | 같은 `BackboneOutput`, `plan_size = l2_step_skip = 10` → `actions` (B,10,2)(환경 단위), `pred_obs` (11,B,16,43,43) | `two_lvl_planner.py:77-83` |

참고: L2 인코더는 `IdentityEncoder`이고 L1 인코딩을 그대로 통과시킨다
(`pldm/models/encoders/encoders.py:424-455`; CPU 확인: L2 인코더 출력 == L1 인코딩, `torch.equal`=True).
따라서 L2 상태 공간은 L1 잠재 공간(18×43×43 = 33,282차원)과 같다.

**(e) `MPPIPlanner.plan` 내부 (두 레벨 공통)** (`pldm/planning/planners/mppi_planner.py:384-530`)
1. **환경마다 순차로**(`for i in range(batch_size)`, `:443`) MPPI 제어기 `ctrls[i]`를 실행한다.
   - `plan_size`가 직전보다 작으면 줄어든 만큼 nominal 행동 열 `U`를 한 칸씩 당긴다(`:444-446`, `mppi_torch.py:174-180`).
   - `change_horizon(plan_size)`: `U`를 잘라내거나 `u_init`(0)으로 늘린다(`:448`, `mppi_torch.py:239-248`).
   - `command(..., shift_nominal_trajectory=False)`(`:452-459`)가 MPPI를 **한 번** 갱신한 뒤 `U` 전체를 반환한다
     (`u_per_command=-1`, `:367`, `mppi_torch.py:230-232`).
2. 갱신된 nominal `U`(가장 좋은 샘플이 아니라 가중 평균)를 모아 (B,T,nu)로 쌓는다(`:461`).
3. 이 nominal 행동 열로 다시 예측기를 굴려 `pred_obs` (T+1,B,…)를 얻는다(`:476-486`). 인덱스 0이 현재 상태다.
4. 행동 정규화와 역정규화: `action_normalizer` 적용(`:491-492`). L1은 `unnormalize_action`으로 환경 단위로 바꾸고(`:497-498`),
   L2는 잠재 행동이라 그대로 둔다(`:494-496`).

**(f) MPPI 한 번 갱신** (`pldm/planning/planners/mppi_torch.py:201-237, 373-403`)
1. 노이즈 ε ~ N(0, Σ)를 (K, T, nu)로 샘플한다(`:376`). **Σ = diag(noise_sigma)는 공분산이다**
   (`mppi_planner.py:329-335` → `MultivariateNormal(covariance_matrix=…)`, `mppi_torch.py:112-114`).
   그래서 표준편차는 √noise_sigma다. L2는 √10≈3.16, L1은 √5≈2.24.
2. 섭동 행동 = U + ε, 이를 `_bound_action`으로 제한한다(`:378-384, 405-411`). 제한 방식은 §4.3.
3. 예측기로 T 스텝 굴리며 스텝별 비용을 더한다(`_compute_rollout_costs`, `:266-371`). 비용은 §5.
4. 총비용 S_k = 롤아웃 비용 + Σ_t U_t·(λΣ⁻¹ε_t) (`:391-402`).
5. 가중치 ω_k ∝ exp(−(S_k − min S)/λ) (`:218-222`), U ← U + Σ_k ω_k ε_k (`:223-229`).
   λ = 0.0025(yaml `:214, :230`)라서 사실상 최소 비용 샘플 쪽으로 강하게 쏠린다.

**(g) 실행** (`mpc.py:525-542`)
- `planned_actions = planning_result.actions[:, i % replan_every:]`이고, 이 중 `[j, 0]`, 즉 **최근 L1 계획의 (i mod 4)번째 행동**을 실행한다.
  재계획 사이의 3 스텝은 새로 계획하지 않고 저장된 계획을 이어서 실행한다. 실행되는 행동의 모양은 환경마다 (2,)이고 환경 단위다.
- `env.step(action)` → `ActionRepeatWrapper`가 같은 행동으로 시뮬레이터를 4번 스텝한다(`wrappers.py:204-219`).
  이 래퍼는 **마지막(4번째) 서브스텝의 (obs, reward, done, info)만 반환한다**.
- `NormEvalWrapper.step`이 새 위치를 이미지로 렌더링해 반환한다(`wrappers.py:312-324`). `done`은 루프에서 쓰이지 않는다.

**(h) 다음 스텝 준비**: `obs_t = current_obs` (`mpc.py:585-586`).

### 1.2 Stage 2 (`mpc.py:238-244`)
- planner는 chunk 시작 때 만든 **평면 L1 MPPIPlanner**다(`mpc.py:175`, `pred_encoder=None`, `mpc.py:120`). Stage 1의 L1 제어기와는 별개이므로
  Stage 1의 warm-start를 이어받지 않는다.
- 목표: 목표 이미지의 L1 `obs_component`를 flatten한 (B, 29584) (`mpc.py:309-316`).
- `plan_size = min(n_steps − i, level1.max_plan_length) = min(500 − i, 50) = 50` (`mpc.py:471-474`). 4 스텝마다 재계획한다.
- Stage 1에서 성공했는지와 무관하게 항상 실행된다.

### 1.3 텐서 모양 요약 (CPU로 확인)

| 대상 | 모양 |
|---|---|
| 관측 이미지 | (B, 3, 98, 98) |
| L1/L2 잠재 상태 `encodings` | (B, 18, 43, 43), flatten 시 33,282 |
| `obs_component` (비용 계산 공간) | (B, 16, 43, 43), flatten 시 29,584 |
| L1 행동 | (B, 10, 2) |
| L2 잠재 행동 z | (B, T_L2, 8) (`hjepa.level2.predictor.z_dim=8`, yaml `:95`) |
| L2 `pred_obs` (웨이포인트 열) | (T_L2+1, B, 16, 43, 43) |
| L1 `pred_obs` | (11, B, 16, 43, 43) |
| L2 posterior 입력(학습 때만 사용) | (B, 20) = 10 스텝 × 2차원 행동 (yaml `:102-104`) |

---

## 2. 상위 레벨(L2) 재계획 시점

- **4 MPC 스텝마다, 무조건** 재계획한다. 조건부 재계획은 없다. `mpc.py:441`의 `i % replan_every == 0`이
  `TwoLvlPlanner.plan` 전체를 호출하므로 **L2와 L1이 항상 함께 재계획된다**(`two_lvl_planner.py:51-83`).
  L2만 따로 주기를 둘 설정은 없다.
- 설정값: `h_d4rl_planning.replan_every = 4` (yaml `:193`; 기본값 1, `pldm/planning/enums.py:43`).
- L2 한 스텝은 학습상 10 MPC 스텝 간격(§4)인데 재계획은 4 스텝마다 하므로, **웨이포인트 한 구간 동안 L2 계획이 약 2.5번 새로 만들어진다**.
- 계획 길이는 남은 Stage 1 스텝에 맞춰 줄어든다: `(470 − i + 9)//10`, 최소 3 (`mpc.py:469-470`).
  `max_plan_length_l2=47`(hard, yaml `:269`), `min_plan_length=3`(yaml `:220`).
- warm-start: 재계획 사이에 `U`를 시간축으로 당기지 않는다. `plan_size`가 줄어드는 재계획(10 스텝에 한 번)에서만 1칸 당긴다
  (`mppi_planner.py:444-446`). 즉 4 스텝이 지나도 `U`의 0번 원소는 그대로 "지금부터의 첫 L2 스텝"으로 다시 쓰인다.
- 쓰이지 않는 기능: `determine_optimal_depths`(깊이 탐색, `hmpc.py:152-196`)는 정의만 있고 호출되지 않는다.
  `probe_depth`, `depth_probe_threshold`(yaml `:224-225`)도 계획 루프에서 읽지 않는다.

## 3. 하위 레벨(L1) 목표 전환 시점

- **L1은 항상 1번 웨이포인트(`pred_obs[1]`)만 목표로 삼는다**(`two_lvl_planner.py:73-75`). `pred_obs[0]`은 현재 상태이고,
  `pred_obs[2:]`는 L2 비용 계산과 로깅에만 쓰인다.
- **"다음 웨이포인트로 넘어가는" 로직은 없다.** 4 스텝마다 L2가 새 계획을 만들고, 그 새 계획의 1번 웨이포인트가 새 목표가 된다.
  거리 임계값이나 도달 판정, 정해진 스텝 수에 따른 전환은 없다. 목표가 바뀌는 시점은 재계획 시점(§2)과 같다.
- L1 계획 길이는 **항상 `l2_step_skip` = 10**이다(`two_lvl_planner.py:80`, 값은 `mpc.py:79-83`에서 `self.model.config.step_skip`,
  yaml `:3, :76`). L1은 "10 스텝 안에 웨이포인트 1에 도달하는" 행동 열을 찾고, 그중 앞의 4개만 실행된다.
- Stage 2로 넘어가는 시점은 고정이다: Stage 1의 470 스텝이 끝나면 넘어간다. `final_trans_norm_cutoff`(yaml `:182`,
  `d4rl/enums.py:20`)는 이름과 달리 **코드 어디에서도 읽지 않는다**(전체 grep 결과 enums 정의뿐).
  `error_threshold`(`d4rl/enums.py:22`)도 마찬가지다.

## 4. 상위 레벨 행동 표현과 시간 길이 정보

### 4.1 MPPI가 샘플링하는 것: 8차원 잠재 행동 z
- L2 MPPI의 행동 차원은 `predictor.action_dim = z_dim = 8`이다(CPU 확인; `hjepa.level2.action_dim: 0`, yaml `:110`,
  `z_dim: 8`, yaml `:95`). `latent_actions = True` (`mpc.py:99`, `mppi_planner.py:292`).
- 계획 중 L2 예측기 호출은 `forward_multiple(state, actions=None, latents=z)`다(`mppi_planner.py:123-131`).
  z_dim>0이면 `prior_model`이 항상 만들어지고(`pldm/models/predictors/sequence_predictor.py:44-50`), 이 경로에서는
  `latents[i]`가 그대로 예측기 행동 입력이 된다(`sequence_predictor.py:233-258, 290-291, 312-320`).
  prior는 계산은 되지만 값은 쓰이지 않는다(`:239-242`).
- **웨이포인트 자체를 샘플링하지 않는다.** 웨이포인트는 샘플된 z 열을 L2 예측기로 굴린 결과(`pred_obs`)다.

### 4.2 학습 때 z가 만들어지는 방식과 시간 길이 정보
- 학습 시 z = posterior(행동 덩어리)다. posterior 입력은 **10 스텝 × 2 = 20차원 행동 열을 flatten한 것**이다
  (`posterior_input_type: 'actions'`, `posterior_input_dim: 20`, yaml `:102-104`; `sequence_predictor.py:269-277`).
- 고정 간격 학습: 61프레임 창을 `l2_step_skip=10`으로 자르고 행동을 10개씩 나눈다(`pldm_envs/diverse_maze/d4rl.py:198-209`).
  모든 구간이 10 스텝이므로 구간 길이 자체가 상수이고, 담을 정보가 없다.
- 적응형 학습(우리 수정): 경계 사이 가변 길이 구간의 행동을 **선형 보간으로 길이 10으로 리샘플링**한다
  (`adaptive_waypoints/d4rl_adaptive.py:72-78`, `segmentation.py:77-95`, `F.interpolate(mode="linear", align_corners=True)`).
  - 스텝당 행동 크기는 보간으로 유지되고 구간 길이 L에 대한 별도 채널은 없다. 그래서 **L 정보는 posterior 입력에서 명시적으로 사라진다**.
    보간 때문에 생기는 모양 왜곡에 간접적으로 남을 수는 있지만, 이를 L로 쓰도록 설계된 경로는 없다.
  - 반면 예측 목표는 L 스텝 뒤의 경계 상태다(`d4rl_adaptive.py:64-66`). 따라서 L2 예측기는 "길이를 모르는 z → 실제로는 L 스텝 뒤 상태"를
    학습한다. 구간 길이의 변동은 z 쪽에서 평균되거나 노이즈로 흡수된다.
- 추론 때: z를 직접 샘플하므로 "이 웨이포인트까지 몇 스텝인가"라는 양은 모델 어디에도 없다. 한편 L1에는 항상 10 스텝을 준다(§3).
  **고정 간격 모델에서는 이 가정이 학습과 일치하지만, 적응형 모델에서는 일치하지 않는다**. 적응형 학습의 구간 길이는 min_seg=8 이상에서 가변이다.

### 4.3 z의 제한 범위 (추론 전용)
- `clamp_actions = latent_actions or config.clamp_actions` = True (`mpc.py:135-142`).
  `normalizer.l2_latent_min/max_bounds`가 있으면 차원별로 **[p₂ + 0.1, p₉₈ − 0.1]**로 clamp한다(`pldm/planning/utils.py:55-72`).
  경계가 없으면 [min_step, max_step] = [−2.5, 2.5]다(yaml `:218-219`, `utils.py:101-102`).
- 경계는 `compute_l2_latent_bounds`가 **학습 데이터셋의 첫 100 배치**에서 posterior 평균의 백분위로 계산한다
  (`pldm/train.py:635-650`, `pldm_envs/utils/normalizer.py:448-560`; `l2_latent_bounds_percentile: 2.0`, yaml `:145`).
  - 주의: 평가 커널은 `adaptive_min_seg`를 지정하지 않으므로 데이터셋은 **고정 간격 `D4RLDataset`**이다(`pldm/data/dataset_factory.py:17-21`).
    따라서 적응형으로 학습한 체크포인트도 **고정 간격 행동 덩어리로 계산한 z 경계** 안에서 계획된다.
- CPU 추정(부록 B; r50 main에서 고정 간격 덩어리 30,000개 사용, 실제 평가의 100 배치와 표본만 다름):
  - 경계 폭: 차원별로 대략 ±0.2~±2.4.
  - N(0, 10·I) 노이즈를 이 경계에 넣으면 **차원별 약 63%가 clamp된다**. 8차원 벡터 중 1개 이상 clamp되는 비율은 99.98%, 평균 5개 차원이 경계값에 붙는다
    (실제로는 U + ε를 clamp하므로 U 위치에 따라 달라진다).
    즉 L2 MPPI 샘플은 대부분 경계 상자의 면이나 모서리에 있다.
- `z_reg_coeff=0.1`(yaml `:231`)로 계산되는 `z_reg`는 **총비용에 더해지지 않는다**(`mppi_torch.py:356-362`에서 계산 후 사용하지 않음).
- L1 행동 제한: `min_step=0, max_step=1, clamp_actions=False`(yaml `:199-200`)이므로 정규화 공간에서 **행동 벡터의 노름을 1 이하로 재스케일**한다
  (`utils.py:105-118`). N(0, 5·I) 샘플의 약 90%가 노름 1로 잘린다(부록 B). 노름 1은 환경 단위로 약 0.41이다
  (`action_std≈0.41`, `normalizer.py:29-30`).

## 5. 각 레벨의 비용 함수

MPPI가 쓰는 비용은 objective의 `__call__`이 아니라 `RunningCost`다(`mppi_planner.py:208-270`, `:339-347`).
그래서 `loss_coeff_first/last`(`objectives_v2.py:104-116`)와 `sum_last_n` 가중은 MPPI 경로에 **적용되지 않는다**.
MPPI는 `sum_all_diffs`와 `sum_last_n`으로 시작 인덱스만 정한다(`mppi_torch.py:288-296, 324-325`).

| | L2 | L1 (Stage 1) | L1 (Stage 2, 평면) |
|---|---|---|---|
| 예측 쪽 | L2 예측기 롤아웃의 `obs_component` (K,16,43,43) | L1 예측기 롤아웃의 `obs_component` → `pred_encoder`(=L2 IdentityEncoder) 통과. 16채널만 들어가 proprio 슬라이스가 0채널이 되므로 결과 = obs 그대로 (CPU 확인) | L1 롤아웃 `obs_component` |
| 목표 쪽 | 목표 이미지(속도 0으로 렌더링, `wrappers.py:260-281`)의 L1 `obs_component` → L2 Identity → flatten (29584) (`mpc.py:328-341`) | L2 `pred_obs[1]` flatten (29584) (`two_lvl_planner.py:73-75`) | 목표 이미지의 L1 `obs_component` flatten (29584) (`mpc.py:309-316`) |
| 공간 | L1 인코더의 obs 잠재 공간 (16×43×43). proprio 제외(`cost_entity: obs_component`, 기본값, `planners/enums.py:128`) | 같음 | 같음 |
| 스텝 비용 c_t | mean_d (pred_t − goal)² (`mppi_planner.py:260-270`) | mean_d (pred_t − waypoint₁)² | mean_d (pred_t − goal)² |
| 시간 합산 | `sum_all_diffs=true` → t=1..T_L2 **모든 스텝 합** (yaml `:223`) | `sum_all_diffs=true` → **10 스텝 모두** 같은 웨이포인트 1과 비교해 합산 (yaml `:202`) | 50 스텝 모두 합 |
| 추가 항 | 섭동 비용 Σ_t U_t·λΣ⁻¹ε_t (`mppi_torch.py:391-402`). `z_reg`는 미적용 | 같음 | 같음 |
| 기타 | `projected_cost=false` → prober로 위치 공간에 사영하지 않음. `cost_dim_range` 전체 | 같음 | 같음 |

함의: 모든 스텝을 합산하므로 L2는 "목표에 일찍 가까워지는" 잠재 경로를, L1은 "웨이포인트 1에 일찍 도달해 머무는" 행동 열을 선호한다.
L1 비용에는 "10 스텝째에 도착하라"는 시간 구조가 없다.

## 6. 종료 조건

- **목표 도달**: `CustomMazeEnv._is_goal_reached`: ‖현재 xy − 목표 xy‖ < **0.5** (환경 좌표)이면 보상 1
  (`pldm_envs/diverse_maze/ant_draw.py:38-60`).
  - `ActionRepeatWrapper`가 마지막 서브스텝의 결과만 반환하므로(`wrappers.py:204-219`), **4번째 서브스텝 시점에 반경 안에 있어야** 보상이 기록된다.
    1~3번째 서브스텝에서 반경을 지나쳐 나가면 기록되지 않는다.
- **성공 판정과 스텝 수**: Stage 1+2의 보상 기록(485개)에서 보상이 처음 나온 인덱스가 `terminations[b]`가 되고,
  그 값이 T=485보다 작으면 성공이다(`hmpc.py:84-96`). 평균 스텝 수는 성공 시행의 그 인덱스 평균이다(`pldm/planning/utils.py:7-25`).
  단위는 MPC 스텝(0부터 시작)이다.
- **조기 종료 없음**: 루프는 `done`이나 성공 여부를 보지 않고 Stage 1 470 스텝 + Stage 2 15 스텝을 끝까지 돈다(`mpc.py:440-600`).
  성공 뒤에도 계속 행동하지만, 첫 보상 인덱스만 쓰므로 판정은 바뀌지 않는다.
- **실패 확정**: 485 스텝이 끝날 때까지 보상 1이 한 번도 없으면 실패다. 중간에 실패를 판정하는 조건은 없다.
- 참고: `load_environment`의 `max_episode_steps=600`(`ant_draw.py:88`)은 TimeLimit 래퍼를 벗겨 내서 쓰이지 않는다(`ant_draw.py:71, 90`).

## 7. 설정 가능한 값 목록

### 7.1 config로 바꿀 수 있는 값

| 결정 | 설정 키 (`eval_cfg.h_d4rl_planning.` 아래) | 현재 값 | 정의 / 사용 위치 |
|---|---|---|---|
| 재계획 주기 (L1·L2 공통) | `replan_every` | 4 | yaml `:193`; `mpc.py:441, 526` |
| 총 스텝 | `hard.n_steps` | 500 (Kaggle override) | yaml `:266`; `mpc.py:435` |
| Stage 2 길이 | `final_trans_steps` | 15 | yaml `:183`; `mpc.py:239, 434` |
| L2 최대 계획 길이 (Stage 1 길이도 결정) | `hard.max_plan_length_l2` | 47 | yaml `:269`; `evaluator.py:269, 294-297`; `mpc.py:431` |
| L2 최소 계획 길이 | `level2.min_plan_length` | 3 | yaml `:220`; `mpc.py:470` |
| L1 계획 길이 (Stage 2 평면 L1에만 적용) | `hard.max_plan_length` | 50 | yaml `:268`; `mpc.py:473` |
| L2 샘플 수 | `hard.level2.mppi.num_samples` | 200 (Kaggle; yaml은 4000) | yaml `:280`; `mppi_planner.py:363` |
| L1 샘플 수 | `hard.level1.mppi.num_samples` | 1000 | yaml `:277` |
| 노이즈 공분산 (L2 / L1) | `level{2,1}.mppi.noise_sigma` | 10 / 5 (분산) | yaml `:228, :212`; `mppi_planner.py:329-335` |
| 온도 λ | `level{1,2}.mppi.lambda_` | 0.0025 / 0.0025 | yaml `:214, :230` |
| 비용 시간 합산 방식 | `level{1,2}.sum_all_diffs`, `level2.sum_last_n` | true / true, 3 | yaml `:202, :223, :226`; `mppi_torch.py:288-296` |
| 비용 공간 | `level{1,2}.cost_entity`, `cost_dim_range`, `projected_cost` | obs_component, 전체, false | `planners/enums.py:127-131`; `mppi_planner.py:313-322, 343-344` |
| L1 행동 크기 제한 | `level1.min_step / max_step / clamp_actions` | 0 / 1 / false | yaml `:199-200`; `utils.py:105-118` |
| L2 z 범위 | `eval_cfg.l2_latent_bounds_percentile`, `level2.min_step/max_step` | 2.0, ±2.5(경계가 없을 때만) | yaml `:145, :218-219`; `train.py:639`; `utils.py:59-72` |
| L2 노이즈 평균/분산을 데이터 통계로 | `eval_cfg.l2_use_latent_mean_std` | false | yaml `:146`; `mppi_planner.py:309-326` |
| 행동 반복 | `action_repeat`, `action_repeat_mode` | 4, id | yaml `:190-191` (모델이 4로 학습되어 사실상 고정) |
| 시드 | `seed` (최상위) | 246 → 평가 시드 263 | yaml `:287`; `train.py:630` |
| chunk 크기 | `n_envs_batch_size` | 20 | yaml `:185`; `mpc.py:54-69` (§8: 시행별 결과에 영향) |
| 오차 적응형 L1 자원 배분 (§7 실험) | `error_adaptive_l1*` | off | `d4rl/enums.py:28-33`; `hmpc.py:199-280` |

### 7.2 코드를 고쳐야만 바꿀 수 있는 값

| 결정 | 현재 동작 | 위치 |
|---|---|---|
| L1 목표 웨이포인트 인덱스 | 항상 `pred_obs[1]` | `two_lvl_planner.py:74` |
| 계층 모드에서 L1 계획 길이 | `l2_step_skip` = 10 고정 (모델 config `step_skip`에서 옴. 바꾸면 L2 계획 길이 계산도 바뀌고, 학습된 L2 간격과도 어긋남) | `two_lvl_planner.py:80`; `mpc.py:79-83, 430-431` |
| L2만 다른 주기로 재계획, 또는 조건부 재계획 | 불가. 매 재계획마다 둘 다 다시 계산 | `mpc.py:441, 476-487`; `two_lvl_planner.py:51-83` |
| 웨이포인트 도달 판정 / 전환 로직 | 없음 | `two_lvl_planner.py` 전체 |
| 재계획당 MPPI 갱신 횟수 | 1회 (`num_refinement_steps`는 저장만 되고 사용되지 않음) | `mppi_planner.py:381, 450-459` |
| warm-start 이동 규칙 | `plan_size`가 줄 때만 당김 | `mppi_planner.py:444-446` |
| 실행 행동 선택 | 가중 평균 nominal `U` (최선 샘플 아님) | `mppi_torch.py:223-232` |
| L1 비용의 시간 가중 | 모든 스텝 동일 가중 합 (`loss_coeff_*`는 MPPI 경로에서 무시) | `mppi_torch.py:288-325`; `mppi_planner.py:208-270` |
| `z_reg` 적용 | 계산만 하고 비용에 더하지 않음 | `mppi_torch.py:356-370` |
| Stage 1 → 2 전환 조건 | 스텝 수 고정 (`final_trans_norm_cutoff` 미사용) | `mpc.py:239-244, 434-436` |
| 목표 도달 반경 | 0.5 | `ant_draw.py:51` |
| 보상 판정 시점 | 행동 반복의 마지막 서브스텝 | `wrappers.py:204-219` |
| 조기 종료 | 없음 | `mpc.py:440` |
| 환경별 MPPI 순차 실행 | 환경마다 Python 루프 | `mppi_planner.py:443-459` |

## 8. 무작위성의 원천

| 원천 | 설명 | 위치 | 시드로 고정되는가 |
|---|---|---|---|
| MPPI 노이즈 | 매 `command`마다 (K,T,nu) 샘플 | `mppi_torch.py:376` | 전역 torch RNG를 쓰므로 시작 상태는 고정됨 (아래 참고) |
| MPPI 초기 `U` | 제어기 생성 시 노이즈 분포에서 샘플. chunk마다 L1·L2·평면 L1 제어기를 새로 생성하고, `HierarchicalD4RLMPCEvaluator.__init__`도 쓰지 않는 h_planner를 하나 만들어 RNG를 소비 | `mppi_torch.py:119-120`; `mpc.py:175-178`; `hmpc.py:48` | 같음 |
| **환경 순서 의존성** | 환경마다 순차로 같은 RNG 스트림에서 노이즈를 뽑으므로, 시행 j의 노이즈는 같은 chunk에서 앞선 시행들의 소비량에 따라 달라짐. **chunk 구성, chunk 크기, 순서가 바뀌면 같은 시행도 다른 노이즈를 받음** | `mppi_planner.py:443-459`; `mpc.py:54-69, 173-175` | 구성이 같을 때만 재현 가능 |
| 평가 시드 | `fixed_rng(seed + epoch·1009 + 17)`. `eval_only`에서는 epoch=0이므로 263 | `train.py:69-83, 630-631, 231` | 예 |
| 계획 전에 소비되는 RNG | L2 z 경계 계산(데이터셋 100 배치), `eval_on_objectives`, probing 학습이 계획보다 먼저 같은 RNG를 소비 | `train.py:635-653`; `evaluator.py:371` | 같은 데이터와 설정이면 예 |
| L2 z 경계 | 위 100 배치(무작위 창)에서 계산하므로 데이터셋, RNG, 체크포인트에 따라 달라짐 | `normalizer.py:448-560` | 예 (조건 같음) |
| GPU 비결정성 | `seed_everything`이 cudnn deterministic, `CUBLAS_WORKSPACE_CONFIG`는 설정하지만 `torch.use_deterministic_algorithms(True)`는 설정하지 않음. Kaggle GPU 기종(P100/T4)이 달라지면 부동소수 결과도 달라짐. λ=0.0025의 지수 가중이 작은 차이를 증폭 | `train.py:45-58` | **아니오** (보장 안 됨) |
| 환경 초기화 | `load_environment`의 `action_space.sample()`과 d4rl reset 노이즈가 있으나, 직후 `set_state(qpos=start, qvel=0)`로 덮어씀 | `ant_draw.py:108-113`; `maze2d_envs_generator.py:110-116` | 영향 없음 (결정적) |
| 시작/목표 | `set_start_target_path` 파일에서 읽음. 생성기 시드 42는 새로 샘플할 때만 쓰임 | `maze2d_envs_generator.py:67-116`; yaml `:186` | 영향 없음 |
| `random_actions` | false | `mpc.py:531-535` | 해당 없음 |

정리: 같은 코드, 체크포인트, 데이터, chunk 구성, GPU 기종이면 재현될 가능성이 높지만 보장되지는 않는다.
§9에서 같은 체크포인트와 같은 120개 인스턴스로 다시 평가한 결과가 106/120으로, 예전 기록 110/120과 달랐다.
그때 chunk 구성이 40개 + 30/30/20으로 이번의 20×6과 달랐으므로, 위 "환경 순서 의존성"만으로도 이 차이를 설명할 수 있다.
시행 단위로 재현하려면 코드 수정이 필요하다: 환경과 재계획마다 `(seed, trial_id, step)`으로 생성기를 따로 만들어 MPPI에 넘기고,
`torch.use_deterministic_algorithms(True)`를 설정한다.

---

## 9. 추론 때 바꿀 수 있는 결정 지점

| 결정 지점 | 지금 고정된 값 | 적응형으로 바꾸려면 | 재학습 필요? |
|---|---|---|---|
| **재계획 시점** | 4 MPC 스텝마다, L1·L2 동시 (`replan_every=4`) | (a) 주기 자체 변경: config만. 단 L1 계획이 10 스텝이므로 `replan_every ≤ 10`이어야 함(`mpc.py:526`의 `actions[:, i % replan_every:]`가 비게 됨). (b) 조건부 재계획(예: L1 비용이나 예측 오차가 임계값을 넘을 때, 웨이포인트 도달 시): `mpc.py:441`의 조건을 함수로 바꾸는 **코드 수정**. (c) L2와 L1 주기 분리: `TwoLvlPlanner`가 직전 `l2_result`를 저장했다가 재사용하도록 **코드 수정** | 아니오 |
| **목표 전환 시점** | 매 재계획마다 새 L2 계획의 `pred_obs[1]`. 도달 판정 없음 | 웨이포인트를 유지하다가 도달하면 다음(`pred_obs[k+1]`)으로 넘기기, 또는 L2를 다시 부르기: `two_lvl_planner.py:73-75`를 상태를 갖는 로직으로 바꾸는 **코드 수정**. 도달 기준은 (i) obs 잠재 MSE 임계값(L1 비용과 같은 척도, 보정 필요) 또는 (ii) 위치 prober(평가 중 학습되는 `probers["locations"]`, `evaluator.py:332`)로 판정 | 아니오 |
| **웨이포인트 거리 (시간 간격)** | L2 한 스텝 = 학습된 점프 간격(고정 간격 모델은 10, 적응형 모델은 가변이지만 z가 그 길이를 모름). L1에게는 항상 10 스텝 | (a) L1에게 주는 지평만 바꾸기: `two_lvl_planner.py:80`의 `plan_size`를 바꾸는 **코드 수정**. (b) 재학습 없이 대리 추정: 웨이포인트마다 L1 롤아웃 중 예측 obs가 웨이포인트에 가장 가까워지는 스텝을 "필요 스텝 수"로 추정해 L1 지평이나 전환 시점에 쓰기 (**코드 수정**, 계산량 증가). (c) L2가 스스로 간격을 고르게 하기: 지금 z에는 길이 정보가 없으므로(§4.2) 구간 길이를 posterior/예측기의 입력이나 출력으로 추가해야 함. 리샘플링(`d4rl_adaptive.py:72-78`)을 바꾸고 L2를 **다시 학습**해야 함 | (a)(b) 아니오, (c) **예** |
| **계산량** | L2 K=200, L1 K=1000, 재계획당 MPPI 1회 갱신, L2 지평 최대 47 | (a) 샘플 수, 지평, 노이즈, λ: config(난이도별). (b) 재계획당 갱신 횟수(반복 MPPI): `mppi_planner.py:450-459`에 반복을 넣는 **코드 수정**. (c) 스텝마다 적응형으로 배분: L1은 §7 실험의 훅(`mpc.py:549-556` `post_l1_step_hook`, `hmpc.py:199-280`)이 이미 있어 그대로 확장 가능. L2용 훅은 없으므로 **코드 수정** 필요. 참고: L2 샘플의 약 63%(차원별)가 경계에서 clamp되므로(§4.3), `noise_sigma`나 경계를 조정하면 같은 K로도 탐색이 달라질 수 있음 | 아니오 |

추가로 짚어 둘 점 (적응형 경계 연구와 직접 관련된 불일치):
1. 적응형으로 학습한 체크포인트도 추론 때는 L1에게 10 스텝 지평을 준다(§3).
2. 적응형으로 학습한 체크포인트의 z 경계도 고정 간격 데이터로 계산된다(§4.3).
3. 재계획이 4 스텝마다 일어나므로, 학습된 L2 간격(10 또는 가변)과 상관없이 실제로 실행되는 것은 언제나 "새 계획의 첫 웨이포인트를 향한 4 스텝"이다.
   그래서 학습 데이터의 경계 배치가 바뀌어도 추론 경로에서 그 차이가 드러나는 통로는 L2 예측기의 1스텝 점프 하나뿐이다.
   §9에서 dp_segmentation 캐시와 signal-1 캐시의 결과가 거의 같았던 것과도 들어맞지만, 이것만으로 원인이라고 단정할 수는 없다.

---

## 부록 A. 현재 hard 평가의 실효 설정 (병합 후)

병합 규칙: 난이도별 `level1/level2`에서 기본값과 다른 필드만 덮어쓴다(`evaluator.py:92-126, 267-313`).

| 항목 | 값 |
|---|---|
| n_steps / Stage 1 / Stage 2 / 총 | 500 / 470 / 15 / 485 |
| replan_every | 4 |
| L2: K, noise 분산, λ, 지평 | 200, 10, 0.0025, min(47, (470−i+9)//10), 최소 3 |
| L1 (Stage 1): K, noise 분산, λ, 지평 | 1000, 5, 0.0025, 10 |
| L1 (Stage 2): 지평 | 50 |
| L2 z 경계 | 2/98 백분위 ± 0.1 마진 |
| 목표 반경 | 0.5 |
| 평가 시드 | 263 (= 246 + 0·1009 + 17) |

## 부록 B. CPU 확인 스크립트 (GPU 미사용)

`/tmp/.../scratchpad/shape_probe.py`와 `sat.py`(세션 임시 파일, 저장소에 넣지 않음)로 확인한 내용:
- `pretrained_baseline.ckpt`를 CPU에 로드해 L1·L2 인코더와 예측기의 텐서 모양을 확인하고, L2 Identity 인코더의 항등성(`torch.equal`)을 확인했다.
- `RunningCost`에 L1 obs 롤아웃(K,16,43,43)을 넣어 proprio 슬라이스가 0채널이 되고 출력이 obs와 같음을 확인했다.
- r50 main `data.p`에서 무작위 61스텝 창 4개씩, 총 30,000개의 10스텝 행동 덩어리를 뽑아 posterior 평균의 2/98 백분위 경계를 추정했다
  (baseline, fixed, adaptive 체크포인트 각각). N(0, 10·I) 샘플의 clamp 비율은 차원별 62~64%다. L1에서는 N(0, 5·I) 샘플의 90.5%가 노름 1을 넘었다.
- 실제 평가의 경계는 학습 데이터셋 100 배치에서 계산하므로 수치는 약간 다를 수 있다.
