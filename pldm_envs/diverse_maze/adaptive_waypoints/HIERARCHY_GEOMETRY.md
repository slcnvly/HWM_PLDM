# 계층 world model의 목표 비용 공간 반경: 결과

사전 등록: `HIERARCHY_GEOMETRY_PREREG.md` (커밋 `91ef2e4`). 코드와 결과는 `latent_law/hg_*.py`, `latent_law/results_hg_*.json`에 있다. CPU만 썼다.

## 1. 코드 확인 — **"주장 후보 성립 가능"**

**(a) L2 플래닝의 목표 비용 공간 = L1 인코더의 obs latent (16×43×43) 그 자체다. 투영은 없다.**
- 목표: 목표 이미지를 L1 인코더에 통과시킨 뒤(`pldm/planning/mpc.py:309-356`), 그 obs와 proprio를 L2 인코더에 넣는다(`mpc.py:382`).
  그 `obs_component`(`mpc.py:384`)를 펼쳐 목표로 쓴다(`mpc.py:389-395`).
- 현재 상태: `TwoLvlPlanner.plan`이 L1 인코더 출력(`BackboneOutput`)을 그대로 L2 플래너에 넘긴다(`two_lvl_planner.py:48-56`, `repr_input=True`). L2 인코더를 거치지 않는다.
- 비용: L2 롤아웃의 `ensemble_obs_component`(`mppi_torch.py:329`)와 목표의 차이를 `RunningCost`가 평균 제곱으로 잰다(`mppi_planner.py:260-270`).
  `projected_cost=false`라 prober 투영이 없다(`mppi_planner.py:343-344`). 시간 합산은 `sum_all_diffs=true`다(yaml `:223`).
- **수치 확인** (사전학습 체크포인트, r50 프레임 200개):
  - L2 인코더 = `IdentityEncoder`이고 학습 파라미터는 **0개**다.
  - L2 인코더 출력의 `obs_component`와 `encodings`는 L1 것과 **`torch.equal`**이다.
  - L2 예측기 출력 `obs_component`의 모양은 (T+1, B, 16, 43, 43)로 L1 obs latent와 같은 공간이다.

**(b) L2 예측기의 상태와 지평**
- 입력·출력 상태: L1 인코딩(obs 16채널 + proprio 2채널, 18×43×43)이다. `prefused_repr=True`라서 예측기 안에서 obs와 proprio로 나눈다.
- 행동: 8차원 잠재 z다(`z_dim: 8`, yaml `:95`). 학습 때는 posterior가 10스텝 행동 덩어리(20차원)로부터 z를 만든다(`sequence_predictor.py:269-277`).
- 학습: 동결된 L1 인코더로 `l2_step_skip = 10` 간격 프레임 7장을 인코딩한다(`hjepa.py:159-187`, `d4rl.py:192-209`).
  L2 예측기가 **6 L2 스텝**을 teacher forcing 없이 굴린다(`jepa.py:240-250`, `T = input_states.shape[0] − 1`, `l2_n_steps: 6`, yaml `:4`).
  손실은 PredictionObs + PredictionProprio다(yaml `:113-115`).
- **L2 한 스텝 = L1 10스텝**, L2 학습 rollout = **L1 60스텝**이다.
  실제 이동량(r50, 3단계): 10스텝은 xy 중앙값 0.55칸(BFS 중앙값 0칸, 90% 1칸), 60스텝은 xy 중앙값 **1.71칸**(90% 3.61칸, BFS 중앙값 **2칸**, 90% 4칸).
- 계획 지평: hard 평가에서 L2 최대 47스텝 = L1 470스텝(INFERENCE_LOOP §2). V0 성공 시행의 평균 속도(0.111칸/스텝)로 환산하면 약 52칸이다.

**(c) L2 전용 표현: 없다.** L2 인코더는 항등이다(`encoders.py:424-455`, yaml `:79` `arch: identity_encoder`).
L2에서 학습되는 모듈은 예측기뿐이다(z용 prior, posterior 포함 182,334개). 상태 표현은 L1에서 물려받는다.

→ 판정 규칙에 따라 **"주장 후보 성립 가능"**이다. L2 전용 표현이 없으므로 2단계는 L1 obs latent 반경만 측정한다.

## 2. L2가 쓰는 표현의 포화 반경

L2 비용 공간은 L1 obs latent와 같은 텐서이므로(1단계에서 `torch.equal` 확인), 반경은 **정의상** LATENT_LAW A2의 학습된 L1 값과 같다.
같은 프레임과 같은 쌍에서 L2 인코더를 거친 표현이 L1과 비트 단위로 같으므로, 재계산은 그 동일성 확인으로 대신했다.

| | R_xy (칸) | R_maze (칸) | 비교 대상 |
|---|---|---|---|
| L2 비용 공간 (= L1 obs latent, 튐 제외) | **1.35** | **2** | |
| L2 학습 rollout 이동량 (L1 60스텝, r50) | xy 중앙값 1.71, 90% 3.61 | BFS 중앙값 2, 90% 4 | P2c |
| 평가 과제 시작 거리 (V0 hard) | | 중앙값 13 (7–17) | P2b |
| L2 계획 지평 (47 L2 스텝 ≈ 52칸 환산, 상한은 과제 거리 13칸) | | 13 | P2d |

**사전 등록 예측 판정**:
- **P2a**: R_xy ∈ [1.08, 1.69] → **맞음** (1.35, 정의상 L1과 같음).
- **P2b**: R_maze ≤ 3.25 → **맞음** (2).
- **P2c** (보조): R_xy < L2 학습 rollout 이동량 xy 중앙값 → **맞음** (1.35 < 1.71). 다만 차이는 작다(0.79배).
- **P2d**: R_maze ≤ 0.25 × 13 = 3.25 → **맞음** (2).

→ **주장 후보: 지지** (P2a, P2b, P2d 모두 맞음).

해석:
- **L2는 예측 지평을 L1의 10배(10 L1 스텝/L2 스텝, 계획 최대 470 L1 스텝)로 늘리지만, 목표 비용은 L1 obs latent에서 재므로 거리 반경(약 2칸)은 그대로다.**
  L2에는 상태 표현을 새로 학습하는 부분이 없으므로(항등 인코더), 계층 구조가 "목표까지 먼 거리"를 재는 능력은 더해지지 않는다.
- P2c의 차이가 작다는 점은 중요하다. L2 학습 rollout(60스텝)도 무작위 데이터에서는 2칸 남짓만 움직인다(√n 확산, 3단계).
  그래서 L2 표현을 따로 학습했더라도 r50 데이터로는 반경이 크게 넓어지기 어렵다. 한계는 구조와 데이터 양쪽에 있다.
