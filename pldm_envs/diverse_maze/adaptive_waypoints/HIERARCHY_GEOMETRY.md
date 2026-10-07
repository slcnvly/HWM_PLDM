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

## 3. 확산 스케일링

(작성 중. n = 160용 200스텝 무작위 시뮬레이션 Kaggle CPU 커널이 끝나면 채운다.)

## 4. 오프라인 비용 비교: 거리 합 대 처음 닿는 시점

(작성 중. CPU 실행 중이다.)

## 5. 선행 연구

2026-10-07 기준으로 웹 검색을 했다. 출처는 arXiv 초록과 HTML 본문이다. 본문에서 직접 확인한 문장은 "확인"으로, 2차 요약에만 의존한 것은 "미확인"으로 표시했다.

| 연구 | 상위 계층 목표 비용 공간 | latent 거리의 유효 반경 vs rollout 길이 | multi-scale / 계층 quasimetric | 우리 주장과 겹치는 부분 | 다른 점 |
|---|---|---|---|---|---|
| **HWM** (Zhang, Terver 외, arXiv 2604.03208, 2026-04) — 우리가 쓰는 모델 | 모든 계층이 **하나의 공유 latent 공간**에서 예측한다. 상위 예측을 하위 목표로 "latent matching"한다(초록, 확인). 우리 코드에서는 L2 인코더가 항등이고 비용은 L1 obs latent MSE다(1절, 직접 확인) | 다루지 않음(초록 기준. 본문의 반경 분석 여부는 미확인) | 없음 | 우리 1절이 이 구조를 코드 수준에서 확인했다 | — |
| **H-JEPA** (Zhang, Terver, Rabbat, LeCun, Balestriero, arXiv 2610.06805, 2026-10-05) | **계층마다 자기 latent 공간**을 학습한다. 최상위는 ‖ẑ_H^(L) − g^(L)‖²를 최소화한다. HWM을 "encoders above level 1 are identity maps. All levels therefore predict and plan in the same latent space"로 기술한다(본문, 확인) | **정성적으로만** 다룬다. 그림(AntMaze 비용 지형) 설명: "level-1 distances are nearly constant away from the anchor, providing little signal to guide a gradient planner until it is close"(확인). 반경을 수치로 재거나 rollout 길이와 연결하지는 않는다(WebFetch 요약 기준, 미확인 가능성 있음) | 계층별 표현 공간이 사실상 multi-scale 거리 역할을 한다. quasimetric 제약은 없다(미확인) | **핵심 주장과 가장 많이 겹친다.** "공유 latent에서 L1 거리는 멀리서 평평하다"는 관찰과 "상위 계층 고유 표현이 성능을 올린다"(L1 동역학 + 상위 비용 투영 ablation, AntMaze. 정확한 수치는 2차 요약만 봐서 미확인)는 결과가 우리 주장 후보의 정성적 버전이다 | 우리 기여는 **정량화**에 있다. (i) 반경을 칸 단위로 쟀다(R_xy 1.35, R_maze 2). (ii) 무작위 초기화, 픽셀, DINOv2와 대조했다. (iii) L2 학습 rollout 이동량(1.71칸)과 비교했다. (iv) √n 확산으로 필요한 rollout 길이를 외삽했다(3절). (v) 반경을 직접 이용하는 비용(처음 닿는 시점)을 오프라인으로 시험했다(4절). 환경도 다르다(우리는 diverse maze, PLDM 데이터) |
| **TEMPO** (Moukheiber, Xue, Chen, arXiv 2610.04988, 2026-10) | 평면(비계층). LeWM, PLDM latent의 L2 + 학습된 시간 거리 F_φ의 혼합 비용 | latent 거리는 "agree within one plan of the goal but diverge several plans away"라고 기술한다. TwoRoom에서 목표가 3 플랜 떨어지면 LeWM이 88%→36%로 떨어진다(확인). 시간 거리 학습 쌍은 75스텝 이내다(확인) | 단일 시간 척도 | "latent 거리는 짧은 범위에서만 진행도와 일치한다"는 관찰이 같다 | 계층이 아니고, 반경을 칸 단위로 재지 않는다. 해결책은 별도 시간 거리 헤드다 |
| **Temporal-Distance-JEPA** (Bai, Xiong, arXiv 2607.25337, 2026-07) | 평면. LeWM 백본에 궤적에서 뽑은 방향성 시간 비용을 더한다. rollout-consistency H = 5(확인) | 포화 반경 서술은 찾지 못했다(NOT FOUND, 요약 기준). LeWM latent 거리와 진행도의 상관은 ρ = 0.65로 보고한다(확인) | 비대칭 시간 비용(quasimetric 계열, 세부 미확인) | latent 유클리드 거리가 "표현 학습의 부산물"이라 진행 비용으로 부적합하다는 문제 제기가 같다 | 계층이 아니다 |
| **Metro-WM** (Lee 외, arXiv 2609.31868, 2026-09) | 상위 계층이 latent 하위 목표를 생성하지 않는다. 대신 실제 관측 프레임 그래프에서 경로를 찾는다(초록, 확인) | 다루지 않음(초록 기준, 미확인) | 그래프가 먼 거리를 담당한다 | "JEPA 상위 계획기의 하위 목표가 실현 불가능하다"는 문제 제기가 우리 backward/벽 웨이포인트 관찰과 닿아 있다 | 반경 분석이 없다(미확인) |
| **Director** (Hafner 외, NeurIPS 2022) | 월드 모델 feature 공간이다. 목표는 goal autoencoder의 이산 코드를 디코딩한 feature다. worker 보상은 max-cosine 유사도다(확인) | 다루지 않음(미확인) | 없음 | 상위 목표 공간이 하위 상태 공간과 같은 feature 공간이라는 점 | RL 정책을 학습한다. 플래닝 비용이 아니다 |
| **HIQL** (Park, Ghosh, Eysenbach, Levine, NeurIPS 2023) | 가치 함수 V(s, φ(g))의 목표 표현 φ로 하위 목표를 표현한다(확인) | 가치 함수 자체가 시간 거리이므로 반경 문제를 학습으로 우회한다(해석) | 1단계 가치 함수를 두 계층이 공유 | 상위 계층이 "거리를 아는 표현"을 쓰면 먼 목표가 풀린다는 대조 사례 | 보상 없는 예측 학습이 아니라 IQL 가치 학습이다 |
| **QRL** (Wang 외, ICML 2023), **Horizon generalization** (Myers, Ji, Eysenbach, ICLR 2025), **TMD** (Myers 외, NeurIPS 2025), **MQE** (Zheng 외, arXiv 2511.07730) | 평면. 학습된 quasimetric 거리를 비용·가치로 쓴다 | Horizon generalization: 짧은 거리(d < c)에서 배운 것이 먼 거리로 일반화되는지를 형식화하고, quasimetric 구조가 이를 가능하게 한다고 주장한다(확인). 우리 latent는 이 성질이 없는 사례다(반경 2칸에서 평평해짐) | MQE는 여러 길이의 다단계 backup을 섞어 한 quasimetric을 배운다(요약 기준, 세부 미확인). QRL 세부는 이번에 다시 확인하지 않음(미확인) | "짧은 거리만 배우면 먼 거리는 구조(삼각 부등식)로 이어 붙여야 한다"는 논리가 우리 해석과 같다 | 계층 world model의 비용 공간을 분석하지 않는다 |
| **ALPS / Laplacian 표현** (Shehmar 외, ICML 2026, arXiv 2602.05031) | Laplacian 표현 공간에서 계층 플래닝 | Laplacian 표현이 "여러 시간 척도의 상태 공간 거리"를 담는다고 주장한다(초록, 확인). 반경 측정 여부는 미확인 | **multi-scale 거리 표현의 가장 가까운 사례** | 상위 계층에 맞는 거리 공간이 따로 필요하다는 방향 | 예측 world model이 아니다 |
| **SoRB, SPTM** (LATENT_LAW D 표 참조) | 그래프 최단 경로 | 짧은 간선만 믿는다는 설계가 반경 개념과 같다 | 그래프가 multi-scale 역할 | 같음 | 거리 학습 방식이 다름 |

**우리 주장과 겹치는 부분 정리**:
- "공유 latent를 쓰는 계층 WM(HWM)에서 상위 계층 목표 비용이 L1 latent에서 계산되고, 그 거리가 멀리서는 평평하다"는 **정성적 관찰은 H-JEPA(2026-10-05)가 이미 했다.**
  같은 저자군이 HWM의 후속으로 계층별 표현을 제안했다. 따라서 "계층 구조가 거리 반경을 늘리지 않는다"를 **새 발견으로 주장하기는 어렵다.**
- 우리가 새로 더할 수 있는 것은 다음이다(미확인 항목은 H-JEPA 본문 전체를 읽고 다시 확인해야 한다).
  - (i) 반경을 칸 단위로 정량화했다.
  - (ii) 반경이 학습 rollout 이동량과 같은 규모라는 연결(P2c, 1.35 < 1.71)과 √n 확산 외삽.
  - (iii) 인코더 대조(무작위, 픽셀, DINOv2).
  - (iv) 반경 안의 정보만 쓰는 비용의 오프라인 검증(4절).
- 반경과 학습 rollout 길이를 정량적으로 연결한 선행 연구는 이번 검색에서 **찾지 못했다**(없다는 확인은 아니다).
