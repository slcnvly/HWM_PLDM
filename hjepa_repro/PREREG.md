# PREREG: H-JEPA 3단, 최상위 계획 길이(level3.horizon) sweep 사전 등록

작성: 2026-10-11 (UTC), 본 측정 시작 전. 이 파일과 `prereg.json`은 커밋 뒤 수정하지 않는다.

## 질문

최상위 레벨의 계획 길이(macro-action 개수)를 과제마다 다르게 골랐다면, 모든 과제에 고정값 하나를 쓸 때보다
성공률이 높아지는가. 그 차이가 선택 편향과 구별될 만큼 큰가.

## 확정값

| 항목 | 값 |
|---|---|
| 모델 | H-JEPA 3단, 체크포인트 `ant/ant_hjepa_l3/seed42/ant_hjepa_l3_object.ckpt` (고정) |
| 설정 | `config/eval/ant_l3.yaml`. `hierarchical_plan_config.level3.horizon`만 바꾼다 |
| 후보 (S = 3) | level3.horizon = 5 (기본), 3, 7. 이 순서가 우선순위다 |
| 과제 (I = 50) | 공식 과제 `assets/eval_trajs/ant/expert_grid_d3_n50.pt` 50개 전부 |
| 반복 (R = 6) | planner seed 100, 101, 102, 103, 104, 105 |
| A (선택용) | seed 100, 101, 102 |
| B (평가용) | seed 103, 104, 105 |
| 저장소 | kevinghst/H-JEPA 커밋 `97054ec`, 코드 수정 없음 |

실행 명령 (`h_jepa/` 안에서):

```bash
python eval.py --config-name ant_l3 seed=<planner_seed> \
  hierarchical_plan_config.level3.horizon=<H> \
  load_eval_trajs_path=assets/eval_trajs/ant/expert_grid_d3_n50.pt eval.num_eval=50 \
  policy=$HJEPA_HOME/ckpts/ant/ant_hjepa_l3/seed42/ant_hjepa_l3_object.ckpt \
  output.dir=$HJEPA_HOME/ckpts/ant/ant_hjepa_l3/seed42/sweep/h<H>_s<planner_seed>
```

horizon 외의 planner 설정(표본 수, 갱신 횟수, 학습률, receding_horizon, 하위 레벨 horizon)은 바꾸지 않는다.

## 발동한 자동 결정 규칙

1. **과제 집합: 공식 50개.** 추가 100개 생성(4단계)을 시도했으나 실패했다.
   OGBench expert 정책 파일을 읽는 단계에서
   `TypeError: ShapedArray.__new__() got an unexpected keyword argument 'named_shape'`가 났다
   (`stable_worldmodel/envs/ogbench/sac_expert.py:305`, 설치된 jax와 정책 pickle의 버전 불일치).
   규칙대로 공식 50개를 쓴다.
2. **예산 축소: 규칙 ④ (S = 3, R = 6, I = 50).**
   예상 시간은 Σ t × (H / 5)이다. 가장 작은 허용 구성(S = 3, R = 6, I = 50)의 합은 900 t다.
   이것이 9시간 안에 들어오려면 t ≤ 36초여야 한다.
   3단계 평가가 아직 끝나지 않아 t의 50과제 평균은 없다. 같은 명령으로 잰 과제 1개의 실측값은
   177.9초다 (2026-10-10 스모크 테스트, T4, `ant_l3`, seed 42). 이 값이면 900 t = 44.5시간이다.
   t가 36초보다 크기만 하면 ①~③을 끝까지 적용해도 9시간을 넘으므로 결과는 같은 구성이다.
   3단계가 끝나면 그 평균 t를 보고서에 적고, 36초보다 큰지 확인한다. 36초 이하로 나오면 이 사전 등록의
   전제가 틀린 것이므로 그 사실을 보고서 맨 앞에 적는다 (구성은 바꾸지 않는다).
3. **실행 환경과 "시간 허용 범위".** 이 컴퓨터에는 GPU가 없어 Kaggle(Tesla T4 2장, 세션당 4개 프로세스 병렬)에서 돌린다.
   "시간 허용 범위"는 Kaggle 주간 GPU 할당량(30시간, 2026-10-17 00:00 UTC 초기화)의 남은 양으로 본다.
   세션 시간 제한(12시간) 때문에 여러 세션으로 나눠 이어 돌린다 (`eval.py`가 끝난 과제 파일을 건너뛴다).

## 실행 순서

- planner seed 바깥, horizon 안쪽: seed 100의 H = 5, 3, 7 → seed 101의 H = 5, 3, 7 → ...
- 세션을 둘로 나눈다: 세션 1이 seed 100, 101, 102, 세션 2가 seed 103, 104, 105를 같은 순서로 돈다.
  각 세션 안에서는 4개 프로세스가 이 순서대로 다음 작업을 가져간다.
- 과제 하나가 끝날 때마다 `chunks/tasks_<k>-<k+1>.json`이 생긴다. 여기서 과제별 결과를 읽어
  `sweep_raw.jsonl`에 `{task, horizon, seed, split, success, env_steps, wall_time}`으로 적는다.
  `env_steps`는 성공 시 `steps_to_success`, 실패 시 null이다 (평가 루프는 성공과 무관하게 350스텝을 돈다).
  `wall_time`은 그 과제의 `evaluation_time`이다. 병렬 실행 중에 잰 값이다.

## 분석 (`analysis/analyze.py`, 실행 스크립트와 별도 파일)

한 과제에서 "쓸 수 있는 seed"는 세 후보가 모두 끝난 seed다. A와 B 각각 쓸 수 있는 seed가 3개 이상인 과제만 분석한다.

1. **고정 최적 (B에서 평가)**: A에서 과제 평균 성공률이 가장 높은 horizon 하나 → 그 horizon의 B 성공률.
2. **과제별 최적 (B에서 평가)**: 과제마다 A 성공률이 가장 높은 horizon → 그 horizon의 그 과제 B 성공률 → 과제 평균.
   **격차 = (2) − (1). 주 결과다.**
3. **순진한 격차 (대조용)**: A와 B를 합친 데이터로 과제별 최적을 고르고 같은 데이터로 평가.
   (3)과 격차의 차이를 선택 편향의 크기로 적는다. 주 결과로 쓰지 않는다.
4. **귀무 분포**: 각 과제의 각 seed 안에서 세 후보의 결과를 무작위로 뒤바꾼 뒤 (1), (2), 격차를 다시 계산한다.
   1000회, 난수 seed 20261011. 관측 격차의 백분위를 적는다.

- 동점 규칙: A에서 과제별 최적이 동점이면 (1)에서 고른 전역 최적 horizon을 택한다. 그것이 동점 후보에 없으면
  우선순위가 높은 후보(5, 3, 7 순)를 택한다. (1)의 전역 최적이 동점이면 우선순위가 높은 후보를 택한다.
- 신뢰구간: 과제 단위 bootstrap 2000회(난수 seed 20261012), 95% 백분위 구간. (1), (2), 격차 각각.
  과제를 뽑으면 그 과제의 모든 시행을 함께 가져가고, 선택 절차 전체를 다시 한다.
- 계산량 교란: horizon별 과제 1개당 평균 벽시계 시간. 과제별 최적으로 선택된 horizon들의 평균 시간(B 시행)과
  고정 최적 horizon의 시간(B 시행)의 비.
- horizon별 표(A, B, 전체 성공률, 과제당 시간), 과제별 최적 horizon의 분포.

## 판정 (사후에 바꾸지 않는다)

- 관측 격차가 귀무 분포의 95 백분위 이하 → **격차 없음**.
- 95 백분위 초과, 격차 95% 구간 하한 > 3%포인트, 선택된 horizon의 평균 시간이 고정 최적의 1.5배 미만 → **격차 있음**.
- 그 외 → **판단 불가**. 숫자를 그대로 보고한다.
- 여섯 seed가 모두 끝난 과제가 하나도 없으면 분석할 수 없으므로 **판단 불가**로 적고 끝난 만큼의 표만 보고한다.
