# HWM_PLDM 원본 대비 추가된 실험 코드 (fork: slcnvly/HWM_PLDM, branch: causal-waypoint-student).
# Eval-time only. 재학습 없음 -- surprise_minseg8_finetuned 체크포인트를 그대로 쓰고,
# 추론 시점의 서브골(레벨2 waypoint) 종료 조건만 시간 기반(고정 replan_every)에서
# 도달 기반으로 바꿔서 비교한다.
#
# mpc.py의 `_compute_replan_mask`(새로 추가된, 기본 동작을 그대로 재현하는 오버라이드
# 포인트)를 오버라이드해서 env별로 독립적으로 "도달했다"/"max_k 초과"를 판정한다.
# TwoLvlPlanner.plan()이 항상 배치 전체에 대해 한 번에 forward pass를 돌리는 구조라서
# (mpc.py 자체를 크게 안 건드리는 한 이건 못 바꿈), 도달 안 한 env가 있어도 배치
# 전체에 대해 planner.plan()은 매 스텝 호출될 수 있다 -- 다만 mpc.py 쪽에서 각 env는
# 자기가 replan_mask=True일 때만 새 계획을 "채택"하므로, 실행되는 행동 자체는 env별로
# 완전히 독립적이다 (계산은 배치로 하되 채택은 env별).
#
# 거리 계산은 RunningCost(mppi_planner.py:221-270)와 같은 latent 공간(L1 backbone의
# obs_component, flatten_conv_output 적용)을 쓰되, 그 함수 자체는 재사용하지 않는다 --
# RunningCost는 MPPI가 상상한 롤아웃 상태에만 쓰이고 진짜 관측을 받지 않기 때문에
# (조사 결과 확인됨), 여기서 새로 "진짜 관측 인코딩 vs 저장된 타겟" 거리를 계산한다.
# 공식은 mpc.py:499(`torch.norm(..., dim=...)`, 진짜 유클리드)을 따른다 -- RunningCost의
# sqrt 없는 MSE보다 사람이 읽는 ε 임계값 해석이 더 직관적이라 그쪽 패턴을 골랐다.
import numpy as np
import torch

from pldm.models.utils import flatten_conv_output
from pldm.planning.d4rl.hmpc import HierarchicalD4RLMPCEvaluator


def encode_real_obs(model, obs_t: torch.Tensor, envs) -> torch.Tensor:
    """Encode the real current observation exactly the way
    TwoLvlPlanner.plan() does (two_lvl_planner.py:48-50) -- same backbone,
    same proprio/location gathering convention as _perform_mpc's own
    replan block -- so the resulting encoding lives in the same latent
    space as l1_planner.objective.target_enc."""
    curr_proprio_pos = None
    if model.level1.using_proprio_pos:
        curr_proprio_pos = torch.from_numpy(
            np.stack([e.get_proprio_pos(normalized=True) for e in envs])
        ).float()

    curr_proprio_vel = None
    if model.level1.using_proprio_vel:
        curr_proprio_vel = torch.from_numpy(
            np.stack([e.get_proprio_vel(normalized=True) for e in envs])
        ).float()

    proprio_l1 = None
    if curr_proprio_pos is not None and curr_proprio_vel is not None:
        proprio_l1 = torch.cat([curr_proprio_pos, curr_proprio_vel], dim=-1).cuda()
    elif curr_proprio_pos is not None:
        proprio_l1 = curr_proprio_pos.cuda()
    elif curr_proprio_vel is not None:
        proprio_l1 = curr_proprio_vel.cuda()

    locations_cuda = None
    if model.level1.using_location:
        curr_locations = torch.from_numpy(
            np.stack([e.get_pos(normalized=True) for e in envs])
        ).float()
        locations_cuda = curr_locations.cuda()

    with torch.no_grad():
        # BUG FIX (caught by the first real Kaggle run): HJEPA has no top-level
        # .backbone -- only .level1.backbone / .level2.backbone. Same object
        # two_lvl_planner.py:48-50 calls (self.l1_planner.model.backbone),
        # just reached via HJEPA.level1 instead of a planner's own .model ref.
        backbone_output = model.level1.backbone(obs_t.cuda(), proprio=proprio_l1, locations=locations_cuda)
    # BUG FIX #2 (caught by the second real Kaggle run): target_enc is NOT the
    # full combined encoding -- hjepa.py's forward_posterior feeds level2's
    # predictor (and hence pred_obs, what reset_targets stores) the OBS-ONLY
    # component (`backbone_output.obs_component`, excludes proprio channels)
    # whenever `level2.backbone.using_proprio` is True (hjepa.py:175-186,
    # confirmed by this run's shape mismatch: 33282 = full encodings,
    # 29584 = obs_component only). Match that space, not .encodings.
    enc = backbone_output.obs_component
    enc = flatten_conv_output(enc) if enc.dim() >= 3 else enc
    return enc


class ArrivalBasedHierarchicalD4RLMPCEvaluator(HierarchicalD4RLMPCEvaluator):
    """Same pattern as ErrorAdaptiveHierarchicalD4RLMPCEvaluator
    (hmpc.py) -- a thin subclass activated by a config flag (checked by the
    caller, pldm/evaluation/evaluator.py), here overriding
    `_compute_replan_mask` (mpc.py) instead of using `post_l1_step_hook`.

    Per-env, per-step decision (i==0 always bootstraps -- no target exists
    yet):
        arrived    = ||encode(obs_t) - target_enc|| < epsilon
        forced     = steps_since_last_replan >= max_k
        min_gap_ok = steps_since_last_replan >= min_gap
        replan     = (arrived or forced) and min_gap_ok

    Records, per replan event, whether it was triggered by arrival or by
    the max_k safety fallback -- see `advance_reason_counts` -- so a mostly-
    forced run can be flagged as "arrival-based termination didn't actually
    engage" during analysis.
    """

    def __init__(self, config, normalizer, model, pixel_mapper, prober=None,
                 prober_l2=None, prefix: str = "d4rl_h_", quick_debug: bool = False,
                 l2_use_latent_mean_std: bool = False):
        super().__init__(
            config=config, normalizer=normalizer, model=model, pixel_mapper=pixel_mapper,
            prober=prober, prober_l2=prober_l2, prefix=prefix, quick_debug=quick_debug,
            l2_use_latent_mean_std=l2_use_latent_mean_std,
        )
        self.arrival_epsilon = config.arrival_epsilon
        self.arrival_max_k = config.arrival_max_k
        self.arrival_min_gap = config.arrival_min_gap
        # reset per call to _perform_mpc (one entry per chunk/trial batch)
        self.advance_reason_counts = {"arrival": 0, "forced": 0, "initial": 0}
        self.last_distances = None  # (bs,) most recent per-env arrival distance, for logging

    def evaluate(self):
        """Same as HierarchicalD4RLMPCEvaluator.evaluate(), plus printing
        advance_reason_counts in a grep-able format -- this runs inside a
        `python -m pldm.train` subprocess the orchestrating kernel script
        can't otherwise reach into, so stdout is the handoff channel (the
        orchestrator captures it via `capture=True`)."""
        import json

        report = super().evaluate()
        print(f"ARRIVAL_STATS: {json.dumps(self.advance_reason_counts)}", flush=True)
        return report

    def _compute_replan_mask(self, i, bs, obs_t, planner, local_offset, envs):
        # BUG FIX (caught by the 3rd real Kaggle run): _perform_mpc is also
        # called in "Stage 2" flat-L1 mode (final_trans_steps, the non-
        # hierarchical tail of an episode -- see mpc.py's _perform_h_mpc),
        # where bilevel_planning=False and `planner` is the bare MPPIPlanner
        # (no .l1_planner attribute -- that only exists on TwoLvlPlanner).
        # This override only makes sense for the hierarchical/bilevel call;
        # fall back to the exact default (fixed replan_every) otherwise.
        if not hasattr(planner, "l1_planner"):
            return super()._compute_replan_mask(i, bs, obs_t, planner, local_offset, envs)

        if i == 0:
            self.advance_reason_counts["initial"] += bs
            return torch.full((bs,), True, dtype=torch.bool)

        current_enc = encode_real_obs(self.model, obs_t, envs)
        target_enc = planner.l1_planner.objective.target_enc
        if target_enc.dim() >= 3:
            target_enc = flatten_conv_output(target_enc)
        distance = torch.norm(current_enc - target_enc.to(current_enc.device), dim=-1).cpu()
        self.last_distances = distance

        arrived = distance < self.arrival_epsilon
        forced = local_offset >= self.arrival_max_k
        min_gap_ok = local_offset >= self.arrival_min_gap
        replan_mask = (arrived | forced) & min_gap_ok

        for j in range(bs):
            if replan_mask[j]:
                self.advance_reason_counts["arrival" if arrived[j] and not forced[j] else "forced"] += 1

        return replan_mask
