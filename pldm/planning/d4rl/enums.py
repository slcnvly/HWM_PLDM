from dataclasses import dataclass
from typing import NamedTuple
import torch
from pldm.planning.enums import MPCConfig
from pldm.planning.planners.enums import PlannerConfig


@dataclass
class D4RLMPCConfig(MPCConfig):
    position_only: bool = True
    subgoal_planning: bool = False
    set_full_states: bool = False
    unique_shortest_path: bool = (
        False  # gen trials where there's a unique shortest path
    )


@dataclass
class HierarchicalD4RLMPCConfig(D4RLMPCConfig):
    final_trans_norm_cutoff: float = 4
    final_trans_steps: int = 15
    error_threshold: float = 1.0
    mock_l1: bool = False
    level2: PlannerConfig = PlannerConfig()
    # Error-adaptive L1 resource allocation (RESULTS.md SS7, eval-only --
    # unrelated to error_threshold above, which is a pre-existing field used
    # elsewhere). See pldm_envs/diverse_maze/adaptive_waypoints/error_adaptive_l1.py.
    error_adaptive_l1: bool = False
    error_adaptive_l1_threshold: float = 0.0
    error_adaptive_l1_config_path: str = ""
    error_adaptive_l1_checkpoint_path: str = ""
    error_adaptive_l1_num_samples_multiplier: float = 2.0
    error_adaptive_l1_horizon_multiplier: float = 1.5
    # Arrival-based subgoal termination (this session's experiment, eval-only,
    # no training-code changes). Unrelated to error_adaptive_l1 above -- this
    # replaces the fixed replan_every cadence with a per-env arrival check.
    # See eval/arrival_based_termination.py.
    arrival_based_termination: bool = False
    arrival_epsilon: float = 0.0
    # ARCHITECTURE CONSTRAINT (discovered by the 4th real Kaggle run): L1's
    # own plan is only ever computed for `l2_step_skip` (=10) raw steps --
    # two_lvl_planner.py:77-83 hardcodes `plan_size=self.l2_step_skip` on
    # every L1 plan() call, unconditionally, whether or not arrival-based
    # termination is active. mpc.py's overflow safety net (added for THIS
    # feature) forces a replan once local_offset reaches that 10-step plan's
    # length regardless of max_k, so any max_k > ~10 is unreachable -- this
    # class's own max_k check never gets to fire first. Confirmed by a real
    # run where max_k=20 produced IDENTICAL success/steps for all three
    # epsilon candidates: mpc.py's overflow-driven replans (every ~10 steps,
    # invisible to this class's advance_reason_counts, which only sees what
    # ITS OWN _compute_replan_mask returned before mpc.py ORs in overflow)
    # dominated every trial, because the arrival distance never happened to
    # drop below any candidate epsilon within the only window that's
    # actually reachable before overflow fires (steps min_gap..10). Default
    # lowered to fit inside that real window so max_k (not overflow) is the
    # effective cap, and so advance_reason_counts is accurate again.
    arrival_max_k: int = 8
    arrival_min_gap: int = 2


class MPCReport(NamedTuple):
    success_rate: float
    success: torch.Tensor
    avg_steps_to_goal: float
    median_steps_to_goal: float
    terminations: list
    one_turn_success_rate: float
    two_turn_success_rate: float
    three_turn_success_rate: float
    num_one_turns: int
    num_two_turns: int
    num_three_turns: int
    num_turns: list
    block_dists: list
    ood_report: dict

    def build_log_dict(self, prefix=""):
        log_dict = {
            f"{prefix}_planning_success_rate": self.success_rate,
            f"{prefix}_avg_steps_to_goal": self.avg_steps_to_goal,
            f"{prefix}_median_steps_to_goal": self.median_steps_to_goal,
            f"{prefix}_one_turn_success_rate": self.one_turn_success_rate,
            f"{prefix}_two_turn_success_rate": self.two_turn_success_rate,
            f"{prefix}_three_turn_success_rate": self.three_turn_success_rate,
            f"{prefix}_num_one_turns": self.num_one_turns,
            f"{prefix}_num_two_turns": self.num_two_turns,
            f"{prefix}_num_three_turns": self.num_three_turns,
            # f"{prefix}_block_dist_p": self.block_dist_p,
            # f"{prefix}_num_turns_p": self.num_turns_p,
        }

        log_dict.update(self.ood_report)
        return log_dict
