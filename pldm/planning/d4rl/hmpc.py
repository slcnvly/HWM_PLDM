from typing import Optional
import torch
import time

from pldm.models.hjepa import HJEPA
from pldm_envs.utils.normalizer import Normalizer
from pldm.planning.utils import *
from pldm.planning.plotting import log_l2_planning_loss, log_planning_plots
from pldm.planning.d4rl.enums import (
    MPCReport,
    HierarchicalD4RLMPCConfig,
)
from pldm.planning.d4rl.mpc import MazeMPCEvaluator
from pldm.planning.enums import PooledMPCResult
from pldm.planning.planners.enums import PlannerType
from pldm.utils import format_seconds
from pldm_envs.diverse_maze.utils import PixelMapper


class HierarchicalD4RLMPCEvaluator(MazeMPCEvaluator):
    def __init__(
        self,
        config: HierarchicalD4RLMPCConfig,
        normalizer: Normalizer,
        model: HJEPA,
        pixel_mapper: PixelMapper,
        prober: Optional[torch.nn.Module] = None,
        prober_l2: Optional[torch.nn.Module] = None,
        prefix: str = "d4rl_h_",
        quick_debug: bool = False,
        l2_use_latent_mean_std: bool = False,
    ):
        super().__init__(
            config=config,
            normalizer=normalizer,
            model=model,
            pixel_mapper=pixel_mapper,
            prober=prober,
            prober_l2=prober_l2,
            prefix=prefix,
            quick_debug=quick_debug,
            l2_use_latent_mean_std=l2_use_latent_mean_std,
        )

        self.hierarchical = True

        # Override to use hierarchical planner
        self.h_planner = self._construct_h_planner(config.n_envs)

    def evaluate(self):
        start_time = time.time()

        data = self._perform_mpc_in_chunks()

        elapsed_time = int(time.time() - start_time)
        print(f"hierarchical d4rl planning took {format_seconds(elapsed_time)}")

        report = self._construct_report(data, elapsed_time=elapsed_time)

        log_l2_planning_loss(data, prefix=self.prefix)

        data.targets = (
            data.targets[:, :2] if len(data.targets.shape) > 1 else data.targets
        )

        if self.config.visualize_planning:
            log_planning_plots(
                result=data,
                report=report,
                idxs=list(range(self.config.n_envs)) if not self.quick_debug else [0],
                prefix=self.prefix,
                n_steps=self.config.n_steps,
                xy_action=True,
                plot_every=self.config.plot_every,
                quick_debug=self.quick_debug,
                pixel_mapper=self.pixel_mapper,
                plot_failure_only=self.config.plot_failure_only,
                log_pred_dist_every=self.config.log_pred_dist_every,
                mark_action=False,
            )

        return data, report

    def _construct_report(self, data: PooledMPCResult, elapsed_time: float = 0):
        T = len(data.reward_history)
        B = data.reward_history[0].shape[0]

        terminations = [T] * B

        for b_i in range(B):
            for t_i in range(T):
                if data.reward_history[t_i][b_i]:
                    terminations[b_i] = t_i
                    break

        successes = [int(x < T) for x in terminations]
        success_rate = sum(successes) / len(successes)

        num_turns = [x.turns for x in self.envs]

        one_turn_successes = [
            successes[i] for i in range(len(num_turns)) if num_turns[i] == 1
        ]
        two_turn_successes = [
            successes[i] for i in range(len(num_turns)) if num_turns[i] == 2
        ]
        three_turn_successes = [
            successes[i] for i in range(len(num_turns)) if num_turns[i] == 3
        ]

        block_dists = [x.block_dist for x in self.envs]

        avg_steps_to_goal = calc_avg_steps_to_goal(data.reward_history)

        median_steps_to_goal = calc_avg_steps_to_goal(
            data.reward_history, reduce_type="median"
        )

        ood_report = self._construct_ood_report(data, successes=successes)

        report = MPCReport(
            success_rate=success_rate,
            success=torch.tensor(successes),
            avg_steps_to_goal=avg_steps_to_goal,
            median_steps_to_goal=median_steps_to_goal,
            terminations=terminations,
            one_turn_success_rate=(
                sum(one_turn_successes) / len(one_turn_successes)
                if one_turn_successes
                else -1
            ),
            two_turn_success_rate=(
                sum(two_turn_successes) / len(two_turn_successes)
                if two_turn_successes
                else -1
            ),
            three_turn_success_rate=(
                sum(three_turn_successes) / len(three_turn_successes)
                if three_turn_successes
                else -1
            ),
            num_one_turns=len(one_turn_successes),
            num_two_turns=len(two_turn_successes),
            num_three_turns=len(three_turn_successes),
            num_turns=num_turns,
            block_dists=block_dists,
            ood_report=ood_report,
        )

        return report

    def determine_optimal_depths(
        self,
        obs,
        targets,
        target_repr,
        planner,
    ):
        config = self.config.level2
        hjepa = self.model
        batch_size = obs.shape[0]
        dist_to_target_repr = torch.full((batch_size, config.max_plan_length + 1), 1e9)

        for depth in range(config.min_plan_length, config.max_plan_length + 1):
            # Get representation of observation
            l1_output = hjepa.level1.backbone(obs.cuda())
            l2_output = hjepa.level2.backbone(
                l1_output.obs_component, proprio=l1_output.proprio_component
            )
            enc2 = l2_output.encodings.detach()

            if config.planner_type == PlannerType.SGD:
                preds_l2, _, _, _ = planner.plan(
                    current_state=enc2,
                    steps_left=depth,
                    repr_input=True,
                )
            else:
                preds_l2, _, _, _ = planner.plan(
                    current_state=enc2,
                    plan_size=depth,
                )

            l2_dist = torch.norm(target_repr - preds_l2[-1], dim=1)
            dist_to_target_repr[:, depth] = l2_dist.detach()

        close_enough = (dist_to_target_repr < config.depth_probe_threshold).type(
            torch.int
        )
        opt_depths = torch.argmin(dist_to_target_repr, dim=1)  # (bs,)
        opt_depths[torch.any(close_enough, dim=1)] = torch.argmax(close_enough, dim=1)[
            torch.any(close_enough, dim=1)
        ]
        opt_steps = opt_depths * hjepa.config.step_skip

        return opt_steps


class ErrorAdaptiveHierarchicalD4RLMPCEvaluator(HierarchicalD4RLMPCEvaluator):
    """RESULTS.md SS7 (eval-only, no training-code changes). After every real
    MPC step, computes a one-step latent prediction error using a separate
    frozen L1-only checkpoint (see
    pldm_envs/diverse_maze/adaptive_waypoints/error_adaptive_l1.py) and, if it
    exceeds config.error_adaptive_l1_threshold, boosts the L1 MPPI planner's
    num_samples (per-env) and planning horizon (whole-batch) for the next
    planning call; otherwise the checkpoint's normal fixed settings are used.
    Only active when config.error_adaptive_l1 is set (checked by the caller,
    pldm/evaluation/evaluator.py) -- this class does no gating itself.
    """

    def __init__(
        self,
        config,
        normalizer,
        model,
        pixel_mapper,
        prober=None,
        prober_l2=None,
        prefix: str = "d4rl_h_",
        quick_debug: bool = False,
        l2_use_latent_mean_std: bool = False,
    ):
        from pldm_envs.diverse_maze.adaptive_waypoints.error_adaptive_l1 import (
            ErrorAdaptiveL1Planner,
            L1ErrorMonitor,
        )

        # Set up before calling super().__init__(): that call constructs
        # self.h_planner via self._construct_h_planner (overridden below),
        # which needs these already in place.
        self._ErrorAdaptiveL1Planner = ErrorAdaptiveL1Planner
        self.error_monitor = L1ErrorMonitor(
            config_path=config.error_adaptive_l1_config_path,
            checkpoint_path=config.error_adaptive_l1_checkpoint_path,
            normalizer=normalizer,
        )
        self.error_threshold = config.error_adaptive_l1_threshold
        self.num_samples_multiplier = config.error_adaptive_l1_num_samples_multiplier
        self.horizon_multiplier = config.error_adaptive_l1_horizon_multiplier
        self._active_l1_proxy = None

        super().__init__(
            config=config,
            normalizer=normalizer,
            model=model,
            pixel_mapper=pixel_mapper,
            prober=prober,
            prober_l2=prober_l2,
            prefix=prefix,
            quick_debug=quick_debug,
            l2_use_latent_mean_std=l2_use_latent_mean_std,
        )
        self.post_l1_step_hook = self._on_post_l1_step

    def _construct_h_planner(self, n_envs: int):
        h_planner = super()._construct_h_planner(n_envs)
        base_num_samples = self.config.level1.mppi.num_samples
        base_plan_size = h_planner.l2_step_skip
        h_planner.l1_planner = self._ErrorAdaptiveL1Planner(
            h_planner.l1_planner,
            base_num_samples=base_num_samples,
            base_plan_size=base_plan_size,
            num_samples_multiplier=self.num_samples_multiplier,
            horizon_multiplier=self.horizon_multiplier,
        )
        self._active_l1_proxy = h_planner.l1_planner
        self.error_monitor.reset()
        return h_planner

    def _on_post_l1_step(self, step, offset, planning_result, current_obs, infos):
        import numpy as np

        proprio_after = torch.from_numpy(
            np.stack([info["proprio"] for info in infos])
        ).float()
        action_taken = planning_result.actions[:, offset].detach().cpu()
        err = self.error_monitor.step_error(current_obs, action_taken, proprio_after)
        boost_mask = None if err is None else (err > self.error_threshold)
        if self._active_l1_proxy is not None:
            self._active_l1_proxy.set_boost_mask(boost_mask)
