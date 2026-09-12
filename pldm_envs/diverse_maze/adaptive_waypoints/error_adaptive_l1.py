# HWM_PLDM 원본 대비 추가된 실험 코드 (fork: slcnvly/HWM_PLDM, branch: adaptive-waypoints).
# See DESIGN.md and RESULTS.md SS7 for the full rationale. Eval-time only --
# nothing here is imported by pldm/train.py or any training path.
#
# Error-adaptive L1 resource allocation: after every real MPC step, compute a
# one-step latent prediction error (same formula as compute_changepoints.py's
# changepoint signal) and, if it exceeds a fixed threshold, boost the L1 MPPI
# planner's num_samples (per-env) and planning horizon (whole-batch) for the
# next planning call. Two deliberate design choices, carried over from
# compute_changepoints.py so the online error and the threshold are on the
# same scale:
#   1. The error is latent-space MSE (encodings vs. one-step-ahead
#      predictions), not a raw xy/pixel distance.
#   2. It's computed with the SAME separate frozen L1-only checkpoint
#      compute_changepoints.py used (3-9-1-seed248...), not the L1 weights
#      inside whichever hierarchical checkpoint (baseline/adaptive) is
#      currently being planned with -- those are two different pretrained
#      models in this repo.
import numpy as np
import torch

from pldm_envs.diverse_maze.adaptive_waypoints.compute_changepoints import (
    compute_error_series,
    load_level1,
)


def compute_r50_error_threshold(
    data_path: str,
    config_path: str,
    checkpoint_path: str,
    device: str = "cuda",
    l2_step_skip: int = 10,
    l2_n_steps: int = 6,
    limit_episodes: int = None,
    n_sigma: float = 2.0,
):
    """Reruns compute_changepoints.py's exact per-step error computation over
    the r50 main dataset (same model, same windowing), but returns the raw
    error distribution's mean/std instead of picking changepoints -- this is
    the threshold basis for error-adaptive L1 resource allocation.

    Returns (mean, std, threshold=mean + n_sigma * std, n_steps_total).
    """
    model = load_level1(config_path, checkpoint_path, device)

    splits = torch.load(f"{data_path}/data.p", weights_only=False)
    images = np.load(f"{data_path}/images.npy", mmap_mode="r")

    window = l2_n_steps * l2_step_skip + 1
    cum_lengths = np.cumsum([len(s["observations"]) for s in splits])
    n_episodes = len(splits) if limit_episodes is None else min(limit_episodes, len(splits))

    all_errs = []
    for ep_idx in range(n_episodes):
        ep_len = len(splits[ep_idx]["observations"])
        if ep_len < window:
            continue
        start = 0 if ep_idx == 0 else cum_lengths[ep_idx - 1]

        obs = splits[ep_idx]["observations"][:window]
        proprio_vel = torch.from_numpy(obs[:, 2:4]).float().unsqueeze(1).to(device)
        img_seq = torch.from_numpy(
            np.array(images[start : start + window])
        ).float().permute(0, 3, 1, 2)
        states = img_seq.unsqueeze(1).to(device)
        actions = torch.from_numpy(
            splits[ep_idx]["actions"][: window - 1]
        ).float().unsqueeze(1).to(device)

        err = compute_error_series(model, states, actions, proprio_vel)
        all_errs.append(err)

        if ep_idx % 200 == 0:
            print(f"[threshold calib] episode {ep_idx}/{n_episodes}", flush=True)

    all_errs = torch.cat(all_errs)
    mean = all_errs.mean().item()
    std = all_errs.std().item()
    threshold = mean + n_sigma * std
    print(
        f"[threshold calib] n_steps={all_errs.numel()} mean={mean:.6f} "
        f"std={std:.6f} threshold(+{n_sigma}sigma)={threshold:.6f}",
        flush=True,
    )
    return mean, std, threshold, int(all_errs.numel())


class L1ErrorMonitor:
    """Computes the online one-step latent prediction error for error-adaptive
    L1 resource allocation, using a SEPARATE frozen L1-only checkpoint (see
    module docstring) -- entirely independent of whichever hierarchical
    checkpoint is being planned with.
    """

    def __init__(self, config_path: str, checkpoint_path: str, normalizer, device: str = "cuda"):
        self.model = load_level1(config_path, checkpoint_path, device)
        self.normalizer = normalizer
        self.device = device
        self._prev_obs = None
        self._prev_proprio = None

    def reset(self):
        """Call once per fresh MPC rollout (new chunk / new set of envs) so
        the first real step doesn't compare against a stale previous obs."""
        self._prev_obs = None
        self._prev_proprio = None

    @torch.no_grad()
    def step_error(self, current_obs: torch.Tensor, action: torch.Tensor, current_proprio: torch.Tensor):
        """
        Args:
            current_obs: (bs, 3, 98, 98) NORMALIZED obs, straight from the
                MPC loop's env.step() this real step.
            action: (bs, 2) raw (unnormalized) action just executed.
            current_proprio: (bs, 2) raw (unnormalized) proprio vel after the
                step (from env info, get_info()'s "proprio", normalized=False).

        Returns:
            (bs,) per-env squared error, or None if there's no previous obs
            yet (first real step of a rollout) -- caller should treat that as
            "no boost" for the upcoming plan, which is correct: there's no
            prior real-step outcome to have detected a surprise from yet.
        """
        obs_after = self.normalizer.unnormalize_state(current_obs).to(self.device)
        proprio_after = current_proprio.to(self.device)

        if self._prev_obs is None:
            self._prev_obs = obs_after
            self._prev_proprio = proprio_after
            return None

        states = torch.stack([self._prev_obs, obs_after], dim=0)  # (2, bs, 3, 98, 98)
        proprio_vel = torch.stack([self._prev_proprio, proprio_after], dim=0)  # (2, bs, 2)
        actions = action.to(self.device).float().unsqueeze(0)  # (1, bs, 2)

        result = self.model.level1.forward_posterior(
            states, actions, proprio_vel=proprio_vel, encode_only=False
        )
        encodings = result.backbone_output.encodings[1:]  # (1, bs, ...)
        predictions = result.pred_output.predictions[1:]
        err = (encodings - predictions).pow(2).flatten(2).mean(dim=-1).squeeze(0)  # (bs,)

        self._prev_obs = obs_after
        self._prev_proprio = proprio_after
        return err.cpu()


class ErrorAdaptiveL1Planner:
    """Wraps an L1 MPPIPlanner (TwoLvlPlanner.l1_planner) to conditionally
    boost num_samples (per-env) and planning horizon (whole-batch) for the
    NEXT plan() call, based on a boost mask set externally after each real
    step. num_samples is per-env because MPPIPlanner already loops per-env in
    Python for action selection (self.ctrls[i].K is a plain, independently
    settable attribute); horizon can't be, because MPPIPlanner.plan() takes
    one shared plan_size per batched call (the dynamics rollout used to
    produce pred_obs/locations for the whole batch requires a uniform
    sequence length) -- so horizon boosts for the whole chunk whenever *any*
    env in it tripped the threshold.
    """

    def __init__(
        self,
        l1_planner,
        base_num_samples: int,
        base_plan_size: int,
        num_samples_multiplier: float = 2.0,
        horizon_multiplier: float = 1.5,
    ):
        self._l1 = l1_planner
        self.base_num_samples = base_num_samples
        self.base_plan_size = base_plan_size
        self.boosted_num_samples = round(base_num_samples * num_samples_multiplier)
        self.boosted_plan_size = round(base_plan_size * horizon_multiplier)
        self._boost_mask = None
        # per-step diagnostics for RESULTS.md SS7 (chunk-level boost rate)
        self.n_steps_seen = 0
        self.n_steps_boosted = 0

    def set_boost_mask(self, boost_mask):
        """boost_mask: (n_envs,) bool tensor, or None (-> no boost, base
        settings for every env)."""
        self._boost_mask = boost_mask

    def plan(self, current_state, plan_size, **kwargs):
        self.n_steps_seen += 1
        if self._boost_mask is not None and bool(self._boost_mask.any()):
            self.n_steps_boosted += 1
            for i, ctrl in enumerate(self._l1.ctrls):
                ctrl.K = (
                    self.boosted_num_samples if self._boost_mask[i] else self.base_num_samples
                )
            eff_plan_size = self.boosted_plan_size
        else:
            for ctrl in self._l1.ctrls:
                ctrl.K = self.base_num_samples
            eff_plan_size = self.base_plan_size

        return self._l1.plan(current_state=current_state, plan_size=eff_plan_size, **kwargs)

    def reset_targets(self, *args, **kwargs):
        return self._l1.reset_targets(*args, **kwargs)

    @property
    def model(self):
        return self._l1.model
