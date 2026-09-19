# HWM_PLDM 원본 대비 추가된 실험 코드 (fork: slcnvly/HWM_PLDM, branch: causal-waypoint-student).
# Eval-time only -- nothing here is imported by pldm/train.py or any training path
# (같은 원칙: pldm_envs/diverse_maze/adaptive_waypoints/error_adaptive_l1.py 참고).
#
# L1ErrorMonitor(error_adaptive_l1.py)와 정확히 같은 패턴을 그대로 따른다:
# 매 실제 스텝(env.step() 이후)마다 causal하게 값을 계산하고, 첫 스텝엔 판단
#불가(None)를 반환한다. 다른 점은 L1ErrorMonitor는 raw error만 보고 MPPI
# 자원 배분을 조정하는 반면, 이 모듈은 surprise(err^2/sigma^2) + 학습된
# student(Δτ 회귀 MLP)로 "지금이 waypoint인가"를 직접 결정한다.
#
# 배치 처리: 실제 MPC eval은 여러 env를 (bs,...) 텐서로 동시에 굴리므로
# (error_adaptive_l1.py의 L1ErrorMonitor와 동일), 이 클래스도 env별 상태
# (prev obs/proprio/action, surprise 이력, 마지막 waypoint 이후 경과 스텝)를
# 전부 (bs,...) 텐서로 유지한다.
import torch

from pldm_envs.diverse_maze.adaptive_waypoints.compute_changepoints import load_level1
from pldm_envs.diverse_maze.adaptive_waypoints.student_model import StudentModel, make_features
from pldm_envs.diverse_maze.adaptive_waypoints.variance_head import VarianceHead, standardize


class CausalWaypointTrigger:
    """Loads three frozen/trained components (L1-only world model, trained
    VarianceHead, trained StudentModel) and, given real observations one
    step at a time, causally decides whether the current step is a waypoint.

    Usage (mirrors L1ErrorMonitor):
        trigger = CausalWaypointTrigger(...)
        trigger.reset()          # once per fresh rollout
        for each real step:
            is_wp, delta_tau_hat = trigger.step(obs_after, action_taken, proprio_after)
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        vhead_checkpoint_path: str,
        student_checkpoint_path: str,
        normalizer=None,
        device: str = "cuda",
        min_seg: int = 8,
        threshold: float = 1.0,
        eps: float = 1e-3,
    ):
        self.model = load_level1(config_path, checkpoint_path, device)

        vckpt = torch.load(vhead_checkpoint_path, map_location="cpu", weights_only=False)
        self.vhead_input_mean, self.vhead_input_std, self.vhead_target_scale = (
            vckpt["input_mean"], vckpt["input_std"], vckpt["target_scale"]
        )
        self.vhead = VarianceHead(self.vhead_input_mean.shape[0], hidden=128)
        self.vhead.load_state_dict(vckpt["model_state"])
        self.vhead.to(device).eval()

        sckpt = torch.load(student_checkpoint_path, map_location="cpu", weights_only=False)
        self.student = StudentModel(sckpt["input_dim"])
        self.student.load_state_dict(sckpt["model_state"])
        self.student.to(device).eval()
        self.target_transform = sckpt["target_transform"]
        self.K = sckpt["K"]
        self.stats = (
            sckpt["enc_mean"].to(device), sckpt["enc_std"].to(device),
            sckpt["act_mean"].to(device), sckpt["act_std"].to(device),
            sckpt["surp_mean"].to(device), sckpt["surp_std"].to(device),
        )

        self.normalizer = normalizer
        self.device = device
        self.min_seg = min_seg
        self.threshold = threshold
        self.eps = eps

        self._prev_obs = None
        self._prev_proprio = None
        self._surprise_hist = None  # (bs, K), lazily allocated on first real step()
        self._steps_since_waypoint = None  # (bs,) int

    def reset(self):
        """Call once per fresh MPC rollout (new chunk / new set of envs) --
        same convention as L1ErrorMonitor.reset()."""
        self._prev_obs = None
        self._prev_proprio = None
        self._surprise_hist = None
        self._steps_since_waypoint = None

    @torch.no_grad()
    def step(self, current_obs: torch.Tensor, action: torch.Tensor, current_proprio: torch.Tensor):
        """
        Args (identical convention to L1ErrorMonitor.step_error):
            current_obs: (bs, 3, 98, 98) obs straight from env.step(), NORMALIZED
                if `self.normalizer` was given at construction (real online MPC
                use), or already-raw pixels if normalizer=None (offline replay
                over recorded images.npy in generate_causal_labels.py -- that
                data was never normalizer-processed to begin with, same
                convention student_labels.py/compute_changepoints.py already use).
            action: (bs, 2) raw action JUST EXECUTED -- the action that
                produced current_obs from the previous obs (i.e. a_{t-1} in
                student_labels.py's indexing), exactly like
                L1ErrorMonitor.step_error's `action` arg. NOT a lookahead
                action -- this call uses `action` directly, it does not
                buffer it for the next call (an earlier draft of this file
                buffered it one step too far; fixed to match
                L1ErrorMonitor's actual behavior exactly).
            current_proprio: (bs, 2) raw proprio vel after the step.

        Returns:
            None on the first call of a rollout (no previous obs yet -- caller
            should treat that as "not a waypoint, don't know yet").
            Otherwise (is_waypoint: (bs,) bool tensor, delta_tau_hat: (bs,) float tensor).
        """
        obs_after = self.normalizer.unnormalize_state(current_obs) if self.normalizer is not None else current_obs
        obs_after = obs_after.to(self.device)
        proprio_after = current_proprio.to(self.device)
        bs = obs_after.shape[0]

        if self._prev_obs is None:
            self._prev_obs, self._prev_proprio = obs_after, proprio_after
            self._surprise_hist = torch.zeros(bs, self.K, device=self.device)
            self._steps_since_waypoint = torch.zeros(bs, dtype=torch.long, device=self.device)
            return None

        states = torch.stack([self._prev_obs, obs_after], dim=0)  # (2, bs, 3, 98, 98)
        proprio_vel = torch.stack([self._prev_proprio, proprio_after], dim=0)  # (2, bs, 2)
        actions_in = action.to(self.device).float().unsqueeze(0)  # (1, bs, 2) -- a_{t-1}, see docstring

        result = self.model.level1.forward_posterior(states, actions_in, proprio_vel=proprio_vel, encode_only=False)
        encodings = result.backbone_output.encodings  # (2, bs, ...)
        predictions = result.pred_output.predictions[1:]  # (1, bs, ...)
        e_prev = encodings[0].flatten(1)  # (bs, repr_dim) -- e_{t-1}
        e_t = encodings[1].flatten(1)  # (bs, repr_dim) -- e_t
        err = (encodings[1] - predictions[0]).pow(2).flatten(1).mean(dim=-1)  # (bs,)

        enc_std_prev = standardize(e_prev.half(), self.vhead_input_mean, self.vhead_input_std).to(self.device)
        log_var_norm = self.vhead(enc_std_prev)
        var = torch.exp(log_var_norm) * self.vhead_target_scale
        surprise = err.pow(2) / (var + self.eps)  # same formula as SS8 / extract_embeddings.py

        # roll the (bs, K) history buffer: drop oldest, append this step's surprise
        self._surprise_hist = torch.cat([self._surprise_hist[:, 1:], surprise.unsqueeze(1)], dim=1)

        a_prev = action.to(self.device).float()
        feat = make_features(e_t, e_prev, a_prev, self._surprise_hist, *self.stats)

        pred = self.student(feat)
        delta_tau_hat = torch.expm1(pred) if self.target_transform == "log1p" else pred
        delta_tau_hat = delta_tau_hat.clamp_min(0.0)

        is_waypoint = (delta_tau_hat <= self.threshold) & (self._steps_since_waypoint >= self.min_seg)

        self._steps_since_waypoint = torch.where(
            is_waypoint, torch.zeros_like(self._steps_since_waypoint), self._steps_since_waypoint + 1
        )

        self._prev_obs, self._prev_proprio = obs_after, proprio_after
        return is_waypoint, delta_tau_hat
