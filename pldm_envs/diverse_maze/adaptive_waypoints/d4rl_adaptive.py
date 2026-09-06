# HWM_PLDM 원본 대비 추가된 실험 코드 (fork: slcnvly/HWM_PLDM, branch: adaptive-waypoints).
# See DESIGN.md. Subclasses D4RLDataset, overriding only _build_l2_sample
# (a small extracted method on the base class -- see the diff on
# pldm_envs/diverse_maze/d4rl.py, a pure behavior-preserving refactor) to
# replace fixed-stride l2 chunking with changepoint-based variable
# boundaries, falling back to the exact baseline behavior for any episode
# missing precomputed changepoints (e.g. too short for a full l2 window).
import os

import torch

from pldm_envs.diverse_maze.adaptive_waypoints.segmentation import (
    boundaries_to_segments,
    resample_to_fixed_length,
)
from pldm_envs.diverse_maze.d4rl import D4RLDataset


class AdaptiveD4RLDataset(D4RLDataset):
    def __init__(self, config, min_seg: int, images_tensor=None, load_l1=True):
        super().__init__(config, images_tensor=images_tensor, load_l1=load_l1)
        self.min_seg = min_seg

        cp_path = os.path.join(
            os.path.dirname(config.path), f"changepoints_minseg{min_seg}.pt"
        )
        if os.path.exists(cp_path):
            self.changepoints = torch.load(cp_path, weights_only=False)
        else:
            print(
                f"WARNING: no precomputed changepoints at {cp_path} -- "
                "AdaptiveD4RLDataset will fall back to uniform stride for "
                "every episode (equivalent to baseline)."
            )
            self.changepoints = {}

    def _build_l2_sample(self, episode_idx, start_idx):
        if self.l2_n_steps_total <= 0:
            return super()._build_l2_sample(episode_idx, start_idx)

        boundaries = self.changepoints.get(episode_idx)
        if boundaries is None:
            # Episode has no cached changepoints (too short, or wasn't in
            # the preprocessing run) -- fall back to exact baseline chunking
            # rather than silently skipping the sample.
            return super()._build_l2_sample(episode_idx, start_idx)

        window = self.l2_n_steps_total  # e.g. 61: config.l2_n_steps*l2_step_skip + 1
        action_len = window - 1  # e.g. 60

        states, locations, actions, proprio_vel, proprio_pos = (
            self._load_data_from_start_idx(
                episode_idx=episode_idx,
                start_idx=start_idx,
                length=window + self.config.stack_states - 1,
                skip_frame=1,  # dense: we pick anchors ourselves below
            )
        )
        assert states.shape[0] == window, (
            f"expected {window} dense states, got {states.shape[0]} -- "
            "changepoints were computed for a different l2_n_steps/l2_step_skip"
        )

        anchor_idx = [0] + list(boundaries) + [window - 1]
        l2_states = states[anchor_idx]
        l2_locations = locations[anchor_idx]
        l2_proprio_vel = proprio_vel[anchor_idx]
        l2_proprio_pos = (
            proprio_pos[anchor_idx] if proprio_pos.numel() else proprio_pos
        )

        seg_len = self.config.l2_step_skip  # keep the action encoder's fixed input size
        segments = boundaries_to_segments(boundaries, action_len)
        assert len(segments) == self.config.l2_n_steps
        l2_actions = torch.stack(
            [resample_to_fixed_length(actions[s:e], seg_len) for s, e in segments],
            dim=0,
        )

        return l2_states, l2_locations, l2_proprio_vel, l2_proprio_pos, l2_actions
