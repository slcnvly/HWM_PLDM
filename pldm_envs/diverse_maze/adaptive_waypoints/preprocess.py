# Shared input normalization for offline analyses (changepoints, boundary study,
# inference study). Uses the same Normalizer class and the same hard-set
# statistics the training/evaluation pipeline uses for maze2d_large_diverse
# (normalizer_hardset: true -> STATS, pldm_envs/utils/normalizer.py:177-189), so
# offline encodings match what the model saw in training and sees in planning:
#   images        -> Normalizer.normalize_state   (eval: maze_draw.py:104-105)
#   proprio vel   -> Normalizer.normalize_proprio_vel (eval: wrappers.py:297-302)
#   actions       -> Normalizer.normalize_action  (training: normalize_sample)
import torch

from pldm_envs.utils.normalizer import STATS, Normalizer

_NORMALIZERS = {}


def eval_normalizer(env_name: str = "maze2d_large_diverse") -> Normalizer:
    if env_name not in _NORMALIZERS:
        s = STATS[env_name]
        _NORMALIZERS[env_name] = Normalizer(
            s["state_mean"], s["state_std"], s["action_mean"], s["action_std"],
            s["location_mean"], s["location_std"], s["proprio_pos_mean"], s["proprio_pos_std"],
            s["proprio_vel_mean"], s["proprio_vel_std"], min_max_state=False, image_based=True,
        )
    return _NORMALIZERS[env_name]


def normalize_images(x: torch.Tensor) -> torch.Tensor:
    """x: (..., 3, H, W) raw pixel values (uint8 or float of uint8 values).
    Returned contiguous: a permuted (channels-last-strided) batch makes the conv
    encoder take a different kernel path (rel. diff ~1e-5 vs the evaluator's
    stacked, contiguous frames)."""
    return eval_normalizer().normalize_state(x).contiguous()


def normalize_proprio_vel(x: torch.Tensor) -> torch.Tensor:
    """x: (..., 2) raw qvel."""
    return eval_normalizer().normalize_proprio_vel(x)


def normalize_actions(x: torch.Tensor) -> torch.Tensor:
    """x: (..., 2) raw env actions."""
    return eval_normalizer().normalize_action(x)
