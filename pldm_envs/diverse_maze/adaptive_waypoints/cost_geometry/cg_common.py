# COST_GEOMETRY shared pieces: representations (via latent_law/ab_saturation.py) and the
# synthetic renderer (4x supersampled coverage, alpha-composited on the map-0 agent-free background).
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
AW = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(AW, "latent_law"))
import ab_saturation as ab  # noqa: E402

REPS = ["B1_trained", "B2_random_seed0", "B2_random_seed1", "B2_random_seed2", "B4_pixels", "B3_dinov2_cls", "B3_dinov2_patch8x8"]
AGENT_RGB = np.array([68.0, 128.0, 77.0])  # C1 median colour
PX_PER_CELL = 9.34
SS = 4  # supersampling factor
W = 98


def load_encoders():
    encs = {"B1_trained": ab.load_model().level1.backbone, "B4_pixels": None}
    for s in range(3):
        encs[f"B2_random_seed{s}"] = ab.random_backbone(s)
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True, skip_validation=True).eval()
    encs["B3_dinov2_cls"] = dino
    encs["B3_dinov2_patch8x8"] = dino
    return encs


@torch.no_grad()
def feats(name, imgs_uint8_hwc, encs):
    """Same path as LATENT_LAW A/B: ab_saturation.features (preprocess.normalize_images for B1/B2/B4,
    224 resize + ImageNet norm for DINOv2)."""
    return ab.features(name, np.ascontiguousarray(imgs_uint8_hwc), encs[name])


def background():
    return np.load(os.path.join(HERE, "cg_backgrounds.npz"))["0"].astype(np.float64)


_G = (np.arange(W * SS) + 0.5) / SS  # supersample centres in px
_R, _C = np.meshgrid(_G, _G, indexing="ij")


def _capsule(p, q, rad):
    pr, pc = p; qr, qc = q
    vr, vc = qr - pr, qc - pc
    L2 = vr * vr + vc * vc
    t = np.clip(((_R - pr) * vr + (_C - pc) * vc) / L2, 0, 1) if L2 > 0 else np.zeros_like(_R)
    return (_R - (pr + t * vr)) ** 2 + (_C - (pc + t * vc)) ** 2 <= rad * rad


def coverage(shape, s, center=None, arm=None, thickness=3.0):
    if shape == "disc":
        m = (_R - center[0]) ** 2 + (_C - center[1]) ** 2 <= (s / 2) ** 2
    elif shape == "square":
        m = (np.abs(_R - center[0]) <= s / 2) & (np.abs(_C - center[1]) <= s / 2)
    elif shape == "arm":
        b, e, t = arm
        m = _capsule(b, e, thickness / 2) | _capsule(e, t, thickness / 2)
    else:
        raise ValueError(shape)
    return m.reshape(W, SS, W, SS).mean((1, 3))


def compose(bg, cov):
    img = bg * (1 - cov[..., None]) + AGENT_RGB * cov[..., None]
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def arm_pose(base, u, s, d):
    """2-link arm, links s/2. Endpoint moves from base + s*u along the line through the base by d.
    Returns (base, elbow, tip). Elbow side kept continuous through the fold at d = s."""
    L = s / 2
    b = np.asarray(base, float)
    r = abs(s - d)
    alpha = np.arccos(np.clip(r / (2 * L), -1, 1))
    if d <= s:
        dirv, sgn = u, 1.0
    else:
        dirv, sgn = -u, -1.0
    ca, sa = np.cos(sgn * alpha), np.sin(sgn * alpha)
    rot = np.array([dirv[0] * ca - dirv[1] * sa, dirv[0] * sa + dirv[1] * ca])
    elbow = b + L * rot
    tip = b + (s - d) * u
    return b, elbow, tip
