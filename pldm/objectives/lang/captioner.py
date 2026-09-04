# HWM_PLDM 원본 대비 추가된 실험 코드 (fork: slcnvly/HWM_PLDM, branch: lang-align).
#
# 웨이포인트 구간(level2 청크)의 이동을 규칙 기반으로 캡션화한다. 좌표 차이만 보고
# "정지" 또는 "8방위 x {짧은/긴} 거리"로 분류하므로 어휘가 작다(기본 설정 K=17).
# 문자열은 build_vocabulary()에서 1회만 생성되고, 학습 루프에서는 segment_caption_ids()가
# 정수 id만 반환한다 (런타임 문자열 생성 없음).
from typing import List

import torch

# id 0: 정지. id 1..N: "짧은 거리 + 방향". id N+1..2N: "긴 거리 + 방향".
_COMPASS_NAMES = [
    "east",
    "northeast",
    "north",
    "northwest",
    "west",
    "southwest",
    "south",
    "southeast",
]


def _direction_names(n_directions: int) -> List[str]:
    if n_directions == len(_COMPASS_NAMES):
        return _COMPASS_NAMES
    return [f"direction_{i}" for i in range(n_directions)]


def build_vocabulary(mag_bins: List[float], n_directions: int = 8) -> List[str]:
    """Return the fixed list of caption strings, indexed by caption id.

    Vocabulary size K = 1 + 2 * n_directions (stationary + {short,long} x directions).
    `mag_bins` must have exactly 2 thresholds: [stationary_cutoff, short_long_cutoff].
    """
    assert len(mag_bins) == 2, "mag_bins must be [stationary_cutoff, short_long_cutoff]"
    dir_names = _direction_names(n_directions)

    vocab = ["stay in place"]
    for size in ("a short distance", "a long distance"):
        for name in dir_names:
            vocab.append(f"move {size} to the {name}")
    return vocab


def segment_caption_ids(
    l2_locations: torch.Tensor,
    location_std: torch.Tensor,
    mag_bins: List[float],
    n_directions: int = 8,
) -> torch.Tensor:
    """Map waypoint-anchor coordinates to caption ids.

    Args:
        l2_locations: (B, n_segments+1, 2), normalized coordinates as stored in
            `D4RLSample.l2_locations` (see pldm_envs/diverse_maze/d4rl.py).
        location_std: (2,) — the normalizer's location std (mean cancels in the diff).
        mag_bins: [stationary_cutoff, short_long_cutoff], in block units.
        n_directions: number of compass sectors (default 8, 45 degrees each).

    Returns:
        LongTensor (B, n_segments) of caption ids in [0, 1 + 2*n_directions).
    """
    assert len(mag_bins) == 2
    std = location_std.to(l2_locations.device).to(l2_locations.dtype)
    d = (l2_locations[:, 1:] - l2_locations[:, :-1]) * std  # (B, n_segments, 2)

    magnitude = d.norm(dim=-1)
    angle_deg = torch.rad2deg(torch.atan2(d[..., 1], d[..., 0])) % 360.0

    sector_width = 360.0 / n_directions
    shifted = (angle_deg + sector_width / 2.0) % 360.0
    dir_idx = (shifted // sector_width).long() % n_directions

    boundaries = torch.tensor(mag_bins, device=d.device, dtype=d.dtype)
    mag_bucket = torch.bucketize(magnitude, boundaries)  # 0=stationary,1=short,2=long

    stationary = mag_bucket == 0
    caption_ids = torch.where(
        stationary,
        torch.zeros_like(dir_idx),
        1 + (mag_bucket - 1).clamp(min=0) * n_directions + dir_idx,
    )
    return caption_ids
