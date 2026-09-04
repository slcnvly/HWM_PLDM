# HWM_PLDM 원본 대비 추가된 실험 코드 (fork: slcnvly/HWM_PLDM, branch: lang-align).
#
# LangAlignObjective: level2 고수준 잠재 행동 l_t(posterior_mu)를 웨이포인트 구간의
# 규칙 기반 캡션 임베딩과 정렬하는 보조 대조 손실. 기존 objectives(prediction.py,
# vicreg.py, kl.py)와 동일한 호출 관례를 따른다: torch.nn.Module,
# __call__(batch, results: List[ForwardResult]) -> LangAlignLossInfo(NamedTuple),
# results[-1]만 사용, 반환값은 스칼라 텐서/문자열 필드만 가진 NamedTuple.
#
# 설계상 중요한 결정 (계획 문서 참고):
#   - 캡션 어휘가 작아(K≈17) 배치 안에서 같은 캡션이 여러 세그먼트에 붙는 게 흔하다.
#     표준 CLIP-style InfoNCE를 그대로 쓰면 같은 캡션끼리 서로 false negative가 되어
#     학습을 방해한다. 그래서:
#       z -> text 방향: K-way softmax cross-entropy (텍스트 쪽 후보가 K개뿐이므로
#         "프로토타입 분류"와 수학적으로 동일 — 근사가 아니라 항등식).
#       text -> z 방향: SupCon (같은 캡션을 가진 모든 z를 positive로 취급, 배치에
#         없는 캡션은 스킵).
#   - text_dispersion: K개 텍스트 프로토타입 간 off-diagonal cosine 유사도 평균에
#     대한 페널티. finetune 모드에서 텍스트 인코더가 전부 같은 지점으로 collapse하는
#     것을 억제한다 (K가 작아 비용 무시 가능).
from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional

import torch
import torch.nn.functional as F

from pldm.configs import ConfigBase
from pldm.models.jepa import ForwardResult
from pldm.models.misc import build_mlp
from pldm.objectives.lang.captioner import build_vocabulary, segment_caption_ids
from pldm.objectives.lang.text_encoder import CaptionEmbedder


class LangAlignLossInfo(NamedTuple):
    total_loss: torch.Tensor
    z2t_loss: torch.Tensor
    t2z_loss: torch.Tensor
    text_offdiag_cossim: torch.Tensor
    z2t_acc: torch.Tensor
    loss_name: str = "lang_align"
    name_prefix: str = ""

    def build_log_dict(self):
        prefix = f"{self.name_prefix}/{self.loss_name}"
        return {
            prefix: self.total_loss.item(),
            f"{prefix}_z2t": self.z2t_loss.item(),
            f"{prefix}_t2z": self.t2z_loss.item(),
            f"{prefix}_z2t_acc": self.z2t_acc.item(),
            f"{prefix}_text_offdiag_cossim": self.text_offdiag_cossim.item(),
        }


@dataclass
class LangAlignObjectiveConfig(ConfigBase):
    global_coeff: float = 0.1
    mode: str = "frozen"  # or "finetune"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    cache_path: Optional[str] = None
    text_lr_scale: float = 0.05
    freeze_text_steps: int = 500
    proj_dim: int = 32
    z_proj_arch: str = "64"
    t_proj_arch: str = "256"
    temperature: float = 0.07
    learnable_temp: bool = True
    max_logit_scale: float = 4.6052  # ln(100), CLIP's cap
    mag_bins: List[float] = field(default_factory=lambda: [0.25, 0.9])
    n_directions: int = 8
    t2z_coeff: float = 1.0
    text_dispersion_coeff: float = 1.0


class LangAlignObjective(torch.nn.Module):
    def __init__(
        self,
        config: LangAlignObjectiveConfig,
        z_dim: int,
        location_std: torch.Tensor,
        name_prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.name_prefix = name_prefix
        self.register_buffer("location_std", location_std.float())

        self.vocab = build_vocabulary(config.mag_bins, config.n_directions)
        self.text_embedder = CaptionEmbedder(
            vocab=self.vocab,
            mode=config.mode,
            model_name=config.model_name,
            cache_path=config.cache_path,
        )

        self.head_z = build_mlp(
            config.z_proj_arch, input_dim=z_dim, output_shape=config.proj_dim, norm=None
        )
        self.head_t = build_mlp(
            config.t_proj_arch,
            input_dim=self.text_embedder.embed_dim,
            output_shape=config.proj_dim,
            norm=None,
        )

        init_logit_scale = torch.log(torch.tensor(1.0 / config.temperature))
        if config.learnable_temp:
            self.logit_scale = torch.nn.Parameter(init_logit_scale)
        else:
            self.register_buffer("logit_scale", init_logit_scale)

        # Only advances under grad-enabled (i.e. training) calls -- eval passes run
        # under torch.no_grad() and must not consume the finetune warmup budget.
        self._step = 0

        # Objectives in this repo are plain standalone objects, not submodules of
        # self.model, so the trainer's `self.model = self.model.cuda()` never
        # touches them -- every existing objective with learnable params
        # (VICReg's Projector, IDM, Probe) explicitly .cuda()s itself; follow the
        # same convention here.
        self.cuda()

    def param_groups(self, base_lr: float) -> List[dict]:
        # NB: "base_lr" must be set alongside "lr" -- see the comment in
        # CaptionEmbedder.param_groups for why (Scheduler overwrites "lr" from
        # "base_lr" every step, defaulting to the model-wide base_lr otherwise).
        groups = [
            {
                "params": list(self.head_z.parameters()) + list(self.head_t.parameters()),
                "lr": base_lr,
                "base_lr": base_lr,
            },
        ]
        if self.config.learnable_temp:
            groups.append({"params": [self.logit_scale], "lr": base_lr, "base_lr": base_lr})
        groups += self.text_embedder.param_groups(base_lr, self.config.text_lr_scale)
        return groups

    def _zero_loss(self, device) -> LangAlignLossInfo:
        z = torch.zeros((), device=device)
        return LangAlignLossInfo(
            total_loss=z,
            z2t_loss=z,
            t2z_loss=z,
            text_offdiag_cossim=z,
            z2t_acc=z,
            name_prefix=self.name_prefix,
        )

    def __call__(self, batch, results: List[ForwardResult]) -> LangAlignLossInfo:
        result = results[-1]
        posteriors = result.pred_output.posteriors  # (T, B, z_dim) or None
        if posteriors is None:
            return self._zero_loss(self.location_std.device)

        if torch.is_grad_enabled():
            self._step += 1
        warm = self._step <= self.config.freeze_text_steps

        T, B, z_dim = posteriors.shape
        z = posteriors.permute(1, 0, 2).reshape(B * T, z_dim)  # (B*T, z_dim)

        l2_locations = batch.l2_locations.to(z.device)
        caption_ids = segment_caption_ids(
            l2_locations, self.location_std, self.config.mag_bins, self.config.n_directions
        )  # (B, T)
        assert caption_ids.shape == (B, T), (
            f"caption_ids shape {tuple(caption_ids.shape)} != expected {(B, T)} "
            "-- l2_locations/posteriors segment count mismatch"
        )
        caption_ids = caption_ids.reshape(B * T)

        proj_z = F.normalize(self.head_z(z), dim=-1)  # (N, proj_dim)

        text_proto = self.text_embedder(warm=warm)  # (K, embed_dim)
        proj_t = F.normalize(self.head_t(text_proto), dim=-1)  # (K, proj_dim)

        logit_scale = self.logit_scale.exp().clamp(max=self.config.max_logit_scale)
        sim = proj_z @ proj_t.t() * logit_scale  # (N, K)

        # z -> text: exact K-way classification against the caption id.
        z2t_loss = F.cross_entropy(sim, caption_ids)
        z2t_acc = (sim.argmax(dim=-1) == caption_ids).float().mean()

        # text -> z: SupCon over whichever captions are actually present in this batch.
        K = proj_t.shape[0]
        log_prob_t2z = F.log_softmax(sim.t(), dim=-1)  # (K, N)
        pos_mask = caption_ids.unsqueeze(0) == torch.arange(K, device=z.device).unsqueeze(1)
        pos_counts = pos_mask.sum(dim=1)  # (K,)
        present = pos_counts > 0
        if present.any():
            summed_logprob = (log_prob_t2z * pos_mask).sum(dim=1)  # (K,)
            per_caption_loss = -summed_logprob[present] / pos_counts[present].clamp(min=1)
            t2z_loss = per_caption_loss.mean()
        else:
            t2z_loss = torch.zeros((), device=z.device)

        proto_sim = proj_t @ proj_t.t()  # (K, K)
        off_diag = ~torch.eye(K, dtype=torch.bool, device=z.device)
        text_offdiag_cossim = proto_sim[off_diag].mean()
        dispersion_penalty = text_offdiag_cossim.clamp(min=0)

        align_loss = 0.5 * (z2t_loss + self.config.t2z_coeff * t2z_loss)
        total_loss = (
            self.config.global_coeff * align_loss
            + self.config.text_dispersion_coeff * dispersion_penalty
        )

        return LangAlignLossInfo(
            total_loss=total_loss,
            z2t_loss=z2t_loss,
            t2z_loss=t2z_loss,
            text_offdiag_cossim=text_offdiag_cossim,
            z2t_acc=z2t_acc,
            name_prefix=self.name_prefix,
        )
