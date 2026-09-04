# HWM_PLDM 원본 대비 추가된 실험 코드 (fork: slcnvly/HWM_PLDM, branch: lang-align).
#
# 캡션 어휘(고정, K개)를 문장 임베딩 테이블 (K, embed_dim)로 매핑한다.
#   - mode="frozen": 생성 시 1회만 인코딩(no_grad) 후 버퍼로 캐싱. 학습 중 HF 호출 없음.
#   - mode="finetune": 인코더를 서브모듈로 들고 매 forward마다 K개 캡션을 다시 인코딩
#     (K가 작으므로 비용 무시 가능). VL-JEPA를 따라 매우 낮은 LR로 함께 학습한다.
#
# sentence-transformers 패키지 대신 transformers의 AutoModel/AutoTokenizer + 수동
# mean-pooling을 쓴다 — SentenceTransformer.encode()는 버전에 따라 내부적으로
# torch.no_grad()를 강제해 파인튜닝 시 그래디언트가 끊길 수 있어 이를 피하기 위함.
import os
from typing import List, Optional

import torch
import torch.nn as nn


def _mean_pooling(
    token_embeddings: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    summed = (token_embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class CaptionEmbedder(nn.Module):
    def __init__(
        self,
        vocab: List[str],
        mode: str = "frozen",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_path: Optional[str] = None,
        max_length: int = 32,
    ):
        super().__init__()
        assert mode in ("frozen", "finetune"), f"unknown mode {mode!r}"
        self.mode = mode
        self.vocab = list(vocab)
        self._max_length = max_length
        self._backbone = None
        self._tokenizer = None

        if cache_path is not None and os.path.exists(cache_path):
            if mode == "finetune":
                raise ValueError(
                    "cache_path is only valid with mode='frozen' "
                    "(finetune needs a live encoder to backprop through)"
                )
            proto = torch.load(cache_path, map_location="cpu")
            assert proto.shape[0] == len(self.vocab), (
                f"cached embedding table has {proto.shape[0]} rows, "
                f"vocab has {len(self.vocab)} entries"
            )
            self.embed_dim = proto.shape[1]
            self.register_buffer("_frozen_proto", proto)
            return

        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._backbone = AutoModel.from_pretrained(model_name)
        self.embed_dim = self._backbone.config.hidden_size

        if mode == "frozen":
            self._backbone.eval()
            for p in self._backbone.parameters():
                p.requires_grad = False
            with torch.no_grad():
                proto = self._encode(self.vocab)
            self.register_buffer("_frozen_proto", proto)
            # done with the live encoder now that the prototype table is cached.
            self._backbone = None
            self._tokenizer = None

    def _encode(self, texts: List[str]) -> torch.Tensor:
        device = next(self._backbone.parameters()).device
        batch = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        ).to(device)
        out = self._backbone(**batch)
        return _mean_pooling(out.last_hidden_state, batch["attention_mask"])

    def forward(self, warm: bool = False) -> torch.Tensor:
        """Returns (K, embed_dim) prototype embeddings for the fixed vocabulary.

        `warm`: if True (finetune mode only), run the encoder under no_grad — used
        by the caller during an initial warmup window before the text encoder's
        gradients are allowed to flow (mirrors VL-JEPA's stabilization heuristic).
        The step counting itself lives in LangAlignObjective, not here, since this
        class has no visibility into whether the current call is train or eval.
        """
        if self.mode == "frozen":
            return self._frozen_proto
        if warm:
            with torch.no_grad():
                return self._encode(self.vocab)
        return self._encode(self.vocab)

    def param_groups(self, base_lr: float, lr_scale: float = 0.05) -> List[dict]:
        if self.mode != "finetune":
            return []
        scaled = base_lr * lr_scale
        # NB: pldm.optimizers.schedulers.Scheduler recomputes "lr" from "base_lr"
        # (falling back to the optimizer-wide base_lr if "base_lr" is absent) on
        # every step, for both Constant and Cosine schedules. Without setting
        # "base_lr" here explicitly, this group's lr would get silently
        # overwritten to the model's base_lr on the very first scheduler step.
        return [{"params": list(self._backbone.parameters()), "lr": scaled, "base_lr": scaled}]
