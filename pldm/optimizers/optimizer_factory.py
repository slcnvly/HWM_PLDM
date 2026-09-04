from typing import List, Optional

import torch
from pldm.optimizers.lars import LARS, exclude_bias_and_norm
import enum


class OptimizerType(enum.Enum):
    Adam = "sgd"
    LARS = "lars"


class OptimizerFactory:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_type: str,
        base_lr: float,
        l1_to_l2_lr_ratio: float,
        # added for lang-align branch: objectives (e.g. LangAlignObjective) can own
        # their own learnable modules (projection heads, a finetuned text encoder).
        # Without this, those parameters are silently never registered with the
        # optimizer and never train -- the same pre-existing gap that VICReg's
        # Projector has, just made explicit here since LangAlign relies on it.
        extra_param_groups: Optional[List[dict]] = None,
    ):
        self.model = model
        self.optimizer_type = optimizer_type
        self.base_lr = base_lr
        self.l1_to_l2_lr_ratio = l1_to_l2_lr_ratio
        self.extra_param_groups = extra_param_groups or []

    def create_optimizer(self):
        if self.optimizer_type == OptimizerType.LARS:
            lars_groups = [{"params": self.model.parameters(), "lr": 0}]
            # added for lang-align branch
            lars_groups += self.extra_param_groups
            optimizer = LARS(
                lars_groups,
                lr=0,
                weight_decay=1e-6,
                weight_decay_filter=exclude_bias_and_norm,
                lars_adaptation_filter=exclude_bias_and_norm,
            )
        elif self.optimizer_type == OptimizerType.Adam:

            models = {
                "level1": self.model.level1,
                "level2": self.model.level2,
            }

            params_list = []

            for level, model in models.items():
                if model is None:
                    continue

                if level == "level1" and self.model.level2 is not None:
                    lr = self.base_lr * self.l1_to_l2_lr_ratio
                else:
                    lr = self.base_lr

                params_list.append(
                    {
                        "params": model.parameters(),
                        "lr": lr,
                    }
                )

                if model.predictor.ensemble_params is not None:
                    params_list.append(
                        {
                            "params": model.predictor.ensemble_params.values(),
                            "lr": lr,
                        }
                    )

            params_list += self.extra_param_groups  # added for lang-align branch

            optimizer = torch.optim.Adam(
                params_list,
                weight_decay=1e-6,
            )
        else:
            raise NotImplementedError("Unknown optimizer type")

        return optimizer
