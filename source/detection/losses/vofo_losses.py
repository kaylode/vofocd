from typing import Any, Dict

import torch
from torch import nn
from .detr_losses import DETRLosses

class VOFOMultiLoss(nn.Module):
    def __init__(self, cls_loss: nn.Module, detr_loss: DETRLosses, **kwargs) -> None:
        super().__init__()
        self.cls_loss = cls_loss
        self.detr_loss = detr_loss

    def forward(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        device: torch.device,
    ):
        
        global_loss, global_loss_dict = self.cls_loss(
            outputs=outputs["img_outputs"],
            batch=batch["img_targets"],
            device=device
        )

        local_loss, local_loss_dict = self.detr_loss(
            outputs=outputs["obj_outputs"],
            batch=batch["obj_targets"],
            device=device
        )

        total_loss = local_loss + global_loss
        global_loss_dict.update(local_loss_dict)

        return total_loss, global_loss_dict

