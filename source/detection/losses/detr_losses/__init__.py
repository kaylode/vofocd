from typing import Dict, List
import torch
from torch import nn
from theseus.base.utilities.cuda import move_to
from .base import SetCriterion
from .matcher import HungarianMatcher


class DETRLosses(nn.Module):
    r"""CELoss is warper of cross-entropy loss"""

    def __init__(self, num_classes, loss_ce=1, loss_bbox=5, loss_giou=2, loss_mask=None, loss_dice=None, cost_class=1, cost_bbox=5, cost_giou=2, eos_coef=0.1, dec_layers=6,**kwargs):
        super().__init__()
        losses = ['labels', 'boxes', 'cardinality']

        self.weight_dict = {
            'loss_ce': loss_ce, 
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou,
        }

        aux_weight_dict = {}
        for i in range(dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in self.weight_dict.items()})
        self.weight_dict.update(aux_weight_dict)

        matcher = HungarianMatcher(
            cost_class=cost_class, 
            cost_bbox=cost_bbox, 
            cost_giou=cost_giou
        )
        
        if loss_mask is not None and loss_dice is not None:
            self.weight_dict.update({
                'loss_mask': loss_mask,
                'loss_dice': loss_dice
            })

        self.criterion = SetCriterion(num_classes, matcher, self.weight_dict, eos_coef=eos_coef, losses=losses)

    def forward(self, outputs: Dict, batch: Dict, device: torch.device) -> torch.Tensor:
        pred = outputs["outputs"]
        target = move_to(batch["targets"], device)

        loss_dict = self.criterion(pred, target)
        losses = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)
        loss_dict = {k:v.item() for k,v in loss_dict.items()}
        loss_dict.update({"T": losses.item()})
        return losses, loss_dict