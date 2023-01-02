from typing import Any, Dict
import torch
import torch.nn as nn


class GaussianLoss(nn.Module):
    """
    Gaussian loss for transfer learning with variational information distillation.    
    """
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(self, outputs: Dict[str, Any], batch: Dict[str, Any], device: torch.device):
        """
        Output the Gaussian loss given the prediction and the target.
        :param tuple(torch.Tensor, torch.Tensor) y_pred: predicted mean and variance for the Gaussian 
        distribution.
        :param torch.Tensor y: target for the Gaussian distribution.
        """
        # KD Loss
        student_variances = outputs["student_outputs"]['variances']
        student_features = outputs["student_outputs"]['inter_features']
        teacher_features = outputs["teacher_outputs"]['inter_features']

        loss = 0
        for (st_feat, st_var, tch_feat) in zip(student_features, student_variances, teacher_features):
            loss += torch.mean(0.5 * ((st_feat - tch_feat) ** 2 / st_var + torch.log(st_var)))

        loss_dict = {
            'GS': loss.item()
        }
        return loss, loss_dict