from typing import Any, Dict
import torch
import torch.nn as nn

class TemperatureScaledKLDivLoss(nn.Module):
    """
    Temperature scaled Kullback-Leibler divergence loss for knowledge distillation (Hinton et al., 
    https://arxiv.org/abs/1503.02531)
    
    :param float temperature: parameter for softening the distribution to be compared.
    """

    def __init__(self, temperature, **kwargs):
        super(TemperatureScaledKLDivLoss, self).__init__()
        self.temperature = temperature
        self.kullback_leibler_divergence = nn.KLDivLoss(reduction="batchmean")

    def forward(self, outputs: Dict[str, Any], batch: Dict[str, Any], device: torch.device):
        """
        Output the temperature scaled Kullback-Leibler divergence loss for given the prediction and the target.
        """

        # KD Loss
        student_outputs = outputs["student_outputs"]['outputs']
        teacher_outputs = outputs["teacher_outputs"]['outputs']

        log_p = torch.log_softmax(student_outputs / self.temperature, dim=1)
        q = torch.softmax(teacher_outputs / self.temperature, dim=1)

        # Note that the Kullback-Leibler divergence is re-scaled by the squared temperature parameter.
        loss = (self.temperature ** 2) * self.kullback_leibler_divergence(log_p, q)
        loss_dict = {
            'KLDiv': loss.item()
        }
        return loss, loss_dict