from typing import Any, Dict
import torch
from torch import nn
from theseus.utilities.cuda import detach

class MedTEXCELoss(nn.Module):
    r"""CELoss is warper of cross-entropy loss"""

    def __init__(self, **kwargs):
        super(MedTEXCELoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, outputs: Dict[str, Any], batch: Dict[str, Any], device: torch.device):
        student_outputs = outputs["student_outputs"]['outputs']
        teacher_outputs = outputs["teacher_outputs"]['outputs']
        teacher_pred = torch.argmax(detach(teacher_outputs), dim=1)
        celoss = self.ce_loss(student_outputs, teacher_pred.view(-1).contiguous()) 
        loss_dict = {
            "CE_Tch": celoss.item(),
        }
        return celoss, loss_dict