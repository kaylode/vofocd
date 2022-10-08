from typing import Any, Dict
import torch
from torch import nn
from theseus.utilities.cuda import detach

class MedTEXLoss(nn.Module):
    r"""CELoss is warper of cross-entropy loss"""

    def __init__(self, lambd = 0.001, **kwargs):
        super(MedTEXLoss, self).__init__()
        self.lambd = lambd
        self.ce_loss = nn.CrossEntropyLoss()
        self.kdcriterion = nn.KLDivLoss()

    def forward(self, outputs: Dict[str, Any], batch: Dict[str, Any], device: torch.device):
        
        # KD Loss
        student_outputs = outputs["student_outputs"]['outputs']
        teacher_outputs = outputs["teacher_outputs"]['outputs']
        teacher_pred = torch.argmax(detach(teacher_outputs), dim=1)

        celoss = self.ce_loss(student_outputs, teacher_pred.view(-1).contiguous()) 

        # Intermediate layers loss
        student_features = outputs["student_outputs"]['inter_features']
        teacher_features = outputs["teacher_outputs"]['inter_features']

        inter_loss = 0
        for st_ft, tch_ft in zip(student_features, teacher_features):
            inter_loss += self.kdcriterion(st_ft, tch_ft)

        total_loss = celoss + self.lambd* inter_loss

        loss_dict = {
             "CE": celoss.item(),
           "Inter": inter_loss.item(),
            "T": total_loss.item()
        }
        return total_loss, loss_dict