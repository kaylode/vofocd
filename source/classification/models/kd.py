from typing import Dict, List, Any, Optional
import torch
from torch import nn
from theseus.utilities.cuda import move_to, detach
from theseus.classification.utilities.logits import logits2labels
from .med_tex import MedTEXTeacher

class KDFramework(nn.Module):
    """Add utilitarian functions for module to work with pipeline
    Args:
        model (Module): Base Model without loss
        loss (Module): Base loss function with stat
    """

    def __init__(
        self, 
        classnames: str = None,
        num_classes: int = 1000,
        **kwargs):

        super().__init__()
        self.teacher = MedTEXTeacher(
            'convnext_small',
            num_classes=num_classes,
            classnames=classnames,
            freeze=True
        )

        self.teacher.eval()

        self.student = MedTEXTeacher(
            'convnext_nano',
            num_classes=num_classes,
            classnames=classnames,
            freeze=False,
        )

        self.num_classes = num_classes
        self.classnames = classnames

    def get_model(self):
        """
        Return the full architecture of the model, for visualization
        """
        return self.student
    
    def forward(self, batch: Dict, device: torch.device):

        student_output_dict = self.student(batch, device)
        student_outputs, _ = student_output_dict['outputs'], student_output_dict['inter_features']

        with torch.no_grad():
            teacher_output_dict = self.teacher(batch, device)

        return {
            'outputs': student_outputs,
            'student_outputs': {
                'outputs': student_output_dict['outputs'],
            },
            'teacher_outputs': {
                'outputs': teacher_output_dict['outputs'],
            }
        } 

    def get_prediction(self, adict: Dict[str, Any], device: torch.device):
        """
        Inference using the model.

        adict: `Dict[str, Any]`
            dictionary of inputs
        device: `torch.device`
            current device 
        """
        outputs = self.forward(adict, device)['outputs']

        if not adict.get('multilabel'):
            outputs, probs = logits2labels(outputs, label_type='multiclass', return_probs=True)
        else:
            outputs, probs = logits2labels(outputs, label_type='multilabel', threshold=adict['threshold'], return_probs=True)

        probs = move_to(detach(probs), torch.device('cpu')).numpy()
        classids = move_to(detach(outputs), torch.device('cpu')).numpy()

        if self.classnames and not adict.get('multilabel'):
            classnames = [self.classnames[int(clsid)] for clsid in classids]
        elif self.classnames and adict.get('multilabel'):
            classnames = [
              [self.classnames[int(i)] for i, c in enumerate(clsid) if c]
              for clsid in classids
            ]
        else:
            classnames = []

        return {
            'labels': classids,
            'confidences': probs, 
            'names': classnames,
        }