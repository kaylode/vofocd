from typing import Dict, List, Any, Optional
import torch
from torch import nn
from theseus.base.utilities.cuda import move_to, detach
from theseus.base.utilities.logits import logits2labels

from source.classification.models.base import MedTEX
from source.classification.models.explainer import ConvAutoencoder, NestedUNet
from .subnetwork import Subnetwork

"""
Given an input image X, the explainer generates an importance score for each of its pixels, where its last layer is 1×1
convolution layer with sigmoid activation.

X' is generated by element-wise multiplication.


additional loss to maximize the mutual
information between the outputs of each i
th intermediate layer
of the teacher and the student  

Recall that the output of the i
th layer of the teacher is
a Ci × Hi × Wi feature map

subnetwork with 1 × 1 convolutional layers to
match the channel dimensions between Teacher and Student

4 block CNN layers, where each block consists of a convolutional layer,
batch normalization, maxpooling and ReLU activation.

. For the explainer,
we adopt auto encoder with skip connections.
"""


class MedTEXStudent(MedTEX):
    """
    Student with AutoEncoder
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = False,
        num_classes: int = 1000,
        classnames: Optional[List] = None,
        freeze: bool = False,
        use_explainer: bool = True,
        **kwargs
    ):
        super().__init__(model_name, pretrained, num_classes, classnames, freeze)

        self.use_explainer = use_explainer
        if self.use_explainer:
            self.auto_encoder = NestedUNet(
                in_ch=3, out_ch=1
            )

        # self.auto_encoder = ConvAutoencoder()

    def forward(self, batch: Dict, device: torch.device):
        x = move_to(batch['inputs'], device)

        if self.use_explainer:
            pixel_map = self.auto_encoder(x)
            x = pixel_map*x

        outputs, features = self.model(x, return_features=True)

        if self.use_explainer:
            return {
                'outputs': outputs, 
                'inter_features': features,
                'pixel_maps': pixel_map.detach()
            }
        else:
            return {
                'outputs': outputs, 
                'inter_features': features
            }

class MedTEXTeacher(MedTEX):
    """
    Student with AutoEncoder
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = False,
        num_classes: int = 1000,
        classnames: Optional[List] = None,
        freeze: bool = False,
        **kwargs
    ):
        super().__init__(model_name, pretrained, num_classes, classnames, freeze)

    def forward(self, batch: Dict, device: torch.device):
        x = move_to(batch['inputs'], device)
        outputs, features = self.model(x, return_features=True)
        return {
            'outputs': outputs, 
            'inter_features': features
        }

class MedTEXFramework(nn.Module):
    """Add utilitarian functions for module to work with pipeline
    Args:
        model (Module): Base Model without loss
        loss (Module): Base loss function with stat
    """

    def __init__(
        self, 
        classnames: str = None,
        num_classes: int = 1000,
        student_explainer: bool = True,
        **kwargs):

        super().__init__()
        self.teacher = MedTEXTeacher(
            'convnext_small',
            num_classes=num_classes,
            classnames=classnames,
            freeze=True
        )

        self.teacher.eval()

        self.student = MedTEXStudent(
            'convnext_nano',
            num_classes=num_classes,
            classnames=classnames,
            freeze=False,
            use_explainer=student_explainer
        )

        self.subnetwork = Subnetwork(self.student.feature_dims, self.teacher.feature_dims)

        self.num_classes = num_classes
        self.classnames = classnames

    def get_model(self):
        """
        Return the full architecture of the model, for visualization
        """
        return self.student
    
    def forward(self, batch: Dict, device: torch.device):

        student_output_dict = self.student(batch, device)
        student_outputs, student_features = student_output_dict['outputs'], student_output_dict['inter_features']

        with torch.no_grad():
            teacher_output_dict = self.teacher(batch, device)
            teacher_outputs, teacher_features = teacher_output_dict['outputs'], teacher_output_dict['inter_features']

        mapped_student_features, student_variances = self.subnetwork(student_features)

        return {
            'outputs': student_outputs,
            'pixel_maps': student_output_dict['pixel_maps'],
            'student_outputs': {
                'outputs': student_outputs,
                'inter_features': mapped_student_features,
                'variances': student_variances
            },
            'teacher_outputs': {
                'outputs': teacher_outputs,
                'inter_features': teacher_features,
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
        output_dict = self.forward(adict, device)
        outputs = output_dict['outputs']
        pixel_maps = output_dict['pixel_maps']

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
            'pixel_maps': pixel_maps,
        }