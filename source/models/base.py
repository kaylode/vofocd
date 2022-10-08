from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
from theseus.utilities.cuda import move_to, detach
from theseus.classification.utilities.logits import logits2labels

from .backbone.convnext import model_factory

class MedTEX(nn.Module):
    """Convolution models from timm
    
    name: `str`
        timm model name
    num_classes: `int`
        number of classes
    from_pretrained: `bool` 
        whether to use timm pretrained
    classnames: `Optional[List]`
        list of classnames
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
        super().__init__()

        self.name = model_name
        self.classnames = classnames
        self.num_classes = num_classes
        self.freeze = freeze

        self.model = model_factory[model_name](
            pretrained=pretrained,
            num_classes=self.num_classes
        )

        self.feature_dims = self.model.feature_dims

        if self.freeze:
            self.freeze_backbone()

    def freeze_backbone(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def get_model(self):
        """
        Return the full architecture of the model, for visualization
        """
        return self.model

    def forward(self, batch: Dict, device: torch.device):
        x = move_to(batch['inputs'], device)
        outputs, features = self.model(x, return_features=True)

        return {
            'outputs': outputs,
            'inter_features': features
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
