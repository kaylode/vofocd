from typing import Dict, Any
from theseus.base.utilities.cuda import move_to, detach
from theseus.base.utilities.logits import logits2labels
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict

class Resnet(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=1000, classnames=None, **kwargs) -> None:
        super().__init__()
        self.model = getattr(torchvision.models, model_name)()
        self.classnames = classnames
        self.num_classes = num_classes

        in_features = self.model.fc.in_features
            # add more layers as required
        classifier = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(in_features, num_classes))
        ]))

        self.model.fc = classifier

    def get_model(self):
        return self.model

    def forward(self, x):
        return self.model(x)

    def forward_batch(self, batch , device):
        inputs = move_to(batch['inputs'], device)
        out = self.forward(inputs)
        return {'outputs': out}

    def get_prediction(self, adict: Dict[str, Any], device: torch.device):
        """
        Inference using the model.
        adict: `Dict[str, Any]`
            dictionary of inputs
        device: `torch.device`
            current device
        """
        outputs = self.forward_batch(adict, device)["outputs"]

        if not adict.get("multilabel"):
            outputs, probs = logits2labels(
                outputs, label_type="multiclass", return_probs=True
            )
        else:
            outputs, probs = logits2labels(
                outputs,
                label_type="multilabel",
                threshold=adict["threshold"],
                return_probs=True,
            )

            if adict.get("no-zeroes"):
                argmaxs = torch.argmax(probs, dim=1)
                tmp = torch.sum(outputs, dim=1)
                one_hots = F.one_hot(argmaxs, outputs.shape[1])
                outputs[tmp == 0] = one_hots[tmp == 0].bool()

        probs = move_to(detach(probs), torch.device("cpu")).numpy()
        classids = move_to(detach(outputs), torch.device("cpu")).numpy()

        if self.classnames and not adict.get("multilabel"):
            classnames = [self.classnames[int(clsid)] for clsid in classids]
        elif self.classnames and adict.get("multilabel"):
            classnames = [
                [self.classnames[int(i)] for i, c in enumerate(clsid) if c]
                for clsid in classids
            ]
        else:
            classnames = []

        return {
            "labels": classids,
            "confidences": probs,
            "names": classnames,
        }