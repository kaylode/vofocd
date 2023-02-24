from typing import Dict, List, Any, Optional
# import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from theseus.base.utilities.cuda import move_to, detach
import torchvision
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2
from .detr_utils import box_ops

class FasterRCNN(nn.Module):
    """DocString"""

    def __init__(
        self,
        model_name: str,
        num_classes: int=6,
        weights: str='DEFAULT',
        classnames: Optional[List] = None,
        weights_backbone: Optional[torchvision.models.resnet.ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V2,
        **kwargs
    ):
        super().__init__()
        self.name = model_name
        self.num_classes = num_classes
        self.weights = weights
        self.classnames = classnames
        self.weights_backbone = weights_backbone
        self.model = fasterrcnn_resnet50_fpn_v2(
            num_classes=num_classes,
            weights_backbone=weights_backbone,
        )

    def get_model(self):
        """
        Return the full architecture of the model, for visualization
        """
        return self.model
    
    def forward_batch(self, batch: Dict, device: torch.device, is_train=False):
        x = move_to(batch['inputs'], device)
        outputs = None
        loss = -1
        loss_dict = {}
        
        if is_train:
            self.model.train()
            y = move_to(batch['targets'], device)
            loss_dict = self.model(x, y)
            loss = sum(loss for loss in loss_dict.values())
            loss_dict = {k:move_to(detach(v), torch.device('cpu')) for k,v in loss_dict.items()}
            
        else:
            self.model.eval()
            outputs = self.model(x)
        return {'outputs': outputs}, loss, loss_dict

    @torch.no_grad()
    def get_prediction(self, adict: Dict[str, Any], device: torch.device):
        """
        Inference using the model.
        adict: `Dict[str, Any]`
            dictionary of inputs
        device: `torch.device`
            current device 

        box format: xyxy
        """
        outputs, _, _ = self.forward_batch(adict, device, is_train=False)
        batch_size = len(outputs['outputs'])

        results = outputs['outputs']

        scores = []
        bboxes = []
        classids = []
        classnames = []
        for result in results:
            score = move_to(detach(result['scores']), torch.device('cpu')).numpy().tolist()
            boxes = move_to(detach(result['boxes']), torch.device('cpu')).numpy().tolist()
            classid = move_to(detach(result['labels']), torch.device('cpu')).numpy().tolist()
            scores.append(score)
            bboxes.append(boxes)
            classids.append(classid)
            if self.classnames:
                classname = [self.classnames[int(clsid)] for clsid in classid]
                classnames.append(classname)

        return {
            'boxes': bboxes,
            'labels': classids,
            'confidences': scores, 
            'names': classnames,
        }