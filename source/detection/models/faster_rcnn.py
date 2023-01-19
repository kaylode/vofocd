from typing import Dict, List, Any, Optional
# import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from theseus.base.utilities.cuda import move_to, detach
from theseus.base.utilities.logits import logits2labels
import torchvision
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2

class FasterRCNN(nn.Module):
    """DocString"""

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        weights: str='DEFAULT',
        # weights_backbone: Optional[torchvision.models.resnet.ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V2,
        **kwargs
    ):
        super().__init__()
        self.name = model_name
        self.num_classes = num_classes
        self.weights = weights
        # self.weights_backbone = weights_backbone
        self.model = fasterrcnn_resnet50_fpn(
            num_classes,
            weights,
            # weights_backbone,
        )

    def get_model(self):
        """
        Return the full architecture of the model, for visualization
        """
        return self.model
    
    def forward_batch(self, batch: Dict, device: torch.device):
        x = move_to(batch['inputs'], device)
        outputs = self.model(x)
        return {
            'outputs': outputs,
        }

    def postprocess(self, outputs: Dict, batch: Dict):
        batch_size = outputs['outputs']['pred_logits'].shape[0]
        target_sizes = torch.Tensor([batch['inputs'].shape[-2:]]).repeat(batch_size, 1)

        results = self.postprocessor(
            outputs = outputs['outputs'],
            target_sizes=target_sizes
        )

        denormalized_targets = batch['targets']
        denormalized_targets = self.postprocessor(
            outputs = denormalized_targets,
            target_sizes=target_sizes
        )

        batch['targets'] = denormalized_targets
        return results, batch
    
    @torch.no_grad()
    def get_prediction(self, adict: Dict[str, Any], device: torch.device):
        """
        Inference using the model.
        adict: `Dict[str, Any]`
            dictionary of inputs
        device: `torch.device`
            current device 
        """
        outputs = self.forward_batch(adict, device)

        batch_size = outputs['outputs']['pred_logits'].shape[0]
        target_sizes = torch.Tensor([adict['inputs'].shape[-2:]]).repeat(batch_size, 1)

        results = self.postprocessor(
            outputs = outputs['outputs'],
            target_sizes=target_sizes
        )
        
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