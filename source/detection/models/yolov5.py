from typing import Dict, List, Any, Optional
# import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from theseus.base.utilities.cuda import move_to, detach
from theseus.base.utilities.logits import logits2labels
from .detr_utils import box_ops

class YoloV5(nn.Module):
    """DocString"""

    def __init__(
        self,
        model_name: str,
        num_classes: int=6,
        classnames: Optional[List] = None,
        **kwargs
    ):
        super().__init__()
        self.name = model_name
        self.num_classes = num_classes
        self.classnames = classnames
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False)

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
            print("Train output")
            print(loss_dict)
            # loss = sum(loss for loss in loss_dict.values())
            # loss_dict = {k:move_to(detach(v), torch.device('cpu')) for k,v in loss_dict.items()}
            # pred = self.model(imgs)  # forward
            # loss, loss_items = compute_loss(pred, targets.to(device))
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