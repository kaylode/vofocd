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
from .detr_utils import box_ops

# class PostProcess(nn.Module):
#     """ This module converts the model's output into the format expected by the coco api"""
#     def __init__(self, min_conf=0.25):
#         super().__init__()
#         self.min_conf = min_conf

#     @torch.no_grad()
#     def forward(self, outputs, target_sizes):
#         """ Perform the computation
#         Parameters:
#             outputs: raw outputs of the model
#             target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
#                           For evaluation, this must be the original image size (before any data augmentation)
#                           For visualization, this should be the image size after data augment, but before padding
#         """

#         if isinstance(outputs, dict) and 'pred_logits' in outputs.keys():
#             logits, boxes = outputs['pred_logits'], outputs['pred_boxes']

#             assert len(logits) == len(target_sizes)
#             assert target_sizes.shape[1] == 2

#             prob = F.softmax(logits, -1)
#             scores, labels = prob[..., :-1].max(-1)
#             # convert to [x0, y0, x1, y1] format
#             boxes = box_ops.box_cxcywh_to_xyxy(boxes)
#             # and from relative [0, 1] to absolute [0, height] coordinates
#             img_h, img_w = target_sizes.unbind(1)
#             scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
#             boxes = boxes * scale_fct[:, None, :]

#             results = []
#             for box, score, label in zip(boxes, scores, labels):
#                 keep_idx = score >= self.min_conf
#                 keep_score = score[keep_idx]
#                 keep_box = box[keep_idx]
#                 keep_label = label[keep_idx]
#                 results.append({'scores': keep_score, 'labels': keep_label, 'boxes': keep_box})
#         else:
#             labels = [i['labels'] for i in outputs]
#             boxes = [i['boxes'] for i in outputs]
#             boxes = [box_ops.box_cxcywh_to_xyxy(box) for box in boxes]
#             img_h, img_w = target_sizes.unbind(1)
#             scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
#             new_boxes = [i*scale for i, scale in zip(boxes, scale_fct)]
#             results = [{'labels': l, 'boxes': b} for l, b in zip(labels, new_boxes)]
#         return results

class FasterRCNN(nn.Module):
    """DocString"""

    def __init__(
        self,
        model_name: str,
        num_classes: int=6,
        weights: str='DEFAULT',
        # min_conf: float = 0.25,
        classnames: Optional[List] = None,
        weights_backbone: Optional[torchvision.models.resnet.ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V2,
        **kwargs
    ):
        super().__init__()
        self.name = model_name
        self.num_classes = num_classes
        self.weights = weights
        self.classnames = classnames
        # self.postprocessor = PostProcess(min_conf=min_conf)
        self.weights_backbone = weights_backbone
        self.model = fasterrcnn_resnet50_fpn(
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
            
        else:
            self.model.eval()
            outputs = self.model(x)
        return {'outputs': outputs}, loss, loss_dict

    # def postprocess(self, outputs: Dict, batch: Dict):
    #     batch_size = outputs['outputs']['pred_logits'].shape[0]
    #     target_sizes = torch.Tensor([batch['inputs'].shape[-2:]]).repeat(batch_size, 1)

    #     results = self.postprocessor(
    #         outputs = outputs['outputs'],
    #         target_sizes=target_sizes
    #     )

    #     denormalized_targets = batch['targets']
    #     denormalized_targets = self.postprocessor(
    #         outputs = denormalized_targets,
    #         target_sizes=target_sizes
    #     )

    #     batch['targets'] = denormalized_targets
    #     return results, batch
    
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
        target_sizes = torch.Tensor([adict['inputs'].shape[-2:]]).repeat(batch_size, 1)

        # results = self.postprocessor(
        #     outputs = outputs['outputs'],
        #     target_sizes=target_sizes``
        # )
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