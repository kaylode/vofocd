from typing import Dict, List, Any, Optional
# import timm
import torch
import torch.nn as nn
from theseus.base.utilities.cuda import move_to, detach
import numpy as np

# Dependency: pip3 install effdet
from effdet.config.model_config import efficientdet_model_param_dict
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from effdet.efficientdet import HeadNet
from effdet.config.model_config import efficientdet_model_param_dict

# # Dependency: pip3 install ensemble-boxes
# from ensemble_boxes import ensemble_boxes_wbf

# def rescale_bboxes(predicted_bboxes, image_sizes, img_size):
#         scaled_bboxes = []
#         for bboxes, img_dims in zip(predicted_bboxes, image_sizes):
#             im_h, im_w = img_dims

#             if len(bboxes) > 0:
#                 scaled_bboxes.append(
#                     (
#                         np.array(bboxes)
#                         * [
#                             im_w / img_size,
#                             im_h / img_size,
#                             im_w / img_size,
#                             im_h / img_size,
#                         ]
#                     ).tolist()
#                 )
#             else:
#                 scaled_bboxes.append(bboxes)

#         return scaled_bboxes

def create_model(drop_path_rate=0.2,
                 soft_nms=True,
                 pretrained_backbone=True,
                 num_classes=6, 
                 image_size=512, 
                 architecture="resnet50",
                 is_train:bool=True):
    efficientdet_model_param_dict['resnet50'] = dict(
        name='resnet50',
        backbone_name='resnet50',
        backbone_args=dict(drop_path_rate=drop_path_rate),
        num_classes=num_classes,
        url = '',
        # url='https://download.pytorch.org/models/resnet50-11ad3fa6.pth', # IMAGENET1K_V2
    )
    
    config = get_efficientdet_config(architecture)
    config.update({'num_classes': num_classes})
    config.update({'image_size': (image_size, image_size)})
    config.update({'soft_nms': soft_nms})
    
    print(10*"*", "Effdet config", 10*"*")
    print(config)

    net = EfficientDet(config, pretrained_backbone=pretrained_backbone)
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
    )

    if is_train:
        return DetBenchTrain(net)
    else:
        return DetBenchPredict(net)

# def run_wbf(predictions, image_size=512, iou_thr=0.44, skip_box_thr=0.43, weights=None):
#     bboxes = []
#     confidences = []
#     class_labels = []

#     for prediction in predictions:
#         boxes = [(prediction["boxes"] / image_size).tolist()]
#         scores = [prediction["scores"].tolist()]
#         labels = [prediction["classes"].tolist()]

#         boxes, scores, labels = ensemble_boxes_wbf.weighted_boxes_fusion(
#             boxes,
#             scores,
#             labels,
#             weights=weights,
#             iou_thr=iou_thr,
#             skip_box_thr=skip_box_thr,
#         )
        
#         boxes = boxes * (image_size - 1)
#         bboxes.append(boxes.tolist())
#         confidences.append(scores.tolist())
#         class_labels.append(labels.tolist())

#     return bboxes, confidences, class_labels

def post_process_detections(detections):
    predictions = []

    for i in range(detections.shape[0]):
        predictions.append(
            _postprocess_single_prediction_detections(detections[i])
        )
    return predictions
    # predicted_bboxes, predicted_class_confidences, predicted_class_labels = run_wbf(
    #     predictions, image_size=512, iou_thr=0.44
    # )s
    # return predicted_bboxes, predicted_class_confidences, predicted_class_labels

def _postprocess_single_prediction_detections(detections, prediction_confidence_threshold=0.2):
    boxes = detections.detach().cpu().numpy()[:, :4]
    scores = detections.detach().cpu().numpy()[:, 4]
    classes = detections.detach().cpu().numpy()[:, 5]
    indexes = np.where(scores > prediction_confidence_threshold)[0]
    boxes = boxes[indexes]
    return boxes, scores[indexes], classes[indexes]
    # return {"boxes": boxes, "scores": scores[indexes], "classes": classes[indexes]}

def _create_dummy_inference_targets(num_images, device, img_size):
    dummy_targets = {
        "bbox": [
            torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=device)
            for i in range(num_images)
        ],
        "cls": [torch.tensor([1.0], device=device) for i in range(num_images)],
        "img_size": torch.tensor(
            [(img_size, img_size)] * num_images, device=device
        ).float(),
        "img_scale": torch.ones(num_images, device=device).float(),
    }

    return dummy_targets

class EffDet(nn.Module):
    """DocString"""

    def __init__(
        self,
        model_name: str,
        num_classes: int=6,
        image_size: int=512,
        architecture: str="resnet50",
        classnames: Optional[List] = None,
        drop_path_rate: float=0.2,
        soft_nms: bool=False,
        pretrained_backbone: bool=True,
        **kwargs
    ):
        super().__init__()
        self.name = model_name
        self.num_classes = num_classes
        self.image_size = image_size
        self.architecture = architecture
        self.classnames = classnames
        self.drop_path_rate = kwargs.get('drop_path_rate', 0.2)
        self.soft_nms = kwargs.get('soft_nms', False)
        self.pretrained_backbone = kwargs.get('pretrained_backbone', True)
        self.model = create_model(
            drop_path_rate=drop_path_rate,
            soft_nms=soft_nms,
            pretrained_backbone=pretrained_backbone,
            num_classes=num_classes-1, 
            image_size=image_size,
            architecture=architecture
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
            targets = batch['targets']
            y = {
                "bbox": [move_to(target["boxes"].float()[:, [1, 0, 3, 2]], device) for target in targets], #yxyx
                "cls": [move_to(target["labels"].float(), device)  for target in targets],
                "img_scale": move_to(torch.tensor([(self.image_size, self.image_size)]*len(targets)).float(), device), 
                "img_size": move_to(torch.ones(len(targets)).float(), device)
            }
            self.model.train()
            loss_dict = self.model(x, y)
            loss = loss_dict['loss']
            loss_dict = {k:move_to(detach(v), torch.device('cpu')) for k,v in loss_dict.items()}
        else:
            self.model.eval()
            y = _create_dummy_inference_targets(x.shape[0], device, self.image_size)

            """
            If errors occur, change these lines (396-398) from effdet library:
                cls_targets_out[level_idx].append(
                    cls_targets[count:count + steps].view([feat_size[0], feat_size[1], -1]))
                box_targets_out[level_idx].append(
                    box_targets[count:count + steps].view([feat_size[0], feat_size[1], -1]))
            to 
                cls_targets_out[level_idx].append(
                    cls_targets[count:count + steps].reshape([feat_size[0], feat_size[1], -1]))
                box_targets_out[level_idx].append(
                    box_targets[count:count + steps].reshape([feat_size[0], feat_size[1], -1]))
            """

            outputs = self.model(x, y)['detections']
            predictions = post_process_detections(outputs)
            outputs = []
            for x, y, z in predictions:
                outputs.append({
                    "boxes": torch.tensor(x).float(),
                    "scores": torch.tensor(y).float(), 
                    "labels": torch.tensor(z).int(),
                })
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