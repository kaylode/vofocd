from typing import Dict, List, Any, Optional
# import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from theseus.base.utilities.cuda import move_to, detach
from theseus.base.utilities.logits import logits2labels
from .detr_utils import box_ops
import numpy as np

# Dependency: pip3 install effdet
from effdet.config.model_config import efficientdet_model_param_dict
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
from effdet.config.model_config import efficientdet_model_param_dict

# Dependency: pip3 install ensemble-boxes
from ensemble_boxes import ensemble_boxes_wbf

def rescale_bboxes(predicted_bboxes, image_sizes, img_size):
        scaled_bboxes = []
        for bboxes, img_dims in zip(predicted_bboxes, image_sizes):
            im_h, im_w = img_dims

            if len(bboxes) > 0:
                scaled_bboxes.append(
                    (
                        np.array(bboxes)
                        * [
                            im_w / img_size,
                            im_h / img_size,
                            im_w / img_size,
                            im_h / img_size,
                        ]
                    ).tolist()
                )
            else:
                scaled_bboxes.append(bboxes)

        return scaled_bboxes

def create_model(num_classes=6, image_size=512, architecture="resnet50"):
    efficientdet_model_param_dict['resnet50'] = dict(
        name='resnet50',
        backbone_name='resnet50',
        backbone_args=dict(drop_path_rate=0.2),
        num_classes=num_classes,
        url='', )
    
    config = get_efficientdet_config(architecture)
    config.update({'num_classes': num_classes})
    config.update({'image_size': (image_size, image_size)})
    
    # print(config)

    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
    )
    return DetBenchTrain(net, config)

def run_wbf(predictions, image_size=512, iou_thr=0.44, skip_box_thr=0.43, weights=None):
    bboxes = []
    confidences = []
    class_labels = []

    for prediction in predictions:
        boxes = [(prediction["boxes"] / image_size).tolist()]
        scores = [prediction["scores"].tolist()]
        labels = [prediction["classes"].tolist()]

        boxes, scores, labels = ensemble_boxes_wbf.weighted_boxes_fusion(
            boxes,
            scores,
            labels,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )
        
        boxes = boxes * (image_size - 1)
        bboxes.append(boxes.tolist())
        confidences.append(scores.tolist())
        class_labels.append(labels.tolist())

    return bboxes, confidences, class_labels

def post_process_detections(detections):
    predictions = []
    # predictions = {
    #     "scores":[], 
    #     "boxes":[],
    #     "labels":[],
    # }
    # outputs = [{
    #     "scores": move_to(torch.tensor(predicted_class_confidences).float(), device),
    #     "boxes": move_to(torch.tensor(predicted_bboxes).float(), device), 
    #     "labels": predicted_class_labels,
    # }]
    for i in range(detections.shape[0]):
        # boxes, scores, labels = _postprocess_single_prediction_detections(detections[i])
        # predictions["scores"].append(scores)
        # predictions["boxes"].append(boxes)
        # predictions["labels"].append(labels)
        # print(labels)
        predictions.append(
            _postprocess_single_prediction_detections(detections[i])
        )

    # predictions["scores"] = torch.tensor(predictions["scores"]).float()
    # predictions["boxes"] = torch.tensor(predictions["boxes"]).float()
    # predictions["labels"] = torch.tensor(predictions["labels"]).int()
    # return [predictions]
    predicted_bboxes, predicted_class_confidences, predicted_class_labels = run_wbf(
        predictions, image_size=512, iou_thr=0.44
    )
    return predicted_bboxes, predicted_class_confidences, predicted_class_labels

def _postprocess_single_prediction_detections(detections, prediction_confidence_threshold=0.2):
    boxes = detections.detach().cpu().numpy()[:, :4]
    scores = detections.detach().cpu().numpy()[:, 4]
    classes = detections.detach().cpu().numpy()[:, 5]
    indexes = np.where(scores > prediction_confidence_threshold)[0]
    boxes = boxes[indexes]
    # return boxes, scores, classes[indexes]
    return {"boxes": boxes, "scores": scores[indexes], "classes": classes[indexes]}

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
        **kwargs
    ):
        super().__init__()
        self.name = model_name
        self.num_classes = num_classes
        self.image_size = image_size
        self.architecture = architecture
        self.classnames = classnames
        self.model = create_model(
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
            outputs = self.model(x, y)['detections']
            predicted_bboxes, predicted_class_confidences, predicted_class_labels = post_process_detections(outputs)
            # scaled_bboxes = rescale_bboxes(
            #     predicted_bboxes=predicted_bboxes, image_sizes=image_sizes, img_size=self.image_size
            # )
            outputs = []
            for x, y, z in zip(predicted_bboxes, predicted_class_confidences, predicted_class_labels):
                outputs.append({
                    "boxes": torch.tensor(x)[:, [1, 0, 3, 2]].float(),
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