import torch
import numpy as np
from ensemble_boxes import weighted_boxes_fusion, nms



def filter_area(
    boxes, labels, confidence_score=None, 
    min_wh=10, max_wh=4096, min_area_pct=None, image_size=None):
    """
    Boxes in xyxy format
    """

    # dimension of bounding boxes
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]

    width = width.astype(int)
    height = height.astype(int)

    picked_index_min = (width >= min_wh) & (height >= min_wh)
    picked_index_max = (width <= max_wh) & (height <= max_wh)

    picked_index = picked_index_min & picked_index_max
    
    if min_area_pct is not None:
        assert image_size is not None, "Need to input image size"
        boxes_areas = width * height
        image_areas = image_size[0] * image_size[1]
        pct = (boxes_areas*1.0 / image_areas) * 100
        picked_index_pct = pct > min_area_pct
        picked_index &= picked_index_pct

    # Picked bounding boxes
    picked_boxes = boxes[picked_index]
    picked_classes = labels[picked_index]
    if confidence_score is not None:
        picked_score = confidence_score[picked_index]
    
    if confidence_score is not None:
        return np.array(picked_boxes), np.array(picked_score), np.array(picked_classes)
    else:
        return np.array(picked_boxes), np.array(picked_classes)

def resize_postprocessing(boxes, current_img_size, ori_img_size, keep_ratio=False):
    """
    Boxes format must be in xyxy
    if keeping ratio, padding will be calculated then substracted from bboxes
    """

    new_boxes = boxes.copy()
    if keep_ratio:
        ori_w, ori_h = ori_img_size
        ratio = float(ori_w*1.0/ori_h)
        
        # If ratio equals 1.0, skip to scaling
        if ratio != 1.0: 
            if ratio > 1.0: # width > height, width = current_img_size, meaning padding along height
                true_width = current_img_size[0]
                true_height = current_img_size[0] / ratio # true height without padding equals (current width / ratio)
                pad_size = int((true_width-true_height)/2) # Albumentation padding
                
                # Subtract padding size from heights
                new_boxes[:,1] -= pad_size
                new_boxes[:,3] -= pad_size
            else: # height > width, height = current_img_size, meaning padding along width
                true_height = current_img_size[1]
                true_width = current_img_size[1] * ratio # true width without padding equals (current height * ratio)
                pad_size = int((true_height-true_width)/2) # Albumentation padding

                # Subtract padding size from widths
                new_boxes[:,0] -= pad_size
                new_boxes[:,2] -= pad_size
            # Assign new width, new height
            current_img_size = [true_width, true_height]
    
    # Scaling boxes to match original image shape 
    new_boxes[:,0] = (new_boxes[:,0] * ori_img_size[0])/ current_img_size[0]
    new_boxes[:,2] = (new_boxes[:,2] * ori_img_size[0])/ current_img_size[0]
    new_boxes[:,1] = (new_boxes[:,1] * ori_img_size[1])/ current_img_size[1]
    new_boxes[:,3] = (new_boxes[:,3] * ori_img_size[1])/ current_img_size[1]
    return new_boxes

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (width, height)
    if isinstance(boxes, torch.Tensor):
        _boxes = boxes.clone()
        _boxes[:, 0].clamp_(0, img_shape[0])  # x1
        _boxes[:, 1].clamp_(0, img_shape[1])  # y1
        _boxes[:, 2].clamp_(0, img_shape[0])  # x2
        _boxes[:, 3].clamp_(0, img_shape[1])  # y2
    else:
        _boxes = boxes.copy()
        _boxes[:, 0] = np.clip(_boxes[:, 0], 0, img_shape[0])  # x1
        _boxes[:, 1] = np.clip(_boxes[:, 1], 0, img_shape[1])  # y1
        _boxes[:, 2] = np.clip(_boxes[:, 2], 0, img_shape[0])  # x2
        _boxes[:, 3] = np.clip(_boxes[:, 3], 0, img_shape[1])  # y2

    return _boxes

def postprocessing(
        preds, 
        current_img_size=None,  # Need to be square
        ori_img_size=None,
        min_iou=0.5, 
        min_conf=0.1,
        mode=None,
        max_dets=None,
        keep_ratio=False,
        output_format='xywh'):
    """
    Input: bounding boxes in xyxy format
    Output: bounding boxes in xywh format
    """
    boxes, scores, labels = preds['bboxes'], preds['scores'], preds['classes']

    if len(boxes) == 0 or boxes is None:
        return {
            'bboxes': boxes, 
            'scores': scores, 
            'classes': labels}

    # Clip boxes in image size
    boxes = clip_coords(boxes, current_img_size)

    # Filter small area boxes
    boxes, scores, labels = filter_area(
        boxes, labels, scores, min_wh=2, max_wh=4096
    )

    current_img_size = current_img_size if current_img_size is not None else None
    if len(boxes) != 0:
        if mode is not None:
            boxes, scores, labels = box_fusion(
                [boxes],
                [scores],
                [labels],
                image_size=current_img_size,
                mode=mode,
                iou_threshold=min_iou)

        indexes = np.where(scores > min_conf)[0]
        
        boxes = boxes[indexes]
        scores = scores[indexes]
        labels = labels[indexes]

        if max_dets is not None:
            sorted_index = np.argsort(scores)
            boxes = boxes[sorted_index]
            scores = scores[sorted_index]
            labels = labels[sorted_index]
            
            boxes = boxes[:max_dets]
            scores = scores[:max_dets]
            labels = labels[:max_dets]

        if ori_img_size is not None and current_img_size is not None:
            boxes = resize_postprocessing(
                boxes, 
                current_img_size=current_img_size, 
                ori_img_size=ori_img_size, 
                keep_ratio=keep_ratio)

        if output_format == 'xywh':
            boxes = change_box_order(boxes, order='xyxy2xywh')


    return {
        'bboxes': boxes, 
        'scores': scores, 
        'classes': labels}

def box_fusion(
    bounding_boxes, 
    confidence_score, 
    labels, 
    mode='wbf', 
    image_size=None,
    weights=None, 
    iou_threshold=0.5):
    """
    bounding boxes: 
        list of boxes of same image [[box1, box2,...],[...]] if ensemble many models
        list of boxes of single image [[box1, box2,...]] if done on one model
        image size: [w,h]
    """

    if image_size is not None:
        if isinstance(image_size, int):
            image_size = [image_size, image_size]
        normalized_boxes = []
        for ens_boxes in bounding_boxes:
            if isinstance(ens_boxes, list):
                ens_boxes = np.array(ens_boxes)
            ens_boxes[:,0] = ens_boxes[:,0]*1.0/image_size[0]
            ens_boxes[:,1] = ens_boxes[:,1]*1.0/image_size[1]
            ens_boxes[:,2] = ens_boxes[:,2]*1.0/image_size[0]
            ens_boxes[:,3] = ens_boxes[:,3]*1.0/image_size[1]
            normalized_boxes.append(ens_boxes)
        normalized_boxes = np.array(normalized_boxes)
    else:
        normalized_boxes = bounding_boxes.copy()

    if mode == 'wbf':
        picked_boxes, picked_score, picked_classes = weighted_boxes_fusion(
            normalized_boxes, 
            confidence_score, 
            labels, 
            weights=weights, 
            iou_thr=iou_threshold, 
            conf_type='avg', #[nms|avf]
            skip_box_thr=0.0001)
    elif mode == 'nms':
        picked_boxes, picked_score, picked_classes = nms(
            normalized_boxes, 
            confidence_score, 
            labels,
            weights=weights,
            iou_thr=iou_threshold)

    if image_size is not None:
        result_boxes = []
        for ens_boxes in picked_boxes:
            ens_boxes[0] = ens_boxes[0]*image_size[0]
            ens_boxes[1] = ens_boxes[1]*image_size[1]
            ens_boxes[2] = ens_boxes[2]*image_size[0]
            ens_boxes[3] = ens_boxes[3]*image_size[1]
            result_boxes.append(ens_boxes)

    return np.array(result_boxes), np.array(picked_score), np.array(picked_classes)