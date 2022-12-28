from typing impoty List, Dict
import os
import numpy as np
import pandas as pd
from tabulate import tabulate

from theseus.base.metrics.metric_template import Metric
from .misc import MatchingPairs, BoxWithLabel

class PrecisionRecall(Metric):
    def __init__(self, num_classes, min_conf=0.2, min_iou=0.5, eps=1e-6, **kwargs):
        self.eps = eps
        self.min_iou = min_iou
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.all_gt_instances = []
        self.all_pred_instances = []

    def update(self, output, batch):
        img_sizes = batch['img_sizes']
        output = output["outputs"] 
        target = batch["targets"] 

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        # width, height = img.width, img.height
        # class_id = int(item[0])
        # if return_scores:
        #     score = float(item[-1])
        #     scores.append(score)
        # w = float(item[3]) * width
        # h = float(item[4]) * height
        # xmin = float(item[1]) * width - w/2
        # ymin = float(item[2]) * height - h/2
        # xmax = xmin + w
        # ymax = ymin + h
        # boxes.append([xmin, ymin, xmax, ymax])
        # labels.append(class_id)

        gt_instances = [BoxWithLabel(gt_id, box, int(text), 1.0) for box, text in zip(gt_boxes, gt_labels)]
        pred_instances = [BoxWithLabel(gt_id, box, int(text), score) for (box, text, score) in zip(pred_boxes, pred_labels, pred_scores)]

        all_gt_instances.append(gt_instances)
        all_pred_instances.append(pred_instances)

    def value(self):
        total_tp, total_fp, total_fn = self.calculate_cfm(
            self.all_pred_instances, 
            self.all_gt_instances
        )
        score = calculate_pr(total_tp, total_fp, total_fn)
        return score

    def calculate_cfm(self, pred_boxes: List[BoxWithLabel], gt_boxes: List[BoxWithLabel]):
        total_fp = []
        total_fn = []
        total_tp = []
        for pred_box, gt_box in zip(pred_boxes, gt_boxes):
            matched_pairs = MatchingPairs(
                pred_box, 
                gt_box, 
                min_iou=self.min_iou, 
                eps=self.eps
            )
            tp = matched_pairs.get_acc()
            fp = matched_pairs.get_false_positive()
            fn = matched_pairs.get_false_negative()
            total_tp += tp
            total_fp += fp
            total_fn += fn
        return total_tp, total_fp, total_fn

    def calculate_pr(self, total_tp, total_fp, total_fn):
        tp_per_class = {
            i: 0 for i in range(len(num_classes))
        }

        fp_per_class = {
            i: 0 for i in range(len(num_classes))
        }

        fn_per_class = {
            i: 0 for i in range(len(num_classes))
        }

        for box, _, _ in total_tp:
            label = box.get_label()
            tp_per_class[label] += 1

        for box in total_fp:
            label = box.get_label()
            fp_per_class[label] += 1

        for box in total_fn:
            label = box.get_label()
            fn_per_class[label] += 1

        precisions = []
        recalls = []
        num_classes = len(num_classes)
        for cls_id in range(num_classes):

            if tp_per_class[cls_id] + fp_per_class[cls_id] == 0:
                precisions.append(-1)
            else:
                precisions.append(
                    tp_per_class[cls_id] / (tp_per_class[cls_id] + fp_per_class[cls_id])
                )

            if tp_per_class[cls_id] + fn_per_class[cls_id] == 0:
                recalls.append(-1)
            else:
                recalls.append(
                    tp_per_class[cls_id] / (tp_per_class[cls_id] + fn_per_class[cls_id])
                )

        np_precisions = np.array(precisions)
        np_recalls = np.array(recalls)

        precision_all = sum(np_precisions[np_precisions!=-1]) / (num_classes - sum(np_precisions==-1))
        recall_all = sum(np_recalls[np_recalls!=-1]) / (num_classes - sum(np_recalls==-1))

        recalls.insert(0, recall_all)
        precisions.insert(0, precision_all)
        num_classes.insert(0, 'All')

        val_summary = {
            "Object name": num_classes,
            "Precision": precisions,
            "Recall": recalls
        }
        
        table = tabulate(
            val_summary, headers="keys", tablefmt="fancy_grid"
        )

        pd.DataFrame(val_summary).to_csv(os.path.join(OUTDIR, 'val_summary.csv'), index=False)