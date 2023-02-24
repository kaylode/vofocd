from typing import Any, Dict
from theseus.base.metrics.metric_template import Metric

class MultiMetricWrapper(Metric):
    def __init__(self, cls_metrics, det_metrics, **kwargs) -> None:
        super().__init__()
        self.cls_metrics = cls_metrics
        self.det_metrics = det_metrics

    def reset(self):
        for metric in self.cls_metrics:
            metric.reset()

        for metric in self.det_metrics:
            metric.reset()
    
    def update(self, output, batch):

        for metric in self.cls_metrics:
            metric.update(
                outputs={
                    'outputs': output['img_outputs']
                }, 
                batch={
                    'inputs': batch['inputs'],
                    'targets': batch['img_targets']
                }, 
            )

        for metric in self.det_metrics:
            metric.update(
                output=output['obj_outputs'],
                batch={
                    'img_ids': batch['img_ids'], 
                    'img_names': batch['img_names'], 
                    'inputs': batch['inputs'],
                    'targets': batch['obj_targets']
                }, 
            )


    def value(self):
        results_dict = {}

        for metric in self.cls_metrics:
            score_dict = metric.value()
            results_dict.update(score_dict)

        for metric in self.det_metrics:
            score_dict = metric.value()
            results_dict.update(score_dict)

        return results_dict