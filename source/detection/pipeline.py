from source.detection.models import MODEL_REGISTRY
from source.detection.losses import LOSS_REGISTRY
from source.detection.datasets import DATASET_REGISTRY
from source.detection.augmentations import TRANSFORM_REGISTRY
from theseus.cv.classification.pipeline import BasePipeline
from theseus.base.utilities.getter import (get_instance, get_instance_recursively)
from theseus.base.utilities.cuda import move_to

class Pipeline(BasePipeline):
    """docstring for Pipeline."""

    def __init__(
        self,
        opt
    ):
        super(Pipeline, self).__init__(opt)
        self.opt = opt

    def init_registry(self):
        super().init_registry()
        self.model_registry = MODEL_REGISTRY
        self.dataset_registry = DATASET_REGISTRY
        self.loss_registry = LOSS_REGISTRY
        self.transform_registry = TRANSFORM_REGISTRY

    def init_criterion(self):
        CLASSNAMES = self.val_dataset.classnames
        criterion = get_instance_recursively(
            self.opt["loss"], 
            registry=self.loss_registry,
            num_classes=len(CLASSNAMES))
        criterion = move_to(criterion, self.device)
        return criterion

    def init_metrics(self):
        CLASSNAMES = self.val_dataset.classnames
        self.metrics = get_instance_recursively(
            self.opt['metrics'], 
            registry=self.metric_registry, 
            num_classes=len(CLASSNAMES),
            classnames=CLASSNAMES)