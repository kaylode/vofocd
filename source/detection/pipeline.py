import torch
from source.detection.models import MODEL_REGISTRY
from source.detection.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY
from source.detection.losses import LOSS_REGISTRY
from theseus.cv.detection.pipeline import DetectionPipeline
from theseus.base.utilities.loggers import LoggerObserver

class DetPipeline(DetectionPipeline):
    """docstring for Pipeline."""

    def __init__(
        self,
        opt
    ):
        super(DetectionPipeline, self).__init__(opt)
        self.opt = opt

    def init_registry(self):
        super().init_registry()
        self.model_registry = MODEL_REGISTRY
        self.loss_registry = LOSS_REGISTRY
        self.dataset_registry = DATASET_REGISTRY
        self.dataloader_registry = DATALOADER_REGISTRY
        self.logger.text(
            "Overidding registry in pipeline...", LoggerObserver.INFO
        )