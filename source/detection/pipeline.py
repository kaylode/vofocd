import os
import torch
from theseus.opt import Config
from theseus.base.pipeline import BasePipeline, BaseTestPipeline
from theseus.base.optimizers import OPTIM_REGISTRY, SCHEDULER_REGISTRY
from theseus.cv.classification.augmentations import TRANSFORM_REGISTRY
from source.classification.losses import LOSS_REGISTRY
from theseus.cv.classification.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY
from theseus.cv.classification.trainer import TRAINER_REGISTRY
from theseus.cv.classification.metrics import METRIC_REGISTRY
from source.classification.models import MODEL_REGISTRY
from source.classification.callbacks import CALLBACKS_REGISTRY
from theseus.base.utilities.loading import load_state_dict
from theseus.base.utilities.loggers import LoggerObserver
from theseus.base.utilities.getter import (get_instance, get_instance_recursively)
from theseus.base.utilities.cuda import move_to


class DetPipeline(BasePipeline):
    """docstring for Pipeline."""

    def __init__(
        self,
        opt: Config
    ):
        super(Pipeline, self).__init__(opt)
        self.opt = opt
 
    def init_registry(self):
        self.model_registry = MODEL_REGISTRY
        self.dataset_registry = DATASET_REGISTRY
        self.dataloader_registry = DATALOADER_REGISTRY
        self.metric_registry = METRIC_REGISTRY
        self.loss_registry = LOSS_REGISTRY
        self.optimizer_registry = OPTIM_REGISTRY
        self.scheduler_registry = SCHEDULER_REGISTRY
        self.callbacks_registry = CALLBACKS_REGISTRY
        self.trainer_registry = TRAINER_REGISTRY
        self.transform_registry = TRANSFORM_REGISTRY
        self.logger.text(
            "Overidding registry in pipeline...", LoggerObserver.INFO
        )