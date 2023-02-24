import os
from theseus.opt import Config
from theseus.cv.classification.pipeline import ClassificationPipeline
from source.classification.datasets import DATASET_REGISTRY
from theseus.base.utilities.loggers import LoggerObserver


class ClsPipeline(ClassificationPipeline):
    """docstring for Pipeline."""

    def __init__(
        self,
        opt: Config
    ):
        super(ClsPipeline, self).__init__(opt)
 
    def init_registry(self):
        super().init_registry()
        self.dataset_registry = DATASET_REGISTRY
        self.logger.text(
            "Overidding registry in pipeline...", LoggerObserver.INFO
        )