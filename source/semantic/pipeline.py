from source.semantic.models import MODEL_REGISTRY
from source.semantic.datasets import DATASET_REGISTRY
from theseus.cv.semantic.pipeline import BasePipeline

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