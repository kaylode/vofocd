from source.detection.models import MODEL_REGISTRY
from source.detection.losses import LOSS_REGISTRY
from theseus.cv.detection.pipeline import BasePipeline

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
        self.loss_registry = LOSS_REGISTRY