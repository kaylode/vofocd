import torch
from source.detection.models import MODEL_REGISTRY
from source.detection.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY
from source.detection.losses import LOSS_REGISTRY
from theseus.cv.detection.pipeline import DetectionPipeline
from theseus.base.utilities.loggers import LoggerObserver

from theseus.base.utilities.loading import load_state_dict

LOGGER = LoggerObserver.getLogger('main')

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

    def init_loading(self):
        self.last_epoch = -1
        if getattr(self, "pretrained", None):
            state_dict = torch.load(self.pretrained, map_location="cpu")
            self.model.model = load_state_dict(self.model.model, state_dict, "model", strict=False)

        pretrained_backbone = self.opt['global'].get('pretrained_backbone', None)
        if pretrained_backbone:
            state_dict = torch.load(pretrained_backbone, map_location="cpu")
            self.model.model.model.backbone[0].body = load_state_dict(
                self.model.model.model.backbone[0].body, 
                state_dict, 
                key="model", strict=False
            )

        # Freeze all norm layers of backbone
        # https://github.com/facebookresearch/detr/issues/154
        try:
            self.model.model.model.backbone[0].freeze_norm_layers()
            LOGGER.text("Freezed normalization layers.", level=LoggerObserver.DEBUG)
        except:
            LOGGER.text("Failed to freeze normalization layers. Continue training...", level=LoggerObserver.WARN)

        if getattr(self, "resume", None):
            state_dict = torch.load(self.resume, map_location="cpu")
            self.model.model = load_state_dict(self.model.model, state_dict, "model")
            self.optimizer = load_state_dict(self.optimizer, state_dict, "optimizer")
            iters = load_state_dict(None, state_dict, "iters")
            self.last_epoch = iters // len(self.train_dataloader) - 1