import torch
from source.detection.models import MODEL_REGISTRY, ModelWithLossandPostprocess
from source.detection.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY
from source.detection.losses import LOSS_REGISTRY
from theseus.base.utilities.loading import load_state_dict
from theseus.base.utilities.download import download_from_url

from theseus.base.pipeline import BasePipeline
from theseus.base.utilities.cuda import get_devices_info
from theseus.base.utilities.loggers import LoggerObserver
from theseus.cv.detection.augmentations import TRANSFORM_REGISTRY
from theseus.cv.detection.callbacks import CALLBACKS_REGISTRY
from theseus.cv.detection.metrics import METRIC_REGISTRY
from theseus.cv.detection.trainer import TRAINER_REGISTRY
from theseus.opt import Config

class DetPipeline(BasePipeline):
    """docstring for Pipeline."""

    def __init__(
        self,
        opt: Config
    ):
        super(DetPipeline, self).__init__(opt)
        self.opt = opt

    def init_registry(self):
        super().init_registry()
        self.model_registry = MODEL_REGISTRY
        self.dataset_registry = DATASET_REGISTRY
        self.dataloader_registry = DATALOADER_REGISTRY
        self.metric_registry = METRIC_REGISTRY
        self.loss_registry = LOSS_REGISTRY
        self.callbacks_registry = CALLBACKS_REGISTRY
        self.trainer_registry = TRAINER_REGISTRY
        self.transform_registry = TRANSFORM_REGISTRY
        self.logger.text("Overidding registry in pipeline...", LoggerObserver.INFO)

    def init_model_with_loss(self):
        model = self.init_model()
        criterion = self.init_criterion()
        self.model = ModelWithLossandPostprocess(model, criterion, self.device)
        self.logger.text(
            f"Number of trainable parameters: {self.model.trainable_parameters():,}",
            level=LoggerObserver.INFO,
        )
        device_info = get_devices_info(self.device_name)
        self.logger.text("Using " + device_info, level=LoggerObserver.INFO)

    def init_loading(self):
        self.resume = self.opt['global']['resume']
        self.pretrained = self.opt['global']['pretrained']
        self.last_epoch = -1
        if self.pretrained:
            if self.pretrained.startswith('https'):
                self.pretrained = download_from_url(self.pretrained)
            state_dict = torch.load(self.pretrained, map_location='cpu')
            self.model.model.model = load_state_dict(self.model.model.model, state_dict, strict=False, key='model')

        if self.resume:
            state_dict = torch.load(self.resume, map_location='cpu')
            self.model.model = load_state_dict(self.model.model, state_dict, 'model')
            self.optimizer = load_state_dict(self.optimizer, state_dict, 'optimizer')
            iters = load_state_dict(None, state_dict, 'iters')
            self.last_epoch = iters//len(self.train_dataloader) - 1