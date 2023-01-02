import torch
from source.detection.models import MODEL_REGISTRY
from source.detection.losses import LOSS_REGISTRY
from theseus.cv.detection.pipeline import Pipeline
from theseus.base.utilities.loading import load_state_dict
from theseus.base.utilities.download import download_from_url

class DetPipeline(Pipeline):
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
        self.logger.text(
            "Overidding registry in pipeline...", LoggerObserver.INFO
        )

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