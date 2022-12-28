import torch 
from source.detection.models import MODEL_REGISTRY, ModelWithLossandPostprocess
from source.detection.losses import LOSS_REGISTRY
from source.detection.datasets import DATASET_REGISTRY
from source.detection.augmentations import TRANSFORM_REGISTRY
from source.detection.metrics import METRIC_REGISTRY
from theseus.cv.classification.pipeline import BasePipeline
from theseus.base.utilities.getter import (get_instance, get_instance_recursively)
from theseus.base.utilities.cuda import move_to
from theseus.base.utilities.loading import load_state_dict
from theseus.base.utilities.download import download_from_url

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
        self.metric_registry = METRIC_REGISTRY

    def init_model(self):
        CLASSNAMES = self.val_dataset.classnames
        model = get_instance(
            self.opt["model"], 
            registry=self.model_registry, 
            num_classes=len(CLASSNAMES),
            classnames=CLASSNAMES)
        model = move_to(model, self.device)
        return model

    def init_model_with_loss(self):
        model = self.init_model()
        criterion = self.init_criterion()
        self.model = ModelWithLossandPostprocess(model, criterion, self.device)
        self.logger.text(f"Number of trainable parameters: {self.model.trainable_parameters():,}", level=LoggerObserver.INFO)
        device_info = get_devices_info(self.device_name)
        self.logger.text("Using " + device_info, level=LoggerObserver.INFO)

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

    def init_loading(self):
        self.resume = self.opt['global']['resume']
        self.pretrained = self.opt['global']['pretrained']
        self.last_epoch = -1
        if self.pretrained:
            if self.pretrained.startswith('https'):
                self.pretrained = download_from_url(self.pretrained)
            state_dict = torch.load(self.pretrained, map_location='cpu')
            self.model.model.model = load_state_dict(self.model.model.model, state_dict, strict=False)

        if self.resume:
            state_dict = torch.load(self.resume, map_location='cpu')
            self.model.model = load_state_dict(self.model.model, state_dict, 'model')
            self.optimizer = load_state_dict(self.optimizer, state_dict, 'optimizer')
            iters = load_state_dict(None, state_dict, 'iters')
            self.last_epoch = iters//len(self.train_dataloader) - 1