import os
import torch
from theseus.opt import Config
from theseus.cv.classification.pipeline import Pipeline
from theseus.base.pipeline import BaseTestPipeline
from source.classification.losses import LOSS_REGISTRY
from source.classification.models import MODEL_REGISTRY
from source.classification.callbacks import CALLBACKS_REGISTRY
from theseus.base.utilities.loading import load_state_dict
from theseus.base.utilities.loggers import LoggerObserver


class ClassificationPipeline(Pipeline):
    """docstring for Pipeline."""

    def __init__(
        self,
        opt: Config
    ):
        super(Pipeline, self).__init__(opt)
        self.opt = opt
        self.stage = self.opt["global"]["stage"]
        os.environ["WANDB_ENTITY"] = "kaylode"
 
    def init_registry(self):
        self.model_registry = MODEL_REGISTRY
        self.loss_registry = LOSS_REGISTRY
        self.callbacks_registry = CALLBACKS_REGISTRY
        self.logger.text(
            "Overidding registry in pipeline...", LoggerObserver.INFO
        )

    def init_loading(self):

        self.resume = self.opt['global']['resume']
        self.last_epoch = -1

        if self.stage == 'normal':
            self.pretrained = self.opt['global']['pretrained']
            if self.pretrained:
                state_dict = torch.load(self.pretrained, map_location='cpu')
                self.model.model = load_state_dict(self.model.model, state_dict, 'model')
        elif self.stage == 'distillation':
            self.pretrained_teacher = self.opt['global']['pretrained_teacher']
            if self.pretrained_teacher:
                state_dict = torch.load(self.pretrained_teacher, map_location='cpu')
                self.model.model.teacher = load_state_dict(self.model.model.teacher, state_dict, 'model')

            self.pretrained_student = self.opt['global']['pretrained_student']
            if self.pretrained_student:
                state_dict = torch.load(self.pretrained_student, map_location='cpu')
                self.model.model.student = load_state_dict(self.model.model.student, state_dict, 'model')

        if self.resume:
            state_dict = torch.load(self.resume)
            self.model.model = load_state_dict(self.model.model, state_dict, 'model')
            self.optimizer = load_state_dict(self.optimizer, state_dict, 'optimizer')
            iters = load_state_dict(None, state_dict, 'iters')
            self.last_epoch = iters//len(self.train_dataloader) - 1


class TestPipeline(BaseTestPipeline):

    def __init__(
        self,
        opt: Config
    ):
        super().__init__(opt)

    def init_loading(self):
        self.weights = self.opt['global']['weights']
        if self.weights:
            state_dict = torch.load(self.weights, map_location='cpu')
            self.model = load_state_dict(self.model, state_dict, 'model')