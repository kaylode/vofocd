import os
import torch
from theseus.opt import Config
from theseus.base.pipeline import BasePipeline
from theseus.base.optimizers import OPTIM_REGISTRY, SCHEDULER_REGISTRY
from theseus.classification.augmentations import TRANSFORM_REGISTRY
from source.losses import LOSS_REGISTRY
from theseus.classification.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY
from theseus.classification.trainer import TRAINER_REGISTRY
from theseus.classification.metrics import METRIC_REGISTRY
from source.models import MODEL_REGISTRY
from source.callbacks import CALLBACKS_REGISTRY
from theseus.utilities.loading import load_state_dict
from theseus.utilities.loggers import LoggerObserver
from theseus.utilities.getter import (get_instance, get_instance_recursively)
from theseus.utilities.cuda import move_to


class Pipeline(BasePipeline):
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

    def init_model(self):
        CLASSNAMES = self.val_dataset.classnames
        model = get_instance_recursively(
            self.opt["model"], 
            registry=self.model_registry, 
            num_classes=len(CLASSNAMES),
            classnames=CLASSNAMES)
        model = move_to(model, self.device)
        return model

    def init_loading(self):

        self.resume = self.opt['global']['resume']
        self.last_epoch = -1

        if self.stage == 'normal':
            self.pretrained = self.opt['global']['pretrained']
            if self.pretrained:
                state_dict = torch.load(self.pretrained)
                self.model.model = load_state_dict(self.model.model, state_dict, 'model')
        elif self.stage == 'distillation':
            self.pretrained_teacher = self.opt['global']['pretrained_teacher']
            if self.pretrained_teacher:
                state_dict = torch.load(self.pretrained_teacher)
                self.model.model.teacher = load_state_dict(self.model.model.teacher, state_dict, 'model')

            self.pretrained_student = self.opt['global']['pretrained_student']
            if self.pretrained_student:
                state_dict = torch.load(self.pretrained_student)
                self.model.model.student = load_state_dict(self.model.model.student, state_dict, 'model')

        if self.resume:
            state_dict = torch.load(self.resume)
            self.model.model = load_state_dict(self.model.model, state_dict, 'model')
            self.optimizer = load_state_dict(self.optimizer, state_dict, 'optimizer')
            iters = load_state_dict(None, state_dict, 'iters')
            self.last_epoch = iters//len(self.train_dataloader) - 1
