from theseus.cv.classification.losses import LOSS_REGISTRY
from .detr_losses import DETRLosses

LOSS_REGISTRY.register(DETRLosses)