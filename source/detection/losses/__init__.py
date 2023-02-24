from theseus.cv.detection.losses import LOSS_REGISTRY
from .detr_losses import DETRLosses
from .vofo_losses import VOFOMultiLoss

LOSS_REGISTRY.register(DETRLosses)
LOSS_REGISTRY.register(VOFOMultiLoss)