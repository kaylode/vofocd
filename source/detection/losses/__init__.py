from theseus.cv.detection.losses import LOSS_REGISTRY
from .detr_losses import DETRLosses
from .dn_dab_detr_losses import DNDABDETRLosses
from .vofo_losses import VOFOMultiLoss

LOSS_REGISTRY.register(DETRLosses)
LOSS_REGISTRY.register(DNDABDETRLosses)
LOSS_REGISTRY.register(VOFOMultiLoss)