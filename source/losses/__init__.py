from theseus.classification.losses import LOSS_REGISTRY
from .kd_loss import MedTEXLoss

LOSS_REGISTRY.register(MedTEXLoss)