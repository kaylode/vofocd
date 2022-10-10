from theseus.classification.losses import LOSS_REGISTRY
from .gaussian_loss import GaussianLoss
from .kldiv_loss import TemperatureScaledKLDivLoss

LOSS_REGISTRY.register(GaussianLoss)
LOSS_REGISTRY.register(TemperatureScaledKLDivLoss)