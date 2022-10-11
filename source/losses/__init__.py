from theseus.classification.losses import LOSS_REGISTRY
from .gaussian_loss import GaussianLoss
from .kldiv_loss import TemperatureScaledKLDivLoss
from .medtex_loss import MedTEXCELoss

LOSS_REGISTRY.register(GaussianLoss)
LOSS_REGISTRY.register(MedTEXCELoss)
LOSS_REGISTRY.register(TemperatureScaledKLDivLoss)