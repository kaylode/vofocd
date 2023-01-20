from .kvasir import KvasirDataset
from .vocal import VocalClassificationDataset
from theseus.cv.classification.datasets import DATALOADER_REGISTRY, DATASET_REGISTRY

DATASET_REGISTRY.register(KvasirDataset)
DATASET_REGISTRY.register(VocalClassificationDataset)