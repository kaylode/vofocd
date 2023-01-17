from .kvasir import KvasirDataset
from theseus.cv.classification.datasets import DATALOADER_REGISTRY, DATASET_REGISTRY

DATASET_REGISTRY.register(KvasirDataset)