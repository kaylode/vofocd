from theseus.base.datasets import DATALOADER_REGISTRY, DATASET_REGISTRY

from .vocal import VocalDetectionDataset

DATASET_REGISTRY.register(VocalDetectionDataset)
