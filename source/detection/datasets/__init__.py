from theseus.base.datasets import DATALOADER_REGISTRY, DATASET_REGISTRY

from .vocal import VocalDetectionDataset
from .vocal_multi import VocalMultiDataset

DATASET_REGISTRY.register(VocalDetectionDataset)
DATASET_REGISTRY.register(VocalMultiDataset)
