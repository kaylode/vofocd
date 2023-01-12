from theseus.base.datasets import DATALOADER_REGISTRY, DATASET_REGISTRY

from .vocal import VocalDetectionDataset
from .coco import COCODataset

DATASET_REGISTRY.register(VocalDetectionDataset)
DATASET_REGISTRY.register(COCODataset)
