from .coco import COCODataset
from theseus.cv.classification.datasets import DATASET_REGISTRY

DATASET_REGISTRY.register(COCODataset)