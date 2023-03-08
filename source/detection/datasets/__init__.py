from theseus.base.datasets import DATALOADER_REGISTRY, DATASET_REGISTRY

from .vocal import VocalDetectionDataset
from .vocal_multi import VocalMultiDataset
from .vocal_mask import VocalMaskDataset
DATASET_REGISTRY.register(VocalDetectionDataset)
DATASET_REGISTRY.register(VocalMultiDataset)
DATASET_REGISTRY.register(VocalMaskDataset)
