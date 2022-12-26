from .coco import DetectionDataset
from theseus.cv.classification.models import MODEL_REGISTRY

MODEL_REGISTRY.register(DetectionDataset)