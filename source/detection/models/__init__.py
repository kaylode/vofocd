from theseus.cv.detection.models import MODEL_REGISTRY
from .detr_custom import DETRCustomBackbone
from .faster_rcnn import FasterRCNN

MODEL_REGISTRY.register(DETRCustomBackbone)
MODEL_REGISTRY.register(FasterRCNN)
