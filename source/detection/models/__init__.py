from theseus.cv.detection.models import MODEL_REGISTRY
from .detr_custom import DETRCustomBackbone

MODEL_REGISTRY.register(DETRCustomBackbone)