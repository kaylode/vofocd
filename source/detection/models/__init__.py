from theseus.cv.detection.models import MODEL_REGISTRY
from .detr_custom import DETRCustomBackbone
from .faster_rcnn import FasterRCNN
from .efficientdet import EffDet
from .vofo_detr import VOFO_DETR

MODEL_REGISTRY.register(DETRCustomBackbone)
MODEL_REGISTRY.register(FasterRCNN)
MODEL_REGISTRY.register(EffDet)
MODEL_REGISTRY.register(VOFO_DETR)