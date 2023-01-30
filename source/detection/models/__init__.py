from theseus.cv.detection.models import MODEL_REGISTRY
from .detr_custom import DETRCustomBackbone
from .faster_rcnn import FasterRCNN
from .efficientdet import EffDet
from .yolov5 import YoloV5

MODEL_REGISTRY.register(DETRCustomBackbone)
MODEL_REGISTRY.register(FasterRCNN)
MODEL_REGISTRY.register(EffDet)
MODEL_REGISTRY.register(YoloV5)