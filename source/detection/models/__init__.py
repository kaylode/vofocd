from source.classification.models import MODEL_REGISTRY
from .detr_convnext import DETRConvnext
from .wrapper import ModelWithLossandPostprocess

MODEL_REGISTRY.register(DETRConvnext)
