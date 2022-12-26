from source.classification.models import MODEL_REGISTRY
from .detr_convnext import DETRConvnext

MODEL_REGISTRY.register(DETRConvnext)
