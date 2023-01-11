from source.semantic.models import MODEL_REGISTRY
from .detrsegm_convnext import DETRSegmConvnext

MODEL_REGISTRY.register(DETRSegmConvnext)