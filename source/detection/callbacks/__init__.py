from .visualization import DetectionVisualizerCallbacks
from theseus.cv.classification.callbacks import CALLBACK_REGISTRY

CALLBACK_REGISTRY.register(DetectionVisualizerCallbacks)