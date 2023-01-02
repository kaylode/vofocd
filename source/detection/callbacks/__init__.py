from .visualization import DetectionVisualizerCallbacks
from theseus.cv.classification.callbacks import CALLBACKS_REGISTRY

CALLBACKS_REGISTRY.register(DetectionVisualizerCallbacks)