from .map import MeanAveragePrecision
from .precision_recall import DetectionPrecisionRecall
from .errorcases import DetectionErrorCases

from theseus.cv.classification.metrics import METRIC_REGISTRY
METRIC_REGISTRY.register(MeanAveragePrecision)
METRIC_REGISTRY.register(DetectionErrorCases)