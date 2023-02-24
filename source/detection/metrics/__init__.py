from .multimetrics import MultiMetricWrapper
from theseus.cv.detection.metrics import METRIC_REGISTRY

METRIC_REGISTRY.register(MultiMetricWrapper)