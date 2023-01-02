from theseus.cv.classification.callbacks import CALLBACKS_REGISTRY
from .medtex_visualizer import MedTEXVisualizationCallbacks

CALLBACKS_REGISTRY.register(MedTEXVisualizationCallbacks)