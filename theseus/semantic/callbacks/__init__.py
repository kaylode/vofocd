from theseus.base.callbacks import CALLBACKS_REGISTRY

from theseus.semantic.callbacks.visualize_callbacks import VisualizerCallbacks

CALLBACKS_REGISTRY.register(VisualizerCallbacks)