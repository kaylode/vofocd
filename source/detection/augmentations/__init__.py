from theseus.cv.classification.augmentations import TRANSFORM_REGISTRY
from albumentations import BboxParams
from .bbox_transforms import BoxOrder


TRANSFORM_REGISTRY.register(BboxParams, prefix='Alb')
TRANSFORM_REGISTRY.register(BoxOrder)