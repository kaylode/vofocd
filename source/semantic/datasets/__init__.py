from theseus.cv.semantic.models import DATASET_REGISTRY
from .food import FoodCSVDataset

DATASET_REGISTRY.register(FoodCSVDataset)