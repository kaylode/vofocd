from .base_model import BaseModel
from .classifier import Classifier
from .models import BaseTimmModel
from .covidnet import CovidNet

def get_model(config, num_classes):

    if config.model_name == 'covidnet':
        net = CovidNet('large', num_classes=num_classes)
    else:
        net = BaseTimmModel(
            name=config.model_name, 
            num_classes=num_classes)
    return net