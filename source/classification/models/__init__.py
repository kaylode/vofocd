from theseus.cv.classification.models import MODEL_REGISTRY

from .medtex import MedTEXStudent, MedTEXTeacher, MedTEXFramework
from .medtex.kd import KDFramework
from .backbone.convnext import ConvNeXt
from .backbone.resnet50 import Resnet

MODEL_REGISTRY.register(MedTEXTeacher)
MODEL_REGISTRY.register(MedTEXStudent)
MODEL_REGISTRY.register(MedTEXFramework)
MODEL_REGISTRY.register(KDFramework)
MODEL_REGISTRY.register(ConvNeXt)
MODEL_REGISTRY.register(Resnet)