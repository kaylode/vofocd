from theseus.cv.classification.models import MODEL_REGISTRY

from .med_tex import MedTEXStudent, MedTEXTeacher, MedTEXFramework
from .kd import KDFramework

MODEL_REGISTRY.register(MedTEXTeacher)
MODEL_REGISTRY.register(MedTEXStudent)
MODEL_REGISTRY.register(MedTEXFramework)
MODEL_REGISTRY.register(KDFramework)