from theseus.classification.models import MODEL_REGISTRY

from .med_tex import MedTEXStudent, MedTEXTeacher, MedTEXFramework

MODEL_REGISTRY.register(MedTEXTeacher)
MODEL_REGISTRY.register(MedTEXStudent)
MODEL_REGISTRY.register(MedTEXFramework)
