import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("Agg")

from theseus.opt import Opts
from source.models.med_tex import (
    MedTEXStudent, MedTEXTeacher
)
from source.pipeline import Pipeline

if __name__ == "__main__":
    opts = Opts().parse_args()
    train_pipeline = Pipeline(opts)
    train_pipeline.fit()
