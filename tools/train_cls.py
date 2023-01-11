import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("Agg")

from theseus.opt import Opts
from source.classification.pipeline import ClassificationPipeline

if __name__ == "__main__":
    opts = Opts().parse_args()
    train_pipeline = ClassificationPipeline(opts)
    train_pipeline.fit()
