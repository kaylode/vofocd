import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("Agg")

from theseus.opt import Opts
from source.detection.pipeline import DetrPipeline

if __name__ == "__main__":
    opts = Opts().parse_args()
    train_pipeline = DetrPipeline(opts)
    train_pipeline.fit()
