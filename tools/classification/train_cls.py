import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("Agg")

from theseus.opt import Opts
from source.classification.pipeline import ClsPipeline

if __name__ == "__main__":
    opts = Opts().parse_args()
    train_pipeline = ClsPipeline(opts)
    train_pipeline.fit()
