import matplotlib as mpl

mpl.use("Agg")
from theseus.opt import Opts
from source.classification.pipeline import ClsPipeline

if __name__ == "__main__":
    opts = Opts().parse_args()
    val_pipeline = ClsPipeline(opts)
    val_pipeline.evaluate()
