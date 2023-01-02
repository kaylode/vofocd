import matplotlib as mpl

mpl.use("Agg")
from theseus.opt import Opts
from source.detection.pipeline import DetPipeline

if __name__ == "__main__":
    opts = Opts().parse_args()
    val_pipeline = DetPipeline(opts)
    val_pipeline.evaluate()
