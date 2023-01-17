import matplotlib as mpl

mpl.use("Agg")
from theseus.opt import Opts
from isbi.source.classification.pipelines.distill_pipeline import ClassificationPipeline

if __name__ == "__main__":
    opts = Opts().parse_args()
    val_pipeline = ClassificationPipeline(opts)
    val_pipeline.evaluate()
