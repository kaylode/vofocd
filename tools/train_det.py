import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("Agg")

from theseus.opt import Opts
<<<<<<< HEAD
from source.detection.pipeline import DetrPipeline

if __name__ == "__main__":
    opts = Opts().parse_args()
    train_pipeline = DetrPipeline(opts)
=======
from source.detection.pipeline import DetPipeline

if __name__ == "__main__":
    opts = Opts().parse_args()
    train_pipeline = DetPipeline(opts)
>>>>>>> 1718613455eb03c6c952899e26d83c7f3d276251
    train_pipeline.fit()
