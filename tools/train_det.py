import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("Agg")

from theseus.opt import Opts
from source.detection.pipeline import DetPipeline, DetPipelineWithIntegratedLoss

if __name__ == "__main__":
    opts = Opts().parse_args()

    model_name = opts['model']['args']['model_name']
    if 'detr' in model_name:    
        train_pipeline = DetPipeline(opts)
    elif model_name in ['faster_rcnn', 'mask_rcnn']:
        train_pipeline = DetPipelineWithIntegratedLoss(opts)
    else:
        raise ValueError()
    train_pipeline.fit()