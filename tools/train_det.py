import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("Agg")

from theseus.opt import Opts
from source.detection.pipeline import DetPipeline, DetPipelineWithIntegratedLoss

if __name__ == "__main__":
    opts = Opts().parse_args()

    model_name = opts['model']['args'].get('model_name', None)
    if model_name in ['detr', None]:    
        train_pipeline = DetPipeline(opts)
    elif model_name in ['faster_rcnn', 'mask_rcnn', 'efficientdet']:
        train_pipeline = DetPipelineWithIntegratedLoss(opts)
    else:
        raise ValueError()
    train_pipeline.fit()