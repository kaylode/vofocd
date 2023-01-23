import matplotlib as mpl

mpl.use("Agg")
from theseus.opt import Opts
from source.detection.pipeline import DetPipeline, DetPipelineWithIntegratedLoss

if __name__ == "__main__":
    # opts = Opts().parse_args()
    # val_pipeline = DetPipeline(opts)
    # val_pipeline.evaluate()
    opts = Opts().parse_args()

    model_name = opts['model']['args']['model_name']
    if 'detr' in model_name:    
        val_pipeline = DetPipeline(opts)
    elif model_name in ['faster_rcnn', 'mask_rcnn']:
        val_pipeline = DetPipelineWithIntegratedLoss(opts)
    else:
        raise ValueError()
    
    val_pipeline.evaluate()