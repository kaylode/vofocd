import matplotlib as mpl

mpl.use("Agg")
import os
import os.path as osp
import json
import cv2
import torch

from theseus.base.pipeline import BaseTestPipeline
from theseus.base.utilities.loggers import LoggerObserver
from theseus.cv.base.utilities.visualization.visualizer import Visualizer
from theseus.cv.detection.augmentations import TRANSFORM_REGISTRY
from source.detection.datasets import DATALOADER_REGISTRY, DATASET_REGISTRY
from source.detection.models import MODEL_REGISTRY
from theseus.opt import Config, Opts


class TestPipeline(BaseTestPipeline):
    def __init__(self, opt: Config):

        super(TestPipeline, self).__init__(opt)
        self.opt = opt

    def init_globals(self):
        super().init_globals()

    def init_registry(self):
        self.model_registry = MODEL_REGISTRY
        self.dataset_registry = DATASET_REGISTRY
        self.dataloader_registry = DATALOADER_REGISTRY
        self.transform_registry = TRANSFORM_REGISTRY
        self.logger.text("Overidding registry in pipeline...", LoggerObserver.INFO)

    @torch.no_grad()
    def inference(self):
        self.init_pipeline()
        self.logger.text("Inferencing...", level=LoggerObserver.INFO)

        visualizer = Visualizer()
        visualizer.set_classnames(self.dataset.classnames)

        saved_result_dir = os.path.join(self.savedir, "json")
        saved_overlay_dir = os.path.join(self.savedir, "overlays")

        os.makedirs(saved_result_dir, exist_ok=True)
        os.makedirs(saved_overlay_dir, exist_ok=True)

        self.model.eval()

        result_dict = []

        for idx, batch in enumerate(self.dataloader):
            inputs = batch["inputs"]
            img_names = batch["img_names"]
            ori_sizes = batch["ori_sizes"]
            
            outputs = self.model.get_prediction(batch, self.device)
            boxes = outputs["boxes"]
            labels = outputs["labels"]
            confidences = outputs["confidences"]

            for (inpt, box, name, label, conf) in zip(
                inputs, boxes, img_names, labels, confidences
            ):
                img_show = visualizer.denormalize(inpt, mean=[0,0,0], std=[1,1,1])
                visualizer.set_image(img_show.copy())
                visualizer.draw_bbox(box, labels=label,scores=conf)
                savepath = os.path.join(saved_overlay_dir, name)
                img_show = visualizer.get_image()
                cv2.imwrite(savepath, img_show[:,:,::-1])
                self.logger.text(f"Save image at {savepath}", level=LoggerObserver.INFO)

                pred_dict = {
                    "image_name": name,
                    "category_id": label,
                    "bbox": box,
                    "score": conf
                }
                result_dict.append(pred_dict)
            
        with open(osp.join(saved_result_dir, 'result.json'), 'w') as f:
            json.dump(result_dict, f)
        self.logger.text(f"Save json at {saved_result_dir}", level=LoggerObserver.INFO)
          
if __name__ == "__main__":
    opts = Opts().parse_args()
    val_pipeline = TestPipeline(opts)
    val_pipeline.inference()
