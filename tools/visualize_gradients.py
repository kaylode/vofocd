import matplotlib as mpl
mpl.use("Agg")
from theseus.opt import Opts

import numpy as np
import os.path as osp
from tqdm import tqdm
from theseus.opt import Config
from source.models import MODEL_REGISTRY
from torchvision.transforms import functional as TFF

from theseus.utilities.loggers import LoggerObserver
from theseus.base.pipeline import BaseTestPipeline
from theseus.classification.utilities.gradcam import CAMWrapper, show_cam_on_image, model_last_layers
from theseus.utilities.visualization.visualizer import Visualizer

import cv2
import matplotlib.pyplot as plt

LOGGER = LoggerObserver.getLogger("main")

model_last_layers.update({
    'convnext_nano': ['stages', -1]
})

def visualize_gradients(gradient, save_path):
    # gradients (1, H, W), sigmoided

    fig = plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.imshow(grid_img)
    plt.tight_layout(pad=0)


class TestPipeline(BaseTestPipeline):
    def __init__(
            self,
            opt: Config
        ):

        super(TestPipeline, self).__init__(opt)
        self.opt = opt

    def init_globals(self):
        super().init_globals()

    def init_registry(self):
        super().init_registry()
        self.model_registry = MODEL_REGISTRY
        self.logger.text(
            "Overidding registry in pipeline...", LoggerObserver.INFO
        )

    def inference(self):
        self.init_pipeline()
        self.logger.text("Inferencing...", level=LoggerObserver.INFO)

        df_dict = {
            'filename': [],
            'label': [],
            'score': []
        }
        
        self.visualizer = Visualizer()
        self.model.zero_grad()

        ## Calculate GradCAM and Grad Class Activation Mapping and
        model_name = "convnext_nano"

        for idx, batch in enumerate(tqdm(self.dataloader)):
            images = batch['inputs']
            img_names = batch['img_names']
            outputs = self.model.get_prediction(batch, self.device)

            try:
                grad_cam = CAMWrapper.get_method(
                    name='gradcam', 
                    model=self.model.student.model, 
                    model_name=model_name, use_cuda=next(self.model.parameters()).is_cuda)
                
                grayscale_cams, label_indices, scores = grad_cam(images, return_probs=True)

            except Exception as e:
                LOGGER.text("Cannot calculate GradCAM", level=LoggerObserver.ERROR)
                LOGGER.text(e, level=LoggerObserver.ERROR)
                return

            preds = outputs['names']
            probs = outputs['confidences']
            pixel_maps = outputs['pixel_maps'].cpu().numpy()

            for idx in range(len(grayscale_cams)):
                image = images[idx]
                label = label_indices[idx]
                grayscale_cam = grayscale_cams[idx, :]
                image_name = img_names[idx]
                pixel_map = pixel_maps[idx].squeeze()
                pixel_map = (pixel_map*255).astype(np.uint8)

                img_show = self.visualizer.denormalize(image)
                if self.dataloader.dataset.classnames is not None:
                    label = self.dataloader.dataset.classnames[label]

                img_cam =show_cam_on_image(img_show, grayscale_cam, use_rgb=True, image_weight=0.8)
                
                # img_grad = pixel_map
                # img_grad = (img_show*pixel_map).astype(np.uint8)

                # img_cam = TFF.to_tensor(img_cam)
                fig = plt.figure(figsize=(3,3))
                plt.imshow(img_cam)
                plt.axis("off")
                plt.tight_layout(pad=0)
                plt.savefig(osp.join(self.savedir, image_name[:-4]+'_cam.jpg'))

                # fig = plt.figure(figsize=(3,3))
                # plt.imshow(img_grad, cmap='gray')
                # plt.axis("off")
                # plt.tight_layout(pad=0)
                # plt.savefig(osp.join(self.savedir, image_name[:-4]+'_grad.jpg'))



                plt.cla()   # Clear axis
                plt.clf()   # Clear figure
                plt.close()
                # asd


if __name__ == '__main__':
    opts = Opts().parse_args()
    val_pipeline = TestPipeline(opts)
    val_pipeline.inference()

        
