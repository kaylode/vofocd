from typing import Dict
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision.transforms import functional as TFF

from theseus.base.callbacks.base_callbacks import Callbacks
from theseus.base.utilities.loggers.observer import LoggerObserver
from theseus.base.utilities.visualization.visualizer import Visualizer


LOGGER = LoggerObserver.getLogger("main")

class MedTEXVisualizationCallbacks(Callbacks):
    """
    Callbacks for visualizing stuff during training
    Features:
        - Visualize datasets; plot model architecture, analyze datasets in sanity check
        - Visualize prediction at every end of validation
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.visualizer = Visualizer()

    @torch.no_grad() #enable grad for CAM
    def on_val_epoch_end(self, logs: Dict=None):
    # def on_train_epoch_start(self, logs: Dict=None):
        """
        After finish validation
        """

        iters = logs['iters']
        last_batch = logs['last_batch']
        model = self.params['trainer'].model
        valloader = self.params['trainer'].valloader

        # Vizualize model predictions
        LOGGER.text("Visualizing pixel map predictions...", level=LoggerObserver.DEBUG)

        images = last_batch["inputs"]
        targets = last_batch["targets"]
        model.eval()
        
        ## Get prediction on last batch
        outputs = model.model.get_prediction(last_batch, device=model.device)
        label_indices = outputs['labels']
        pixel_maps = outputs['pixel_maps']
        scores = outputs['confidences']
            
        pred_batch = []
        for idx in range(len(images)):
            image = images[idx]
            target = targets[idx].item()
            label = label_indices[idx]
            score = scores[idx]
            pixel_map = pixel_maps[idx]

            np_image = pixel_map.cpu().numpy().transpose(1,2,0)

            image_weight = 0.7
            img_show = self.visualizer.denormalize(image)
            cam = (1 - image_weight) * np_image + image_weight * img_show
            cam = np.uint8(255 * cam)

            self.visualizer.set_image(cam)
            if valloader.dataset.classnames is not None:
                label = valloader.dataset.classnames[label]
                target = valloader.dataset.classnames[target]

            if label == target:
                color = [0,1,0]
            else:
                color = [1,0,0]

            self.visualizer.draw_label(
                f"GT: {target}\nP: {label}\nC: {score:.4f}", 
                fontColor=color, 
                fontScale=0.8,
                thickness=2,
                outline=None,
                offset=100
            )
            
            pred_img = self.visualizer.get_image()
            pred_img = TFF.to_tensor(pred_img)
            pred_batch.append(pred_img)

            if idx == 63: # limit number of images
                break

        # Prediction images
        pred_grid_img = self.visualizer.make_grid(pred_batch)
        fig = plt.figure(figsize=(10,10))
        plt.imshow(pred_grid_img)
        plt.axis("off")
        plt.tight_layout(pad=0)
        LOGGER.log([{
            'tag': "Validation/prediction",
            'value': fig,
            'type': LoggerObserver.FIGURE,
            'kwargs': {
                'step': iters
            }
        }])

        plt.cla()   # Clear axis
        plt.clf()   # Clear figure
        plt.close()