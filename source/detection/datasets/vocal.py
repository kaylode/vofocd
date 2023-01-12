import os
from typing import List, Optional

import cv2
import numpy as np
from .coco import COCODataset

class VocalDetectionDataset(COCODataset):
    def __init__(
        self,
        image_dir: str,
        label_path: str,
        transform: Optional[List] = None,
        **kwargs
    ):
        super().__init__(image_dir, label_path, transform, **kwargs)

    def load_image(self, image_index):
        image_info = self.fns.loadImgs(self.image_ids[image_index])[0]

        _, image_name = image_info["file_name"].split('_')
        path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(path)
        height, width, c = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        return image, image_info["file_name"], width, height