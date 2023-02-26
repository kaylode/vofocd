import os
from typing import List, Optional

import cv2
import numpy as np
from theseus.cv.detection.datasets.coco import COCODataset
from pycocotools.coco import COCO

class VocalDetectionDataset(COCODataset):
    def __init__(
        self,
        image_dir: str,
        label_path: str,
        transform: Optional[List] = None,
        include_background: bool= False,
        **kwargs
    ):
        self.include_background = include_background
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
    
    def _load_data(self):
        self.fns = COCO(self.label_path)
        self.image_ids = self.fns.getImgIds()
        # load class names (name -> label)
        categories = self.fns.loadCats(self.fns.getCatIds())
        categories.sort(key=lambda x: x["id"])

        if self.include_background:
            self.classes = {'background': 0}
            self.idx_mapping = {0:'background'}
            self.classnames = ['background']
        else:
            self.classes = {}
            self.idx_mapping = {}
            self.classnames = []
        
        for c in categories:
            idx = len(self.classes)
            self.classes[c["name"]] = idx
            self.idx_mapping[c["id"]] = idx
            self.classnames.append(c["name"])

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        self.num_classes = len(self.labels)