from typing import List, Optional
import os.path as osp
import pandas as pd
import os
import cv2
import json
import numpy as np
from theseus.base.utilities.loggers.observer import LoggerObserver
from theseus.cv.classification.datasets import ClassificationDataset

LOGGER = LoggerObserver.getLogger("main")

class VocalClassificationDataset(ClassificationDataset):
    def __init__(
        self, 
        image_dir: str,
        json_path: str, # coco format file
        transform: Optional[List] = None,
        **kwargs):
        super().__init__(**kwargs)
        self.image_dir = image_dir
        self.transform = transform
        self.json_path = json_path
        self._load_data()


    def _load_data(self):
        with open(self.json_path, 'r') as f:
            data = json.load(f)['images']

        classnames = []
        for item in data:
            classname, filename = item['file_name'].split('_')
            self.fns.append([osp.join(self.image_dir, filename), classname])
            classnames.append(classname)
        
        self.classnames =  sorted(list(set(classnames)))
        self.classes_idx = {c:i for i,c in enumerate(self.classnames)}
        
    def _calculate_classes_dist(self):
        """
        Calculate distribution of classes
        """
        LOGGER.text("Calculating class distribution...", LoggerObserver.DEBUG)
        self.classes_dist = []

        for _, label in self.fns:        
            self.classes_dist.append(self.classes_idx[label])
        return self.classes_dist