from typing import List, Optional
import os.path as osp
import pandas as pd
from theseus.base.utilities.loggers.observer import LoggerObserver
from theseus.cv.classification.datasets import ClassificationCSVDataset

LOGGER = LoggerObserver.getLogger("main")

class KvasirDataset(ClassificationCSVDataset):
    def __init__(self, image_dir: str, csv_path: str, txt_classnames: str, transform: Optional[List] = None, **kwargs):
        super().__init__(image_dir, csv_path, txt_classnames, transform, **kwargs)

    def _load_data(self):
        with open(self.txt_classnames, "r") as f:
            self.classnames = f.read().splitlines()

        # Mapping between classnames and indices
        for idx, classname in enumerate(self.classnames):
            self.classes_idx[classname] = idx
        self.num_classes = len(self.classnames)

        # Load csv
        df = pd.read_csv(self.csv_path)
        for _, row in df.iterrows():
            video_name, organ_name, label, classification_name = row
            
            if organ_name == 'Lower GI':
                organ_name = 'lower-gi-tract'
            else:
                organ_name = 'upper-gi-tract'

            image_path = osp.join(
                self.image_dir, organ_name, classification_name, label, video_name+'.jpg'
            )
            self.fns.append([image_path, label])
    
    def _calculate_classes_dist(self):
        """
        Calculate distribution of classes
        """
        LOGGER.text("Calculating class distribution...", LoggerObserver.DEBUG)
        self.classes_dist = []

        for _, label in self.fns:        
            self.classes_dist.append(self.classes_idx[label])
        return self.classes_dist