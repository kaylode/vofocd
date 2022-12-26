from typing import List, Optional
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
# from theseus.cv.semantic.datasets.dataset import SemanticDataset


class DetectionDataset(nn.Module):
    r"""SemanticCSVDataset multi-labels segmentation dataset
    Reads in .csv file with structure below:
        filename   | label
        ---------- | -----------
        <img1>.jpg | <mask1>.jpg
    image_dir: `str`
        path to directory contains images
    mask_dir: `str`
        path to directory contains masks
    transform: Optional[List]
        transformatin functions
        
    """
    def __init__(
            self, 
            image_dir: str, 
            label_path: str, 
            txt_classnames: str,
            transform: Optional[List] = None,
            **kwargs):
        super(FoodCSVDataset, self).__init__(**kwargs)
        self.image_dir = image_dir
        self.label_path = label_path
        self.transform = transform
        self.txt_classnames = txt_classnames
        self._load_data()

    def _load_data(self):
        """
        Read data from csv and load into memory
        """

        df = pd.read_csv(self.txt_classnames, header=None, sep="\t")
        self.classnames = df[1].tolist()

        # Mapping between classnames and indices
        for idx, classname in enumerate(self.classnames):
            self.classes_idx[classname] = idx
        self.num_classes = len(self.classnames)

        data = json.load(open(self.label_path, 'r'))
        print(data.keys())
        self.fns.append([image_name, mask_name])

     def __getitem__(self, idx):
        
        image, boxes, labels, img_id, img_name, ori_width, ori_height = self.load_augment(idx)
        if self.transforms:
            item = self.transforms(image=image, bboxes=boxes, class_labels=labels)
            # Normalize
            image = item['image']
            boxes = item['bboxes']
            labels = item['class_labels'] 
            boxes = np.array([np.asarray(i) for i in boxes])
            labels = np.array(labels)

        if len(boxes) == 0:
            return self.__getitem__((idx+1)%len(self.image_ids))
        labels = torch.LongTensor(labels) # starts from 1
        boxes = torch.as_tensor(boxes, dtype=torch.float32) 

        target = {}

        if self.box_format == 'yxyx':
            boxes = change_box_order(boxes, 'xyxy2yxyx')

        target['boxes'] = boxes
        target['labels'] = labels
        

        return {
            'img': image,
            'target': target,
            'img_id': img_id,
            'img_name': img_name,
            'ori_size': [ori_width, ori_height]
        }

    def collate_fn(self, batch):
        imgs = torch.stack([i['input'] for i in batch])
        masks = torch.stack([i['target']['mask'] for i in batch])
        img_names = [i['img_name'] for i in batch]
        ori_sizes = [i['ori_size'] for i in batch]
        
        masks = self._encode_masks(masks)
        return {
            'inputs': imgs,
            'targets': masks,
            'img_names': img_names,
            'ori_sizes': ori_sizes
        }


