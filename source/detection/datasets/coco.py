from typing import List, Optional
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from .base import DetectionDataset

class COCODataset(DetectionDataset):
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
        super(COCODataset, self).__init__(**kwargs)
        self.image_dir = image_dir
        self.label_path = label_path
        self.transform = transform
        self.txt_classnames = txt_classnames
        self._load_data()

    def _load_data(self):
        self.classnames = open(self.txt_classnames, 'r').read().splitlines()
        # Mapping between classnames and indices
        for idx, classname in enumerate(self.classnames):
            self.classes_idx[classname] = idx
        self.num_classes = len(self.classnames)
        self.fns = COCO(self.label_path)
        self.image_ids = self.fns.getImgIds()

    def load_image(self, image_index):
        image_info = self.fns.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.image_dir, image_info['file_name'])
        image = cv2.imread(path)
        height, width, c = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        return image, image_info['file_name'], width, height

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.fns.getAnnIds(
            imgIds=self.image_ids[image_index], 
            iscrowd=None
        )
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations
        
        # parse annotations
        coco_annotations = self.fns.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] <= 2 or a['bbox'][3] <= 2:
                continue
            
            # some annotations have wrong coordinate
            if a['bbox'][0] < 0 or a['bbox'][1] < 0:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox'] # xywh
            annotation[0, 4] = self.idx_mapping[a['category_id']]
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

    def load_image_and_boxes(self, idx):
        """
        Load an image and its boxes, also do scaling here
        """
        img, img_name, ori_width, ori_height  = self.load_image(idx)
        img_id = self.image_ids[idx]
        annot = self.load_annotations(idx)
        box = annot[:, :4]
        label = annot[:, -1]

        return img, box, label, img_id, img_name, ori_width, ori_height

    def collate_fn(self, batch):
        imgs = torch.stack([s['img'] for s in batch])
        targets = [s['target'] for s in batch]
        img_ids = [s['img_id'] for s in batch]
        img_names = [s['img_name'] for s in batch]
        img_scales = torch.tensor([1.0]*len(batch), dtype=torch.float)
        img_sizes = torch.tensor([imgs[0].shape[-2:]]*len(batch), dtype=torch.float)
        ori_sizes = [s['ori_size'] for s in batch]

        return {
            'imgs': imgs, 
            'targets': targets, 
            'img_ids': img_ids,
            'img_names': img_names,
            'img_sizes': img_sizes, 
            'img_scales': img_scales,
            'ori_sizes': ori_sizes
        }

    def __len__(self) -> int:
        return len(self.image_ids)


