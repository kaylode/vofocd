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

    def _load_mask(self, label_path):
          mask = Image.open(label_path).convert('RGB')
          mask = np.array(mask)[:,:,::-1] # (H,W,3)
          mask = np.argmax(mask, axis=-1)  # (H,W) with each pixel value represent one class
          return mask

    def _encode_masks(self, masks):
        """
        Input masks from _load_mask(), but in shape [B, H, W]
        Output should be one-hot encoding of segmentation masks [B, NC, H, W]
        """
        
        one_hot = torch.nn.functional.one_hot(masks.long(), num_classes=self.num_classes) # (B,H,W,NC)
        one_hot = one_hot.permute(0, 3, 1, 2) # (B,NC,H,W)
        return one_hot.float()

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

