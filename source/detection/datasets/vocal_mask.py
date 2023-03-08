from typing import List, Optional, Dict
import torch

import cv2
import numpy as np
from .vocal import VocalDetectionDataset

class VocalMaskDataset(VocalDetectionDataset):
    def __init__(
        self,
        image_dir: str,
        label_path: str,
        transform: Optional[List] = None,
        include_background: bool= False,
        binary_thresholding: float = 0.0, # 1.0 means not used
        **kwargs
    ):
        super().__init__(image_dir, label_path, transform, include_background,**kwargs)
        self.binary_thresholding = int(binary_thresholding*255)

    def create_mask_version(self, tensor_image, threshold=0.2, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        denorm = tensor_image.numpy().transpose((1, 2, 0))*np.array(std)+np.array(mean)
        grayscale = cv2.cvtColor(np.uint8(denorm[...,::-1]*255), cv2.COLOR_RGB2GRAY)
        _,mask = cv2.threshold(grayscale, threshold, 255, cv2.THRESH_BINARY_INV)
        mask = torch.Tensor(mask)
        return mask

    def __getitem__(self, idx: int) -> Dict:
        """
        Get one item
        """
        (
            image,
            boxes,
            labels,
            img_id,
            img_name,
            ori_width,
            ori_height,
        ) = self.load_image_and_boxes(idx)

        if self.transform:
            item = self.transform(image=image, bboxes=boxes, class_labels=labels)
            # Normalize
            image = item["image"]
            boxes = item["bboxes"]
            labels = item["class_labels"]

        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self.image_ids))
        
        bin_mask = self.create_mask_version(image, threshold= self.binary_thresholding)

        labels = torch.LongTensor(labels)  # starts from 1
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        return {
            "img": image,
            "target": target,
            "img_id": img_id,
            "img_name": img_name,
            "ori_size": [ori_width, ori_height],
            'bin_mask': bin_mask
        }
    
    def collate_fn(self, batch):
        imgs = torch.stack([s["img"] for s in batch])
        targets = [s["target"] for s in batch]
        img_ids = [s["img_id"] for s in batch]
        img_names = [s["img_name"] for s in batch]
        ori_sizes = [s["ori_size"] for s in batch]
        bin_masks = torch.stack([s["bin_mask"] for s in batch])

        return {
            "inputs": imgs,
            "targets": targets,
            "img_ids": img_ids,
            "img_names": img_names,
            "ori_sizes": ori_sizes,
            "bin_masks": bin_masks,
        }