import os
import os.path as osp
from typing import List, Optional, Dict
import torch

import json
import cv2
import numpy as np
from pycocotools.coco import COCO
from theseus.cv.detection.datasets.coco import COCODataset

class VocalMultiDataset(COCODataset):
    def __init__(
        self,
        image_dir: str,
        label_path: str,
        transform: Optional[List] = None,
        **kwargs
    ):
        super().__init__(image_dir, label_path, transform, **kwargs)

    def _load_data(self):
        self.fns = COCO(self.label_path)
        self.image_ids = self.fns.getImgIds()
        # load class names (name -> label)
        categories = self.fns.loadCats(self.fns.getCatIds())
        categories.sort(key=lambda x: x["id"])

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

        # Objects
        with open(self.label_path, 'r') as f:
            data = json.load(f)['images']
        img_classnames = []
        for item in data:
            classname, _ = item['file_name'].split('_')
            img_classnames.append(classname)
        
        self.img_classnames = sorted(list(set(img_classnames)))
        self.img_classes_idx = {c:i for i,c in enumerate(self.img_classnames)}

    def load_image(self, image_index):
        image_info = self.fns.loadImgs(self.image_ids[image_index])[0]
        img_classname, image_name = image_info["file_name"].split('_')
        path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(path)
        height, width, c = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        img_class_idx = self.img_classes_idx[img_classname]
        return image, image_info["file_name"], width, height, img_class_idx
    
    def load_image_and_boxes(self, idx):
        """
        Load an image and its boxes, also do scaling here
        """
        img, img_name, ori_width, ori_height, img_class_idx = self.load_image(idx)
        img_id = self.image_ids[idx]
        annot = self.load_annotations(idx, ori_width, ori_height)
        box = annot[:, :4]
        label = annot[:, -1]

        return img, box, label, img_id, img_name, ori_width, ori_height, img_class_idx
    
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
            img_class_idx
        ) = self.load_image_and_boxes(idx)

        if self.transform:
            item = self.transform(image=image, bboxes=boxes, class_labels=labels)
            # Normalize
            image = item["image"]
            boxes = item["bboxes"]
            labels = item["class_labels"]

        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self.image_ids))

        labels = torch.LongTensor(labels)  # starts from 1
        img_label = torch.LongTensor([img_class_idx])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        return {
            "img": image,
            "obj_target": target,
            'img_target': img_label,
            "img_id": img_id,
            "img_name": img_name,
            "ori_size": [ori_width, ori_height],
        }
    
    def collate_fn(self, batch):
        imgs = torch.stack([s["img"] for s in batch])
        targets = [s["obj_target"] for s in batch]
        img_targets = torch.stack([s["img_target"] for s in batch])
        img_ids = [s["img_id"] for s in batch]
        img_names = [s["img_name"] for s in batch]
        ori_sizes = [s["ori_size"] for s in batch]

        return {
            "inputs": imgs,
            "obj_targets": targets,
            "img_targets": img_targets,
            "img_ids": img_ids,
            "img_names": img_names,
            "ori_sizes": ori_sizes,
        }

    def __len__(self) -> int:
        return len(self.image_ids)