import torch
import torch.utils.data as data

import numpy as np
from .dataset import ImageSet
from torch.utils.data.sampler import WeightedRandomSampler

def class_imbalance_sampler(labels):
    class_count = torch.bincount(labels.squeeze())
    class_weighting = 1. / class_count
    sample_weights = np.array([class_weighting[t] for t in labels.squeeze()])
    sample_weights = torch.from_numpy(sample_weights)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    return sampler

class ImageLoader(data.DataLoader):
    def __init__(self,  root_dir, txt_path, image_size, type, batch_size):

        self.root_dir = root_dir
        self.dataset = ImageSet(
            root_dir=root_dir, 
            txt_path=txt_path,
            image_size=image_size,
            _type=type
        )

        if  type == 'train':
            labels = torch.LongTensor(self.dataset.classes_dist).unsqueeze(1)
            sampler = class_imbalance_sampler(labels)
        else:
            sampler = None

        self.collate_fn = self.dataset.collate_fn
        
        super(ImageLoader, self).__init__(
            self.dataset,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            sampler=sampler,
            collate_fn=self.collate_fn)

