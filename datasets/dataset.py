import os
import torch
import torch.utils.data as data
from torchvision.transforms import transforms as tf

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from datasets.augmentations.transforms import get_augmentation, Denormalize
from datasets.augmentations.custom import RandomCutmix, RandomMixup


class ImageSet(data.Dataset):
    """
    Reads a folder of images, sup + unsup
    """
    def __init__(self, root_dir, txt_path, image_size, _type='train'):
        self.root_dir = root_dir
        self.txt_path = txt_path
        self.classes = ['pneumonia', 'normal', 'COVID-19']

        self.mapping_classes()
        self.fns = self.load_data()
        self.transforms = get_augmentation(image_size, _type=_type)
        if _type == 'train':
            # MixUp and CutMix
            mixup_transforms = []
            mixup_transforms.append(RandomMixup(self.num_classes, p=1.0, alpha=0.2))
            mixup_transforms.append(RandomCutmix(self.num_classes, p=1.0, alpha=1.0))
            self.mixupcutmix = tf.RandomChoice(mixup_transforms)
        else:
            self.mixupcutmix = None

    def load_data(self):
        fns = []

        with open(self.txt_path) as f:
            data = f.read().splitlines()

        for row in data:
            patient_id, image_name, label, dataset_id = row.split(' ')
            label_id = self.classes_idx[label]
            fns.append((image_name, label_id))

        return fns

    def mapping_classes(self):
        self.classes_idx = {}
        self.idx_classes = {}
        idx = 0
        for cl in self.classes:
            self.classes_idx[cl] = idx
            self.idx_classes[idx] = cl
            idx += 1
        self.num_classes = len(self.classes)
    
    def count_dict(self):
        cnt_dict = {}
        for cl in self.classes:
            num_imgs = len(os.listdir(os.path.join(self.dir,cl)))
            cnt_dict[cl] = num_imgs
        return cnt_dict
    
    def visualize_item(self, index = None, figsize=(15,15)):
        """
        Visualize an image with its bouding boxes by index
        """

        if index is None:
            index = np.random.randint(0,len(self.fns))
        item = self.__getitem__(index)
        img = item['img']
        label = item['target']

        # Denormalize and reverse-tensorize
        normalize = False
        if self.transforms is not None:
            for x in self.transforms.transforms:
                if isinstance(x, tf.Normalize):
                    normalize = True
                    denormalize = Denormalize(mean=x.mean, std=x.std)

        # Denormalize and reverse-tensorize
        if normalize:
            img = denormalize(img = img)

        label = label.numpy().item()
        self.visualize(img, label, figsize = figsize)

    
    def visualize(self, img, label, figsize=(15,15)):
        """
        Visualize an image with its bouding boxes
        """
        fig,ax = plt.subplots(figsize=figsize)

        # Display the image
        ax.imshow(img)
        plt.title(self.classes[int(label)])
        plt.show()

    def plot(self, figsize = (8,8), types = ["freqs"]):
        
        ax = plt.figure(figsize = figsize)
        
        if "freqs" in types:
            cnt_dict = self.count_dict()
            plt.title("Classes Distribution")
            bar1 = plt.bar(list(cnt_dict.keys()), list(cnt_dict.values()), color=[np.random.rand(3,) for i in range(len(self.classes))])
            for rect in bar1:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')
        
        plt.show()
        
        
    def __len__(self):
        return len(self.fns)
    
    def __str__(self):
        s1 = "Number of samples: " + str(len(self.fns)) + '\n'
        s2 = "Number of classes: " + str(len(self.classes)) + '\n'
        return s1 + s2

    def load_image(self, image_id):
        img_path = os.path.join(self.root_dir, image_id+'.jpg')
        img = Image.open(img_path).convert('RGB')
        aug_img = self.transforms(img)
        return aug_img, img

    def __getitem__(self, index):
        image_id, label = self.fns[index]
        img, ori_img = self.load_image(image_id)
        if not isinstance(label, list):
            label  = torch.LongTensor([label])
        else:
            label  = torch.LongTensor(label)

        return {
            "img" : img,
            'ori_img': ori_img,
            "target" : label}

    def collate_fn(self, batch):
        ori_imgs = [s['ori_img'] for s in batch]
        imgs = torch.stack([s['img'] for s in batch])
        targets = torch.stack([s['target'] for s in batch])

        if self.mixupcutmix is not None:
            imgs, targets = self.mixupcutmix(imgs, targets.squeeze(1))

        return {
            'ori_imgs': ori_imgs,
            'imgs': imgs,
            'targets': targets
        }
