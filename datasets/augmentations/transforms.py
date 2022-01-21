import numpy as np
from torchvision.transforms import transforms as tf
from torchvision.transforms import RandAugment
from .custom import RandomCutmix, RandomMixup

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class Denormalize(object):
    """
    Denormalize image and boxes for visualization
    """
    def __init__(self, mean = MEAN, std = STD, **kwargs):
        self.mean = mean
        self.std = std
        
    def __call__(self, img, **kwargs):
        """
        :param img: (tensor) image to be denormalized
        :param box: (list of tensor) bounding boxes to be denormalized, by multiplying them with image's width and heights. Format: (x,y,width,height)
        """
        mean = np.array(self.mean)
        std = np.array(self.std)
        img_show = img.numpy().squeeze().transpose((1,2,0))
        img_show = (img_show * std+mean)
        img_show = np.clip(img_show,0,1)
        return img_show

def get_augmentation(image_size, _type='train'):
    train_transforms = tf.Compose([
        tf.Resize((image_size, image_size)),
        tf.RandomResizedCrop((224, 224)),
        RandAugment(num_ops = 2, magnitude = 9, num_magnitude_bins = 31), 
        tf.ToTensor(),
        tf.Normalize(mean=MEAN, std=STD),
    ])

    val_transforms = tf.Compose([
        tf.Resize((image_size, image_size)),
        tf.ToTensor(),
        tf.Normalize(mean=MEAN, std=STD),
    ])
    
    return train_transforms if _type == 'train' else val_transforms