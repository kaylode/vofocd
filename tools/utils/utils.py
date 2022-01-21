import torch
import torch.nn as nn
import torchvision
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

def one_hot_embedding(labels, num_classes):
    '''
    Embedding labels to one-hot form.

    :param labels: (LongTensor) class labels, sized [N,].
    :param num_classes: (int) number of classes.
    :return: (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]            # [N,D]


def draw_image_gradcam(image, mask, text, image_name=None, figsize=(10,10)):

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(image)
    cam = cam / np.max(cam)
    img_cam = np.uint8(255 * cam)
    img_cam = cv2.cvtColor(img_cam, cv2.COLOR_BGR2RGB)

    plt.close('all')
    fig = plt.figure(figsize=figsize)

    # Display the image
    plt.imshow(img_cam)
    plt.axis('off')
    fig.text(.5, .05, text, ha='center')

    if image_name:
        plt.savefig(image_name, bbox_inches='tight')

    return fig