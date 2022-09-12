import os
import cv2
from scipy import stats
import numpy as np
from tqdm import tqdm

import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

import PIL

dark_value=0.5

spath = '/home/AD/rraina/pain/Frustration/generated_faces/8.6_cropMat/'
target = '{}_darken{}/'.format(spath[:-1], dark_value)

class darken(object):
    """Darken image by parameter p
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        image = np.asarray(image)
        image = (image * self.p).astype(np.uint8)
        image = PIL.Image.fromarray(np.uint8(image))
        image = transforms.ToTensor()(image)
        image = transforms.ToPILImage()(image)
        return image

if not os.path.exists(target):
    os.mkdir(target)

for d in tqdm(os.listdir(spath)): # spath = './UNBCMcMaster_cropped/Images'
    if d[-3:]!='png' and d[-3:]!='jpg':
        continue
    name = os.path.join(spath,d)
    image = cv2.imread(name)

    darkened_image = darken(0.5)(image)
    darkened_image = np.array(darkened_image)

    cv2.imwrite(os.path.join(target,d), darkened_image)