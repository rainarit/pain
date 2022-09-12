''' script to predict PSPI from image '''
from __future__ import print_function
from __future__ import division
import argparse

import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

import numpy as np
import time
import os
import copy
from collections import defaultdict, deque
import datetime

import McMasterDataset
import imp
#import sklearn.metrics
imp.reload(McMasterDataset)
from McMasterDataset import *
from vgg_face import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from tqdm import tqdm
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def main():

    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)

    data_transforms = {
        'train': transforms.Compose([
            # BGR2RGB(),
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize([0.3873985 , 0.42637664, 0.53720075], [0.2046528 , 0.19909547, 0.19015081])
        ]),
        'val': transforms.Compose([
            # BGR2RGB(),
            transforms.ToPILImage(),
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize([0.3873985 , 0.42637664, 0.53720075], [0.2046528 , 0.19909547, 0.19015081])
        ]),
        'test': transforms.Compose([
            # BGR2RGB(),
            transforms.ToPILImage(),
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize([0.3873985 , 0.42637664, 0.53720075], [0.2046528 , 0.19909547, 0.19015081])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    image_dir = "/mnt/cube/projects/xiaojing/data/UNBCMcMaster_cropped/Images0.3"
    label_dir = "/mnt/cube/projects/xiaojing/data/UNBCMcMaster"

    subjects = []
    for d in next(os.walk(image_dir))[1]:
        subjects.append(d[:3])

    val_subj, test_subj = np.random.choice(subjects, 2)

    datasets_dict = {x: McMasterDataset(image_dir, label_dir, val_subj, test_subj, x, data_transforms[x]) for x in ['train', 'val', 'test']}

    total_set = torch.utils.data.ConcatDataset([datasets_dict['train']])
    total_dataloader = torch.utils.data.DataLoader(total_set, batch_size=1)

    mean = 0.
    std = 0.
    for images in tqdm(total_dataloader):
        images = images['image']
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(total_dataloader.dataset)
    std /= len(total_dataloader.dataset)

    print(mean)
    print(std)


if __name__ == '__main__':
    main()