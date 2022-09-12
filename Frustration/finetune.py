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
from PIL import Image

import numpy as np
import time
import os
import copy
from collections import defaultdict, deque
import datetime

import McMasterDataset
from RacialImageDataset import *
import imp
#import sklearn.metrics
imp.reload(McMasterDataset)
from McMasterDataset import *
from vgg_face import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from tqdm import tqdm
import random
import csv
import glob

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='/home/AD/rraina/pain/Frustration/generated_faces/LightSkin__centercropped_zscore_shoulder', metavar='DIR', type=str, help='path to dataset')
parser.add_argument('--batch-size', default=32, type=int, metavar='N', help='batch size')
parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)', dest='weight_decay')
parser.add_argument('--num-classes', default=10, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--model-name', default='face1_vgg', type=str, help='model name')

class SyntheticDataset(Dataset):
    def __init__(self, image_dir, transform=None, preloading=True):
        """
        Args:
            image_dir (string): Path to the image data "/home/AD/rraina/pain/Frustration/generated_faces/8.6/"
        """
        self.imagepath = image_dir
        self.image_files = [(root, name) for root,dirs,files in os.walk(self.imagepath) for name in sorted(files) if name[-3:]=='png' or name[-3:]=='jpg']

    def __len__(self):
        return sum([len(self.image_files)])

    def get_item_helper(self, idx):
        """
        Return: sample
            an example of sample:
                sample['image'] = np.ndarray 3xWxH
                sample['image_dir'] = './../data/racial_pain_cropped/Stimuli/Experiment1/BlackTargets'
                sample['image_id'] = 'Subj4_PainMorph01.jpg'
                sample['label'] = 1
        """
        img_dir = self.image_files[idx][0]
        img_name = self.image_files[idx][1]
        image = cv2.imread(os.path.join(img_dir, img_name))
        image_id = img_name

        pspi_au = ['au4','au6','au7','au9','au10','au12','au20','au25','au26','au43']

        image_au = int(path.split('_')[1].split('.png')[0].split('.')[0])

        try:
            label = int(image_id[-6:-4])
        except:
            label = int(image_id[-5:-4])

        sample = {'image': image, 'image_dir': img_dir, 'image_id': image_id, "label": label}

        if self.transform:
            sample['image'] = self.transform(image)

        return sample

    def __getitem__(self, idx):
        sample = {}
        if self.preloading:
            sample = self.huge_dictionary[idx]
        else:
            sample = self.get_item_helper(idx)
        return sample


def train(model, dataloader, criterion, optimizer, num_epochs):
    model.train()

def main():
    args = parser.parse_args()

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
            transforms.Normalize([0.3139,0.3527,0.5082],[0.1173,0.1253,0.1412]),
            transforms.Normalize([0.3873985 , 0.42637664, 0.53720075], [0.2046528 , 0.19909547, 0.19015081])
        ]),
        'val': transforms.Compose([
            # BGR2RGB(),
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.3873985 , 0.42637664, 0.53720075], [0.2046528 , 0.19909547, 0.19015081])
        ]),
        'test': transforms.Compose([
            # BGR2RGB(),
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.3873985 , 0.42637664, 0.53720075], [0.2046528 , 0.19909547, 0.19015081])
        ]),
    }

    image_dir = args.data

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = VGG_16()
    model.load_weights()
    num_ftrs = model.fc8.in_features
    model.fc8 = nn.Linear(num_ftrs, args.num_classes)
    model = model.to(device)
    pretrained_model_path = '/mnt/cube/projects/xiaojing/shoulder_pain_detection_weightall/newnorm_PSPIAU/models_sf1/0.pth'
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))



if __name__ == '__main__':
    main()