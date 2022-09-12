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
import csv

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='/home/AD/rraina/pain/Frustration/generated_faces/LightSkin__centercropped_zscore_shoulder', metavar='DIR', type=str, help='path to dataset')
parser.add_argument('--batch-size', default=32, type=int, metavar='N', help='batch size')
parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)', dest='weight_decay')
parser.add_argument('--num-classes', default=10, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--model-name', default='face1_vgg', type=str, help='model name')

def main():
    args = parser.parse_args()

    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)

    data_transforms = transforms.Compose([
            # BGR2RGB(),
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.3873985 , 0.42637664, 0.53720075], [0.2046528 , 0.19909547, 0.19015081])
        ])

    print("Initializing Images...")

    image_dir = args.data

    model = VGG_16()
    model.load_weights()
    num_ftrs = model.fc8.in_features
    model.fc8 = nn.Linear(num_ftrs, args.num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    for sf in range(0, 10):
        for cv in tqdm(range(0, 5)):
            pretrained_model_path = '/mnt/cube/projects/xiaojing/shoulder_pain_detection_weightall/newnorm_PSPIAU/models_sf{}/{}.pth'.format(sf, cv)
            model.load_state_dict(torch.load(pretrained_model_path, map_location=device))

            # Iterate directory
            res = []
            for path in os.listdir(image_dir):
                res.append(path)
            res.sort()

            with open('/home/AD/rraina/pain/Frustration/results/{}_{}.csv'.format(os.path.basename(os.path.normpath(image_dir)), '_'.join(pretrained_model_path.split('/')[-2:]).split('.')[0]), 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',',escapechar=' ', quoting=csv.QUOTE_NONE)
                csvwriter.writerow(['ImagePath', 'PredPSPI', 'PredAU4','PredAU6','PredAU7','PredAU10','PredAU12',
                                    'PredAU20','PredAU25','PredAU26','PredAU43', 'ActualPSPI', 'ActualAU4', 'ActualAU6', 
                                    'ActualAU7', 'ActualAU9', 'ActualAU10', 'ActualAU12', 'ActualAU20', 'ActualAU25', 'ActualAU26', 'ActualAU43'])
                for path in res:
                    inputs = cv2.imread(os.path.join(image_dir, path))
                    inputs = data_transforms(inputs)
                    inputs = inputs.to(device).unsqueeze(0)
                    
                    pspi_au = ['au4','au6','au7','au9','au10','au12','au20','au25','au26','au43']
                    au_dict = {}
                    for au in pspi_au:
                        if len(path.split('{}.'.format(au))) > 1:
                            activation = float(path.split('{}.'.format(au))[1].split('_')[0].split('.')[0])
                            if au == 'au43' and activation > 0:
                                au_dict[au] = 1.0
                            else:
                                au_dict[au] = activation*0.5
                        else:
                            au_dict[au] = 0.0
                    pspi_actual = au_dict['au4'] + max(au_dict['au6'], au_dict['au7']) + max(au_dict['au9'], au_dict['au10']) + au_dict['au43']
                    with torch.set_grad_enabled(False):
                        outputs = model(inputs) * torch.FloatTensor([16] + [5]*8 + [1]).to(device)
                        row = [path]
                        row.extend(list(map(str, outputs[0].tolist())))
                        #au[[3,5,6,9,11,19,24,25,42]] --> au[[4,6,7,10,12,20,25,26,43]]
                        row.extend([pspi_actual, au_dict['au4'], au_dict['au6'], au_dict['au7'], au_dict['au9'], au_dict['au10'], au_dict['au12'],au_dict['au20'], au_dict['au25'], au_dict['au26'], au_dict['au43']])
                        csvwriter.writerow(row)
                

if __name__ == '__main__':
    main()