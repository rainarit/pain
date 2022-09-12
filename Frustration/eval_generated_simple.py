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
            #transforms.ToTensor(),
            #transforms.Normalize([0.3139,0.3527,0.5082],[0.1173,0.1253,0.1412]),
            #transforms.ToPILImage(),
            #darken(0.3),
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize([0.3139,0.3527,0.5082],[0.1173,0.1253,0.1412]),
            transforms.Normalize([0.3873985 , 0.42637664, 0.53720075], [0.2046528 , 0.19909547, 0.19015081]),
        ])


    image_dir = args.data

    model = VGG_16()
    model.load_weights()
    num_ftrs = model.fc8.in_features
    model.fc8 = nn.Linear(num_ftrs, args.num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    for sf in range(0, 10):
        for cv in range(0, 5):
            if sf == 1 and cv == 0:
                pretrained_model_path = '/mnt/cube/projects/xiaojing/shoulder_pain_detection_weightall/newnorm_PSPIAU/models_sf{}/{}.pth'.format(sf, cv)
                model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
                model.eval()

                # Iterate directory
                res = []
                for path in os.listdir(image_dir):
                    res.append(path)
                res.sort()

                with open('/home/AD/rraina/pain/Frustration/results/{}_{}_without[0.3139,0.3527,0.5082],[0.1173,0.1253,0.1412].csv'.format(os.path.basename(os.path.normpath(image_dir)), '_'.join(pretrained_model_path.split('/')[-2:]).split('.')[0]), 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter=',',escapechar=' ', quoting=csv.QUOTE_NONE)
                    csvwriter.writerow(['ImagePath', 
                                        'PredPSPI', 
                                        'PredAU4','PredAU6','PredAU7','PredAU10','PredAU12','PredAU20','PredAU25','PredAU26','PredAU43', 
                                        'ActualPSPI', 
                                        'ActualAU4','ActualAU6','ActualAU7','ActualAU9','ActualAU10','ActualAU12','ActualAU20','ActualAU25','ActualAU26','ActualAU43'])

                    for path in tqdm(res):
                        print(path)
                        inputs = cv2.imread(os.path.join(image_dir, path))
                        #inputs = (inputs - 130.18870532512665)/36.52914226055145
                        inputs = data_transforms(inputs)
                        inputs = inputs.to(device).unsqueeze(0)

                        image_au = int(path.split('_')[1].split('.png')[0].split('.')[0])
                        #image_au = int(path.split('au')[1].split('.png')[0].split('.')[0])
                        image_au_activation = float(path.split('_')[1].split('.png')[0].split('.')[1])
                        #image_au_activation = float(path.split('au')[1].split('.png')[0].split('.')[1])
                        actual_aus = ['4','6','7','9','10','12','20','25','26','43']
                        au_dict = {}
                        for au in actual_aus:
                            if int(au) == image_au:
                                if int(au) == 43:
                                    image_au_activation = image_au_activation * 0.5
                                else:
                                    image_au_activation = image_au_activation * 0.5
                                au_dict[au] = image_au_activation
                            else:
                                au_dict[au] = 0.0

                        pspi_actual = au_dict['4'] + max(au_dict['6'], au_dict['7']) + max(au_dict['9'], au_dict['10']) + au_dict['43']
                        with torch.set_grad_enabled(False):
                            outputs = model(inputs) * torch.FloatTensor([16] + [5]*9).to(device)
                            row = [path]
                            row.extend(list(map(str, outputs[0].tolist())))
                            #au[[3,5,6,9,11,19,24,25,42]] --> au[[4,6,7,10,12,20,25,26,43]]
                            row.extend([pspi_actual, au_dict['4'], au_dict['6'], au_dict['7'], au_dict['9'], au_dict['10'], au_dict['12'],au_dict['20'], au_dict['25'], au_dict['26'], au_dict['43']])
                            csvwriter.writerow(row)
                

if __name__ == '__main__':
    main()