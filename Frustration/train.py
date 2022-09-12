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

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('--batch-size', default=1, type=int, metavar='N', help='batch size')
parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)', dest='weight_decay')
parser.add_argument('--num-classes', default=10, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--model-name', default='face1_vgg', type=str, help='model name')

def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, scaler):
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"

    for sample in metric_logger.log_every(data_loader, 100, header):
        inputs = sample['image']
        labels = sample['au'].float()

        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            # Get model outputs and calculate loss
            output = model(inputs)
            loss = criterion(output, labels)

        scaler.scale(loss.sum(0).mean()).backward()
        scaler.step(optimizer)
        scaler.update()

        lr_scheduler.step()

        metric_logger.update(loss=loss.sum(0).mean().item(), lr=optimizer.param_groups[0]["lr"])

def validate(model, criterion, data_loader, device):
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    correct = 0
    total = 0

    for sample in metric_logger.log_every(data_loader, 1, header):
        inputs = sample['image']
        labels = sample['au'].float()

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.cuda.amp.autocast():
            # Get model outputs and calculate loss
            output = model(inputs)
            loss = criterion(output, labels)

            total += labels.size(0)
            correct += (output == labels).sum().item()

        metric_logger.update(loss=loss.sum(0).mean().item(), acc=100 * correct / total)

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

    print("Initializing Datasets and Dataloaders...")

    image_dir = "/mnt/cube/projects/xiaojing/data/UNBCMcMaster_cropped/Images0.3"
    label_dir = "/mnt/cube/projects/xiaojing/data/UNBCMcMaster"

    subjects = []
    for d in next(os.walk(image_dir))[1]:
        subjects.append(d[:3])

    val_subj, test_subj = np.random.choice(subjects, 2)

    datasets_dict = {x: McMasterDataset(image_dir, label_dir, val_subj, test_subj, x, data_transforms[x]) for x in ['train', 'val', 'test']}

    shuffle = {'train': True, 'val': False, 'test': False}
    dataloaders_dict = {x: torch.utils.data.DataLoader(datasets_dict[x], batch_size=args.batch_size, shuffle=shuffle[x]) for x in ['train', 'val', 'test']}

    au4_list = []
    au6_list = []
    au7_list = []
    au10_list = []
    au12_list = []
    au20_list = []
    au25_list = []
    au26_list = []
    au43_list = []
    for sample in tqdm(dataloaders_dict['train']):
        #au = au[[3,5,6,9,11,19,24,25,42]] --> au = au[[4,6,7,10,12,20,25,26,43]]
        au4 = sample['au'].tolist()[0][0]
        au6 = sample['au'].tolist()[0][1]
        au7 = sample['au'].tolist()[0][2]
        au10 = sample['au'].tolist()[0][3]
        au12 = sample['au'].tolist()[0][4]
        au20 = sample['au'].tolist()[0][5]
        au25 = sample['au'].tolist()[0][6]
        au26 = sample['au'].tolist()[0][7]
        au43 = sample['au'].tolist()[0][8]
        if au4 not in au4_list:
            print('AU4: ', au4)
            au4_list.append(au4)
        if au6 not in au6_list:
            print('AU6: ', au6)
            au6_list.append(au6)
        if au7 not in au7_list:
            print('AU7: ', au7)
            au7_list.append(au7)
        if au10 not in au10_list:
            print('AU10: ', au10)
            au10_list.append(au10)
        if au12 not in au12_list:
            print('AU12: ', au12)
            au12_list.append(au12)
        if au20 not in au20_list:
            print('AU20: ', au20)
            au20_list.append(au20)
        if au25 not in au25_list:
            print('AU25: ', au25)
            au25_list.append(au25)
        if au26 not in au26_list:
            print('AU26: ', au26)
            au26_list.append(au26)
        if au43 not in au43_list:
            print('AU43: ', au43)
            au43_list.append(au43)
        

    return

    model = VGG_16()
    model.load_weights()
    num_ftrs = model.fc8.in_features
    model.fc8 = nn.Linear(num_ftrs, args.num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # Observe that all parameters are being optimized
    last_layer = list(model.children())[-1]
    ignored_params = list(map(id, last_layer.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params,
                    model.parameters())

    optimizer = optim.Adam([
                    {'params': base_params},
                    {'params': last_layer.parameters(), 'lr': args.lr}
                ], lr=args.lr, weight_decay=args.weight_decay)
    
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(dataloaders_dict['train']) * args.epochs)) ** 0.9)

    criterion = nn.MSELoss(reduction='none')

    scaler = torch.cuda.amp.GradScaler()


    for epoch in range(0, args.epochs):
        train_one_epoch(model, criterion, optimizer, dataloaders_dict['train'], lr_scheduler, device, epoch, scaler)
        validate(model, criterion, dataloaders_dict['val'], device)

class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )

class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str}")

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

if __name__ == '__main__':
    main()