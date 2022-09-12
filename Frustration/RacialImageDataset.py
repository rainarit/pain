from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys
sys.path.insert(0, './../')
from McMasterDataset_func import *
import torch
import os
import cv2
import time
import numpy as np
import sys
import pandas as pd
from scipy import stats
import PIL

class RacialImageDataset(Dataset):
    """ McMaster Shoulder Pain Dataset. """

    def __init__(self, image_dir, transform=None, preloading=True):
        """
        Args:
            image_dir (string): Path to the image data "racial_pain_cropped/Stimuli/Experiment2"
            transform (callable, optional): Optional transfomr to be applied on a sample
            preloading (optional): True if want to load data at initialization
        """
        self.imagepath = image_dir
        self.image_files = [(root, name) for root,dirs,files in os.walk(self.imagepath) for name in sorted(files) if name[-3:]=='png' or name[-3:]=='jpg']
        self.transform = transform
        self.preloading = preloading
        self.huge_dictionary = self.preload() if preloading else None


    def __len__(self):
        return sum([len(self.image_files)])

    def preload(self):
        print('Preloading data...')
        tic = time.time()
        huge_dictionary = []

        if sys.version_info[0] >= 3:
            printProgressBar_py3(0, self.__len__(), prefix = 'Progress:', suffix = 'Complete', length = 50)
        else:
            printProgressBar_py2(0, self.__len__(), prefix = 'Progress:', suffix = 'Complete', length = 50)

        for idx in range(self.__len__()):
            sample = self.get_item_helper(idx)
            huge_dictionary.append(sample)

            if sys.version_info[0] >= 3:
                printProgressBar_py3(idx + 1, self.__len__(), prefix = 'Progress:', suffix = 'Complete', length = 50)
            else:
                printProgressBar_py2(idx + 1, self.__len__(), prefix = 'Progress:', suffix = 'Complete', length = 50)

        toc = time.time()
        print('Finished in %.2f minutes.' %float((toc - tic)/60))
        return huge_dictionary
        # for key, value in huge_dictionary.items():
        #     pickle.dump(huge_dictionary, open(key+'_'+self.subset, 'wb'))

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

class BGR2RGB(object):
    """Convert BGR image to RGB.
    """

    def __call__(self, image):
        image = image[:,:,::-1]

        return image

class prewhiten(object):
    def __call__(self, x):
        mean = x.mean()
        std = x.std()
        std_adj = std.clamp(min=1.0/(float(x.numel())**0.5))
        y = (x - mean) / std_adj
        return y

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
        return image


class zscore_matching(object):
    """ Histogram Matching to target_hist
    """

    def __init__(self, mean, std, excludeblack = False, excludewhite = False):
        self.mean = mean
        self.std = std
        self.excludeblack = excludeblack
        self.excludewhite = excludewhite

    def __call__(self, image):
        # image = match_histogram(image, self.target_hist)
        hsv_img = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2HSV) 
        zscore_img = hsv_img[:,:,2].astype(np.float)
        if self.excludewhite:
            idx = zscore_img<255
        if self.excludeblack:
            idx = zscore_img>0
        else:
            idx = np.ones(zscore_img.shape)
        idx = idx.astype(int)
        zscore_img[idx] = stats.zscore(zscore_img[idx])
        matched_img_v = zscore_img
        matched_img_v[idx] = (zscore_img[idx] * self.std + self.mean)
        matched_img_v[matched_img_v>255] = 255
        matched_img_v[matched_img_v<0] = 0    
        hsv_img[:,:,2] = matched_img_v
        matched_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        return matched_img
        

class change_background(object):
    """ change white background to specified value in hsv
    """

    def __init__(self, newbg):
        self.newbg = newbg

    def __call__(self, image):
        # image = match_histogram(image, self.target_hist)
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
        matched_img_v = hsv_img[:,:,2].astype(np.float)
        matched_img_v[matched_img_v>=255] = self.newbg
        matched_img_v[matched_img_v<0] = 0    
        hsv_img[:,:,2] = matched_img_v
        matched_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        return matched_img