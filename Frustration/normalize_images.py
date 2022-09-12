import os
import cv2
from scipy import stats
import numpy as np
from tqdm import tqdm

spath = '/home/AD/rraina/pain/Frustration/generated_faces/8.6_cropped/'

mean=130
std=130

target = '/home/AD/rraina/pain/Frustration/generated_faces/8.6_cropped_normalized/'

if not os.path.exists(target):
    os.mkdir(target)

for d in tqdm(os.listdir(spath)): # spath = './UNBCMcMaster_cropped/Images'
    if d[-3:]!='png' and d[-3:]!='jpg':
        continue
    name = os.path.join(spath,d)
    image = cv2.imread(name)
    hsv_img = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2HSV) 
    zscore_img = hsv_img[:,:,2].astype(np.float)
    idx = np.ones(zscore_img.shape)
    idx = idx.astype(int)
    zscore_img[idx] = stats.zscore(zscore_img[idx])
    matched_img_v = zscore_img
    matched_img_v[idx] = (zscore_img[idx] * std + mean)
    matched_img_v[matched_img_v>255] = 255
    matched_img_v[matched_img_v<0] = 0    
    hsv_img[:,:,2] = matched_img_v
    matched_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    cv2.imwrite(os.path.join(target,d), matched_img)