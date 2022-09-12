import os
import math
import scipy.io
import random

from mtcnn.mtcnn import MTCNN
import cv2

detector = MTCNN()

extensionfactor = 0.1

spath = '/home/AD/rraina/pain/Frustration/generated_faces/AU25_experiments/morphe/'
target = '/home/AD/rraina/pain/Frustration/generated_faces/AU25_experiments/morphe_cropped/'

prev_bounding_box = [136, 175, 283, 404]

for (root,dirs,files) in os.walk(spath, topdown=True): 
    random.shuffle(files)
    for f in files:
        if (f[-3:] != 'png') and (f[-3:] != 'jpg'):
            continue
        
        subdir = root[len(spath):]
        os.makedirs(os.path.join(target, subdir), exist_ok=True)

        image = cv2.imread(os.path.join(root, f))

        result = detector.detect_faces(image)
        if not result:
            bounding_box = prev_bounding_box

            crop_img = image[max(0,bounding_box[1]- int(math.ceil(bounding_box[3]*extensionfactor/2))):min(image.shape[0],bounding_box[1]+bounding_box[3]+int(math.ceil(bounding_box[3]*extensionfactor/2))),
                    max(0,bounding_box[0]-int(math.ceil((bounding_box[2]*extensionfactor/2)))):min(image.shape[1],bounding_box[0]+bounding_box[2]+int(math.ceil((bounding_box[2]*extensionfactor/2)))), ]
                
            cv2.imwrite(os.path.join(target, subdir, f), crop_img)   
            continue 

        # Result is an array with all the bounding boxes detected. 
        bounding_box = result[0]['box']

        #bounding_box = prev_bounding_box

        crop_img = image[max(0,bounding_box[1]- int(math.ceil(bounding_box[3]*extensionfactor/2))):min(image.shape[0],bounding_box[1]+bounding_box[3]+int(math.ceil(bounding_box[3]*extensionfactor/2))),
                    max(0,bounding_box[0]-int(math.ceil((bounding_box[2]*extensionfactor/2)))):min(image.shape[1],bounding_box[0]+bounding_box[2]+int(math.ceil((bounding_box[2]*extensionfactor/2)))), ]

        # if crop_img.shape[1] < 300:
        #     bounding_box = prev_bounding_box
        #     crop_img = image[max(0,bounding_box[1]- int(math.ceil(bounding_box[3]*extensionfactor/2))):min(image.shape[0],bounding_box[1]+bounding_box[3]+int(math.ceil(bounding_box[3]*extensionfactor/2))),
        #             max(0,bounding_box[0]-int(math.ceil((bounding_box[2]*extensionfactor/2)))):min(image.shape[1],bounding_box[0]+bounding_box[2]+int(math.ceil((bounding_box[2]*extensionfactor/2)))), ]
        # else:
        #     prev_bounding_box = bounding_box

        cv2.imwrite(os.path.join(target, subdir, f), crop_img)