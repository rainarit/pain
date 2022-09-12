import os
import math
import scipy.io
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import argparse

import cv2

''' references
https://automaticaddison.com/how-to-do-histogram-matching-using-opencv/
https://towardsdatascience.com/histogram-matching-ee3a67b4cbc1
https://www.imageeprocessing.com/search/label/Histogram%20equalization#:~:text=Color%20histogram%20equalization%20can%20be,will%20not%20enhance%20the%20image.
'''

def rgb2hsi (rgb_img):
  """
  This is a function to convert rgb color image to hsi image
  :param rgm_img:rgb color image
  :return:hsi image
  """
  #Save the number of rows and columns of the original image
  row=np.shape (rgb_img) [0]
  col=np.shape (rgb_img) [1]
  #Copy the original image
  hsi_img=rgb_img.copy ()
  #Channel split the image
  b, g, r=cv2.split (rgb_img)
  #Normalize the channel to [0,1]
  [b, g, r]=[i/255.0 for i in ([b, g, r])]
  h=np.zeros ((row, col)) #define h channel
  i=(r + g + b)/3.0 #Calculate the i channel
  s=np.zeros ((row, col)) #define s channel
  for i in range (row):
    den=np.sqrt ((r [i] -g [i]) ** 2+ (r [i] -b [i]) * (g [i] -b [i]))
    thetha=np.arccos (0.5 * (r [i] -b [i] + r [i] -g [i])/den) #Calculate the angle
    h=np.zeros (col) #define temporary array
    #den>0 and g>= b element h is assigned thetha
    h [b [i]<= g [i]]=thetha [b [i]<= g [i]]
    #den>0 and the element h of g<= b is assigned thetha
    h [g [i]<b [i]]=2 * np.pi-thetha [g [i]<b [i]]
    #den<0's element h is assigned a value of 0
    h [den == 0]=0
    h [i]=h/(2 * np.pi) #Assign to h channel after radian
  #Calculate s channel
  for i in range (row):
    min=[]
    #Find the minimum value of each group of rgb values
    for j in range (col):
      arr=[b [i] [j], g [i] [j], r [i] [j]]
      min.append (np.min (arr))
    min=np.array (min)
    #Calculate s channel
    s [i]=1-min * 3/(r [i] + b [i] + g [i])
    #i is a value of 0 directly assigned 0
    s [i] [r [i] + b [i] + g [i] == 0]=0
  #Expand to 255 for easy display,Generally h component is between [0,2pi] and s and i are between [0,1]
  hsi_img [:,:, 0]=h * 255
  hsi_img [:,:, 1]=s * 255
  hsi_img [:,:, 2]=i * 255
  return hsi_img

def hsi2rgb (hsi_img):
  """
  This is a function to convert hsi image to rgb image
  :param hsi_img:hsi color image
  :return:rgb image
  """
  #Save the number of rows and columns of the original image
  row=np.shape (hsi_img) [0]
  col=np.shape (hsi_img) [1]
  #Copy the original image
  rgb_img=hsi_img.copy ()
  #Channel split the image
  h, s, i=cv2.split (hsi_img)
  #Normalize the channel to [0,1]
  [h, s, i]=[i/255.0 for i in ([h, s, i])]
  r, g, b=h, s, i
  for i in range (row):
    h=h [i] * 2 * np.pi
    #h is greater than or equal to 0 and less than 120 degrees
    a1=h>= 0
    a2=h<2 * np.pi/3
    a=a1 & a2 #fancy indexing in the first case
    tmp=np.cos (np.pi/3-h)
    b=i [i] * (1-s [i])
    r=i [i] * (1 + s [i] * np.cos (h)/tmp)
    g=3 * i [i] -r-b
    b [i] [a]=b [a]
    r [i] [a]=r [a]
    g [i] [a]=g [a]
    #h is greater than or equal to 120 degrees and less than 240 degrees
    a1=h>= 2 * np.pi/3
    a2=h<4 * np.pi/3
    a=a1 & a2 #fancy index for the second case
    tmp=np.cos (np.pi-h)
    r=i [i] * (1-s [i])
    g=i [i] * (1 + s [i] * np.cos (h-2 * np.pi/3)/tmp)
    b=3 * i [i]-r-g
    r [i] [a]=r [a]
    g [i] [a]=g [a]
    b [i] [a]=b [a]
    #h is greater than or equal to 240 degrees and less than 360 degrees
    a1=h>= 4 * np.pi/3
    a2=h<2 * np.pi
    a=a1 & a2 #fancy index in the third case
    tmp=np.cos (5 * np.pi/3-h)
    g=i [i] * (1-s [i])
    b=i [i] * (1 + s [i] * np.cos (h-4 * np.pi/3)/tmp)
    r=3 * i [i]-g-b
    b [i] [a]=b [a]
    g [i] [a]=g [a]
    r [i] [a]=r [a]
  rgb_img [:,:, 0]=b * 255
  rgb_img [:,:, 1]=g * 255
  rgb_img [:,:, 2]=r * 255
  return rgb_img

def generate_histogram(img, space='hsv', do_print=False):
    """
    @params: img: can be a grayscale or color image. We calculate the Normalized histogram of this image.
    @params: do_print: if or not print the result histogram
    @return: will return both histogram and the grayscale image 
    """
    # img is colorful, so we convert it to hsv then take value channel
    if len(img.shape)==3:
        if space == 'hsv':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif space == 'hsi':
            img = rgb2hsi(img)
        gr_img = img[:,:,2]
    else:
        gr_img = img

    gr_img = gr_img.ravel()
    '''now we calc grayscale histogram'''
    gr_hist = np.zeros([256])
    for pixel in range(gr_img.shape[0]):
        pixel_value = int(gr_img[pixel])
        gr_hist[pixel_value] += 1
    '''normalizing the Histogram'''
    gr_hist = gr_hist / float(gr_img.shape[0])
    if do_print:
        print_histogram(gr_hist, name="n_h_img", title="Normalized Histogram")
    return gr_hist, img

def print_histogram(_histrogram, name, title):
    plt.figure()
    plt.title(title)
    plt.plot(_histrogram, color='#ef476f')
    plt.bar(np.arange(len(_histrogram)), _histrogram, color='#b7b7a4')
    plt.ylabel('Number of Pixels')
    plt.xlabel('Pixel Value')
    plt.savefig("hist_" + name)

def equalize_histogram(histo, img=None,  _print=False):

    eq_histo = np.cumsum(histo)
    
    if img is not None:
        '''enhance image as well:'''
        en_img = np.zeros_like(img)
        for x_pixel in range(img.shape[0]):
            for y_pixel in range(img.shape[1]):
                pixel_val = int(img[x_pixel, y_pixel])
                en_img[x_pixel, y_pixel] = eq_histo[pixel_val]*255
        return eq_histo, en_img
    if _print:
        print_histogram(eq_histo, name="eq_"+str(index), title="Equalized Histogram")
    return eq_histo

def calculate_lookup(src_cdf, ref_cdf):
    """
    This method creates the lookup table
    :param array src_cdf: The cdf for the source image
    :param array ref_cdf: The cdf for the reference image
    :return: lookup_table: The lookup table
    :rtype: array
    """
    lookup_table = np.zeros(256)
    lookup_val = 0
    for src_pixel_val in range(len(src_cdf)):
        lookup_val
        for ref_pixel_val in range(len(ref_cdf)):
            if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                lookup_val = ref_pixel_val
                break
        lookup_table[src_pixel_val] = lookup_val
    return lookup_table

def match_histogram(inp_img, hist_input, e_hist_input, e_hist_target, _print=False):
    '''map from e_hist_input to 'target_hist '''
    en_img = np.zeros_like(inp_img)
    tran_hist = np.zeros_like(e_hist_input)
    for i in range(len(e_hist_input)):
        tran_hist[i] = find_value_target(val=e_hist_input[i], target_arr=e_hist_target)
    if _print:
        print_histogram(tran_hist, name="trans_hist_", title="Transferred Histogram")
    '''enhance image as well:'''
    for x_pixel in range(inp_img.shape[0]):
        for y_pixel in range(inp_img.shape[1]):
            pixel_val = int(inp_img[x_pixel, y_pixel])
            en_img[x_pixel, y_pixel] = tran_hist[pixel_val]
    return en_img
    
def calculate_mean_std(histo):
    n=len(histo)
    _sum=0
    prod=0
    sqsum=0
    for x,y in enumerate(histo):
        _sum+=y
        prod+=x*y
    mean=prod/_sum
    for x,y in enumerate(histo):
        dx=x-mean
        sqsum+=y*dx*dx
    # σ²
    variance=sqsum/_sum
    stdv=math.sqrt(variance)

    return mean, stdv

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='/home/AD/rraina/pain/Frustration/generated_faces/LightSkin/', metavar='DIR', type=str, help='path to dataset')
parser.add_argument('--mode', default='custom', type=str, help='mode of changing contrast')
parser.add_argument('--method', default='histmatch', type=str, help='method of changing contrast')
parser.add_argument('--space', default='hsv', type=str, help='hsi or hsv space')
parser.add_argument('--diffexp', default=True, type=bool, help='whether matching for each experiment or not')
parser.add_argument('--excludewhite', default=True, type=bool, help='whether excluding white or not')
parser.add_argument('--backgroundcolor', default=0, type=int, help='background color')
parser.add_argument('--mean', default=80, type=int, help='custom mean for custom zscore')
parser.add_argument('--std', default=80, type=int, help='custom mean for custom zscore')

def main():
    args = parser.parse_args()

    spath = args.data
    target = None

    mode = args.mode #'w2b': white matched to average black, 'b2w': black matched to average white, '2shoulder': black/white to average shoulder, 'custom': custom mean/std
    method = args.method  #'histmatch': histogram matching; 'zscore': match mean and std
    space = args.space # hsi or hsv
    diffexp = args.diffexp #whether matching for each experiment or not
    if mode == '2shoulder':
        diffexp = False
    excludewhite = args.excludewhite
    backgroundcolor = args.backgroundcolor # default 255
    # set mean and std for custom zscore
    mean=args.mean
    std=args.std

    if mode == '2shoulder':
        target_path = f'{spath[:-1]}_{method}'
        target_hist_path = f'/mnt/cube/projects/xiaojing/{space}value_shoulderpain_histogram.npz'
    elif mode == 'custom':
        target_path = f'{spath[:-1]}_{method}_mean{mean}std{std}_background{backgroundcolor}'

    if not diffexp and not (mode=='custom'):
        target_hist_xy = np.load(target_hist_path)
        target_hist = target_hist_xy['hist']
        if excludewhite:
            whiteth = np.where(np.diff(target_hist)<0)[-1][-1]
            target_hist = target_hist[:whiteth]
            target_hist = target_hist / float(target_hist.shape[0])
        e_target_hist = equalize_histogram(target_hist)
        mean_all, std_all = calculate_mean_std(target_hist)
        mean = target_hist_xy['mean']*255
        std = target_hist_xy['std']*255
        print(mean, std)

    for (root,dirs,files) in os.walk(spath, topdown=True): 

        subdir = root[len(spath):]

        if diffexp and not (mode=='custom'):
            target_hist_xy = np.load(target_hist_path)
            target_hist = target_hist_xy['hist']
            if excludewhite:
                whiteth = np.where(np.diff(target_hist)<0)[-1][-1]
                target_hist = target_hist[:whiteth]
                target_hist = target_hist / float(target_hist.shape[0])
            e_target_hist = equalize_histogram(target_hist)
            mean_all, std_all = calculate_mean_std(target_hist)
            median_all = np.argmax(target_hist[:-1])
            mean = target_hist_xy['mean']
            std = target_hist_xy['std']
            print(mean_all, median_all)

        for f in os.listdir(root):

            if (f[-3:] != 'png') and (f[-3:] != 'jpg'):
                continue
            
            subdir = root[len(spath):]
            os.makedirs(os.path.join(target_path, subdir), exist_ok=True)

            image = cv2.imread(os.path.join(root, f))

            if method == 'histmatch':        
                histo, hsv_img = generate_histogram(image, space)
                if excludewhite:
                    whiteth = np.where(np.diff(histo)<0)[-1][-1]
                    histo = histo[:whiteth]
                    histo = histo / float(histo.shape[0])
                e_histo= equalize_histogram(histo)

                lookup = calculate_lookup(e_histo, e_target_hist)
                if excludewhite:
                    lookup[whiteth:] = 255

                matched_img_v = cv2.LUT(hsv_img[:,:,2], lookup)
            
            elif method == 'zscore':
                hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
                zscore_img = hsv_img[:,:,2].astype(np.float)
                if excludewhite:
                    idx = zscore_img<255
                else:
                    idx = np.ones(zscore_img.shape)
                zscore_img[idx] = stats.zscore(zscore_img[idx])
                matched_img_v = zscore_img
                matched_img_v[idx] = (zscore_img[idx] * std + mean)
                matched_img_v[matched_img_v>=255] = backgroundcolor
                matched_img_v[matched_img_v<0] = 0
            
            hsv_img[:,:,2] = matched_img_v
            if space == 'hsv':
                matched_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
            elif space == 'hsi':
                matched_img = hsi2rgb(hsv_img)
                
            cv2.imwrite(os.path.join(target_path, subdir, f), matched_img)

if __name__ == '__main__':
    main()