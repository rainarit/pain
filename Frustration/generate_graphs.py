import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import numpy as np
import seaborn as sns
from os import listdir
from os.path import isfile, join
import ast
import pandas as pd
from scipy.spatial import distance
import os
import PIL

for filename in tqdm(os.listdir('/home/AD/rraina/pain/Frustration/results/')):
    f = os.path.join('/home/AD/rraina/pain/Frustration/results/', filename)
    # checking if it is a file
    if os.path.isfile(f):
      csv_file = f
      png_file = '/home/AD/rraina/pain/Frustration/graph_results/{}.png'.format(filename[:-4])

      df = pd.read_csv(csv_file)
      euro_df = df[df['ImagePath'].str.contains('e_')]
      african_df = df[df['ImagePath'].str.contains('a_')]
      euro_black_df = df[df['ImagePath'].str.contains('eb_')]
      african_white_df = df[df['ImagePath'].str.contains('aw_')]

      euro_actuals = []
      euro_preds = []

      african_actuals = []
      african_preds = []

      euro_black_actuals = []
      euro_black_preds = []

      african_white_actuals = []
      african_white_preds = []

      aus = [4,6,7,10,12,20,25,26,43]

      for au in aus:
        au_euro_df = euro_df[euro_df['ImagePath'].str.contains('{}e_{}'.format(str(au),str(au)))]
        au_euro_df = au_euro_df.sort_values('ActualAU{}'.format(au))
        euro_actuals.append(list((au_euro_df['ActualAU{}'.format(au)]/5.0)*100))
        euro_preds.append(list((au_euro_df['PredAU{}'.format(au)]/5.0)*100))

        au_african_df = african_df[african_df['ImagePath'].str.contains('{}a_{}'.format(str(au),str(au)))]
        au_african_df = au_african_df.sort_values('ActualAU{}'.format(au))
        african_actuals.append(list(au_african_df['ActualAU{}'.format(au)]/5.0*100))
        african_preds.append(list(au_african_df['PredAU{}'.format(au)]/5.0*100))

        au_euro_black_df = euro_black_df[euro_black_df['ImagePath'].str.contains('{}eb_{}'.format(str(au),str(au)))]
        au_euro_black_df = au_euro_black_df.sort_values('ActualAU{}'.format(au))
        euro_black_actuals.append(list((au_euro_black_df['ActualAU{}'.format(au)]/5.0)*100))
        euro_black_preds.append(list((au_euro_black_df['PredAU{}'.format(au)]/5.0)*100))

        au_african_white_df = african_white_df[african_white_df['ImagePath'].str.contains('{}aw_{}'.format(str(au),str(au)))]
        au_african_white_df = au_african_white_df.sort_values('ActualAU{}'.format(au))
        african_white_actuals.append(list(au_african_white_df['ActualAU{}'.format(au)]/5.0*100))
        african_white_preds.append(list(au_african_white_df['PredAU{}'.format(au)]/5.0*100))

      sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
      sns.set_style('ticks')
      fig, axs = plt.subplots(3, 3, figsize=(30, 20))
      count = 0
      for row in range(0,3):
        for col in range(0,3):
          sns.regplot(ax=axs[row, col], x=euro_actuals[count], y=euro_preds[count], scatter_kws={"color":"red"}, line_kws={"color": "red"}, ci=None, label="European")
          sns.regplot(ax=axs[row, col], x=african_actuals[count], y=african_preds[count], scatter_kws={"color":"green"}, line_kws={"color": "green"}, ci=None, label="African")
          sns.regplot(ax=axs[row, col], x=african_white_actuals[count], y=african_white_preds[count], scatter_kws={"color":"blue"}, line_kws={"color": "blue"}, ci=None, label="African White")
          sns.regplot(ax=axs[row, col], x=euro_black_actuals[count], y=euro_black_preds[count], scatter_kws={"color":"orange"}, line_kws={"color": "orange"}, ci=None, label="European Black")
          axs[row, col].set_title('AU{}'.format(aus[count]))
          axs[row, col].set_xlabel('Actual AU{} Activation (%)'.format(aus[count]))
          axs[row, col].set_ylabel('Model AU{} Activation (%)'.format(aus[count]))
          axs[row, col].legend(loc='upper left')
          count+=1
      fig.suptitle(str(filename[:-4]), y=1.02, fontsize=16)
      fig.tight_layout()
      fig.savefig(png_file)