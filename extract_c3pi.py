import pickle
import iz_val as val
from sliding_hog import slideExtract, location_to_df, extract_im
import numpy as np
import cv2
import os
import pandas as pd
from skimage import io
from sliding_hog import location_to_df, slideExtract

'''
This was made to automate the SVM-HOG of consumer images. 
'''
file = open('svm_prob_clf_pkl2', 'rb')
model = pickle.load(file)

df = val.read_pill_data()
df = df.loc[df['Class'] == 'C3PI_Test']

names = [name for name in df['Name']]

root = '/home/ngs/pillproject/PillProjectDisc4'

flist = []
for dirs, subdirs, files in os.walk(root):
    for file in files:
        if file.endswith('.JPG'):
            flist.append(file)

out_dir = '/home/izzy/projects/vision/pilldata/consumer_images'
checked = os.listdir(out_dir)

local_files = [x for (x,y) in zip(flist, names) if x in names and x not in checked]

root = '/home/ngs/download_images_only_non_sorted'
for file in local_files:
    im = io.imread(f'{root}/{file}')
    #im = io.imread(f'{root}/images/{file}')
    df = location_to_df(
                        slideExtract(im, model)
    )
    df.to_csv(f'{out_dir}/{file}.csv')
    im = extract_im(df, im)
    cv2.imwrite(f'{out_dir}/{file}', im)

local_files = [file for file in local_files if file not in checked]



newlist = []
for file in local_files:
    if file not in checked:
        newlist.append(file)