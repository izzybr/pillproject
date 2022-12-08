from iz_val import testdir, read_pill_data
import os
import numpy as np
from make_image import make_image2
import pandas as pd
from image_mask import bettercrop
from skimage import io
from multiprocessing import Pool
from random_index import random_index
'''
This was written to create a set of augmented images fro 
'''
#from joblib import Parallel, delayed

#rng = np.random.default_rng()
#LAYOUT = 'MC_C3PI_REFERENCE_SEG_V1.6'
#N_OF_FEAT = 200
#df = read_pill_data()
#df = df.loc[df['Layout'] == 'MC_C3PI_REFERENCE_SEG_V1.6']

#names = [name for name in df['Name']]

#vals = v for v in os.listdir('/home/ngs/color_sort_color') #if not v in ['RED', 'TURQUOISE']]
#vals = os.listdir('/home/ngs/color_sort_color')
#feat = 'Color'

#flist, dlist = [],[]

#for root, subdirs, files in os.walk('/home/ngs/pillproject'):
#    for file in files:
#        if file.endswith('.PNG') and file in names:
#            files.append(file)
#            dlist.append(os.path.join(root, file))

    #print('RIP RBG')
    
'''def find_images(value: str, df: pd.DataFrame, n_of_feat: int = 200, local_files: bool = True):
    '''
 #   value: 
    '''
    if(local_files):
        lf = 
'''

def augment(value: str, df: pd.DataFrame = df, n_of_feat: int = 200, feat: str = 'Color'):
    '''
    feat = Feature of the dataset to be used for training
    value = String of the feature
    df: DataFrame conta
    '''
    fd = df.copy(deep=True)
    write_to = './output/'
    testdir(f'{write_to}/{feat}/{value}')
    fd = df.loc[df[feat] == value]
    names = [name for name in fd['Name']]
    flist, dlist = [],[]
    
    for root, subdirs, files in os.walk('/home/ngs/pillproject'):
        for file in files:
            # selecting by .PNG is necessary because some .wmv files are present in the dataset
            if file.endswith('.PNG') and file in names:
                flist.append(file)
                dlist.append(os.path.join(root, file))
    #print(f'{len(dlist)}\n{len(flist)}\n')
    # This needs a handler if there aren't enough 
    idx = random_index(0, len(flist), n_of_feat)
    print(f'{idx}\n{len(flist)}\n{len(dlist)}')
    #im_paths, im_names = [], []
    for x in idx:
    #    im_paths.append(dlist[x])
    #    im_names.append(flist[x])
    
    #for im_path, im_name in im_paths, im_names:
        im = io.imread(dlist[x])
        im = bettercrop(im)
        for i in range(20):
            imm = make_image2(im)
            output = f'{write_to}/{feat}/{value}/{i}-{flist[x]}'
            io.imsave(output, imm)
            print(f'Wrote {output}\n')
    return True
      

#with Pool() as P:
#    x = P.map(augment, vals)
#with parallel_backend('threading', n_jobs=10):
#Parallel(n_jobs=-1)(
#    delayed(map)(vals)
#    )
    #Parallel(n_jobs = -2)(delayed(map)(random_bg, l))
    ## Seems to work but uses all 12 threads???
        
        #name_this_later(v, fd, 200, 'Color')
    
        
        
    
def GaussianMatrix(X,sigma):
    row,col=X.shape
    GassMatrix=np.zeros(shape=(row,row))
    X=np.asarray(X)
    i=0
    for v_i in X:
        j=0
        for v_j in X:
            GassMatrix[i,j]=Gaussian(v_i.T,v_j.T,sigma)
            j+=1
        i+=1
    return GassMatrix

def Gaussian(x,z,sigma):
    return np.exp((-(np.linalg.norm(x-z)**2))/(2*sigma**2))