from skimage.feature import hog
from sklearn import svm
#import skimage.io
import pandas as pd
import numpy as np
from skimage import io
from image_mask import bettercrop
from background import random_bg
from superimpose import superimpose
from resize import sane_resizing
from rotation import arbitrary_rotation
from smallest_rect import small_img_rect
from sklearn.model_selection import train_test_split
#from sklearn.externals import joblib
import os
import glob
from joblib import dump, load
import iz_ip
import iz_val as val
from joblib import parallel_backend, Parallel, delayed

#val.cr
with parallel_backend('threading', n_jobs=10):
    #Parallel(n_jobs = -2)(delayed(map)(random_bg, l))
    ## Seems to work but uses all 12 threads???
    Parallel()(delayed(clf3.fit(l1, labs2)))
    

    for img in imgs2:
        im = skimage.io.imread(f'/home/izzy/projects/vision/pilldata/converted_pngs/{img}')
        im = png_to_hog(im)
        imhog = hog(im, block_norm='L2-Hys')
        np.save(f'hog_{img}', imhog)
        
    # Your scikit-learn code here

################################################

df = val.read_pill_data()
os.chdir('/home/izzy/projects/vision/pilldata/png')
fileroot = '/home/ngs/pillproject'
root = '/home/izzy/projects/vision/pilldata/png'
names = [name for name in df['Name']]
sources = [source for source in df['Source']]
for name, source in list(zip(names, sources)):
    bg = random_bg()
    rng = np.random.default_rng()
    angle = rng.integers(1, 360)
    io.imsave(f'./blank_bg/bg{name}', bg)
    im = io.imread(f'{fileroot}/{source}/images/{name}')
    im = bettercrop(im)
    im = arbitrary_rotation(im, angle)
    im = small_img_rect(im)
    im = sane_resizing(im)
    im = superimpose(im, bg)
    io.imsave(f'./std/{name}', im)
# positive dir = '/home/izzy/projects/vision/pilldata/png/std'

images = []
labels = []
pos_dir = '/home/izzy/projects/vision/pilldata/png/std'
for image in os.listdir(pos_dir):
    if image.endswith('.PNG'):
        images.append(f'{pos_dir}/{image}')
        labels.append(1)

# negative dir = '/home/izzy/projects/vision/pilldata/png/blank_bg'
neg_dir = '/home/izzy/projects/vision/pilldata/png/blank_bg'
for image in os.listdir(neg_dir):
    images.append(f'{neg_dir}/{image}')
    labels.append(0)

from sklearn.model_selection import train_test_split

#xtrain, xtest, ytrain, ytest = train_test_split(images, 
#                                                labels, 
#                                                train_size=1, 
#                                                shuffle = True)

   
from skimage import color

for image in images:
    im = io.imread(image)
    
    if(im.shape[::-1][0] == 4):
        im = im[:, :, :3]
        
    im = color.rgb2gray(im)
    
    imhog = hog(im, 
                block_norm='L2-Hys',
                transform_sqrt=True
                )
    
    np.save(f'{image}_hog', imhog)
    
files = []
labels = []

for f in glob.glob(
                    os.path.join(pos_dir, "*.npy")
                    ):
    files.append(np.load(f))
    labels.append(1)
    
for f in glob.glob(
                    os.path.join(neg_dir, '*.npy')
                    ):
    files.append(np.load(f))
    labels.append(0)
    
xtrain, xtest, ytrain, ytest = train_test_split(files, 
                                                labels, 
                                                train_size=0.9, 
                                                shuffle = True)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

fd = []
fdlist = []
train_labels = []
for x in xtrain:
    fd = np.load(x)
    fdlist.append(fd)
    train_labels.append()

clf = make_pipeline(StandardScaler(),
                    svm.LinearSVC()
                    )

def compareclf(test, train):
    
    clf.probability = True
    clf.fit(xtrain, ytrain)
    diff = clf.predict(xtest) - ytest
    np.count_nonzero(diff)

    clf2 = make_pipeline(
                        StandardScaler(),
                        svm.SVC(probability = True
                                )
                     )

    clf2.fit(xtrain, ytrain)
    clf2_prob = clf2.predict_proba(xtest)
    diff2 = clf2.predict(xtest) - ytest
    idx2 = np.where(diff2 != 0)


clf3 = make_pipeline(StandardScaler(),
                    svm.NuSVC(probability = True)
                     )

clf3.fit(xtrain, ytrain)
clf3_prob = clf3.predict_proba(xtest)
diff3 = clf3.predict(xtest) - ytest
idx3 = np.where(diff3 != 0)


from sklearn.linear_model import SGDClassifier
clf4 = make_pipeline(StandardScaler(),
                    SGDClassifier(max_iter=1000, 
                                  tol=1e-3,
                                  loss = 'log_loss'
                                )
                    )
clf4.fit(xtrain, ytrain)
clf4_prob = clf4.predict_proba(xtest)
diff4 = clf4.predict(xtest) - ytest
idx4 = np.where(diff4 != 0)
### log loss result has only 1/0?
# try ‘modified_huber’
clf4.fit(xtrain, ytrain)

clf4b = make_pipeline(StandardScaler(),
                    SGDClassifier(max_iter=1000, 
                                  tol=1e-3,
                                  loss = 'hinge',
                                  eta0=0.01,
                                  learning_rate = 'adaptive'
                                  )
                    )
clf4b.fit(xtrain, ytrain)
clf4b_prob = clf4b.predict_proba(xtest)
diff4b = clf4b.predict(xtest) - ytest
idx4b = np.where(diff4b != 0)





fdlist = []
trainlabels = []

for file in files:
    labels.append(1)

for file in glob.glob(
                        os.path.join(neg_dir, '*.npy')
                    ):
    files.append(file)

for file in files:
    fd = np.load(file)
    fdlist.append(fd)
    labels.append(0) 

    
    
def plusone(x):
    return x + 1

os.chdir('/home/izzy/projects/vision/DGMD14/project/hog_pos')
imgs = os.listdir('/home/izzy/projects/vision/pilldata/converted_pngs')

imgs2 = imgs[:10000]

ctl_im = skimage.io.imread(f'/home/izzy/projects/vision/pilldata/converted_pngs/{imgs[101]}')

Parallel(n_jobs = 6)(delayed(m))



    
    

def rmap_to_hog(n):
    test = png_to_hog(random_bg())
    fd = hog(test, lock_norm='L2-Hys')
    np.save(f'bg_{n}')
    
l = np.arange(0, 10000)
l = l.tolist()
    
for n in xtrain:
    if(im.shape[::-1][0] == 4):
        im = im[:, :, :3]
    im = skimage.color.rgb2gray(im)
    imhog = hog(im, transform_sqrt=True)
    np.save(f'bg_{n}', imhog)

def png_to_hog(im):
    if(im.shape[::-1][0] == 4):
        im = im[:, :, :3]
    im = skimage.color.rgb2gray(im)
    return(im)
    

hog_pos = '/home/izzy/projects/vision/DGMD14/project/hog_pos'
hog_neg = '/home/izzy/projects/vision/DGMD14/project/hog_neg'
fdlist = []
labels = []

files = glob.glob(
                os.path.join(hog_pos, "*.npy")
                )

for file in files:
    fd = np.load(file)
    fdlist.append(fd)
    labels.append(1)

files = glob.glob(os.path.join(hog_neg, "*.npy"))
for file in files:
    fd = np.load(file)
    fdlist.append(fd)
    labels.append(0)
   
l1 = fdlist[9000:11000:1]
labs2 = labels[9000:11000:1]
clf2.fit(l1, labs2)

accuracy = []
xy = fdlist[1:500:1]
for x in xy:
    x = x.reshape(1, -1)
    cls = clf3.predict(x)
    probA = clf3.probA_
    probB = clf3.probB_
    accuracy.append([cls, probA, probB])

pos_cnt = np.count_nonzero(np.array(accuracy))


accuracy = []
xy = fdlist[11000:11500:1]
for x in xy:
    x = x.reshape(1, -1)
    accuracy.append(clf3.predict(x))



    
test_im = random_bg()
test_im = png_to_hog(im)
test_im_hog = hog(test_im)
test_im_hog = test_im_hog.reshape(1, -1)
clf.predict(test_im_hog)



ctl_im = skimage.io.imread(f'/home/izzy/projects/vision/pilldata/converted_pngs/{imgs[101]}')
ctl_im = png_to_hog(ctl_im)
ctl_im_hog = hog(ctl_im)
ctl_im_hog = ctl_im_hog.reshape(1, -1)
clf.predict(ctl_im_hog)
