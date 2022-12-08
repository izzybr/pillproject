from sklearn.svm import LinearSVC
from skimage.feature import hog
from skimage import color
import pandas as pd
import cv2


import time
def slideExtract(image, model, windowSize=(224,224),channel="RGB",step=61):
    t0 = time.time()
    # slightly adapted from https://www.kaggle.com/code/mehmetlaudatekman/support-vector-machine-object-detection
    # Converting to grayscale
    if channel == "RGB":
        img = color.rgb2gray(image)
    elif channel == "BGR":
        img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    elif channel.lower()!="grayscale" or channel.lower()!="gray":
        raise Exception("Invalid channel type")
    
    # We'll store coords and features in these lists
    coords = []
    #features = []
    
    hIm,wIm = image.shape[:2] 
    
    # W1 will start from 0 to end of image - window size
    # W2 will start from window size to end of image
    # We'll use step (stride) like convolution kernels.
    for w1,w2 in zip(range(0,wIm-windowSize[0],step),range(windowSize[0],wIm,step)):
        for h1,h2 in zip(range(0,hIm-windowSize[1],step),range(windowSize[1],hIm,step)):
            window = img[h1:h2,w1:w2]
            hg = hog(window,
                     block_norm='L2-Hys',
                     transform_sqrt=True
                     ).reshape(1, -1)
            
            cls = model.predict(hg)
            
            probA = model.predict_proba(hg)
            
            coords.append((w1,w2,h1,h2, cls, probA[0][0], probA[0][1]))
            
    t = time.time() - t0
    print(f'Time = {t}')
    return(coords)

def showcls(df):
    '''
    A convenience function for viewing the classification results in a df.
    '''
    df = df.loc[df['cls'] == 1]
    return df


def extract_im(df, im, window: int = 0):
    '''
    Extract the window indentified by slideExtract, with, optionally, an additional window on each side of the highest probability window returned by slideExtract
    '''
    w1, w2, h1, h2 = df.loc[df['probB'].idxmax()][['w1', 'w2', 'h1', 'h2']]
    wlow = max(w1 - window, 0)
    whigh = min(w2 + window, im.shape[1])
    hlow = max(h1 - window, 0)
    hhigh = min(h2 + window, im.shape[0])
    print(f'Input image shape = {im.shape}')
    print(f'Extract returning image from: [{hlow}:{hhigh},{wlow}:{whigh}]')
    return im[hlow:hhigh, wlow:whigh]

def location_to_df(wlist):
    w1 = []
    w2 = []
    h1 = []
    h2 = []
    cls = []
    probA = []
    probB = []
    for l in wlist:
        w1.append(l[0])
        w2.append(l[1])
        h1.append(l[2])
        h2.append(l[3])
        cls.append(l[4])
        probA.append(l[5])
        probB.append(l[6])
    df = pd.DataFrame(
                list(
                    zip(
                        w1, 
                        w2, 
                        h1, 
                        h2, 
                        cls, 
                        probA, 
                        probB
                    )
                ), 
                columns = [
                           'w1',
                           'w2', 
                           'h1', 
                           'h2', 
                           'cls', 
                           'probA', 
                           'probB'
                        ]
                )
    return df
    
    
