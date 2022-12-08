import cv2
import pandas as pd
import numpy as np
import requests

path = '/home/ngs/pillproject/pp_df.csv'
df = pd.read_csv(path)

# Strings are an object in a df and pain to work with
df['Name'] = pd.Series(df['Name'], dtype = 'string')
df['Source'] = pd.Series(df['Source'], dtype ='string')
df = df.loc[ df['Layout'] == 'MC_C3PI_REFERENCE_SEG_V1.6']

imglist = [f for f in df['Name']]

def testdir(path):
    if(os.access(path, os.W_OK)):
        return True
    else:
        os.mkdir(path)
    if not(os.access(path, os.W_OK)):
        print(f"could not create {path}")
        return None

def small_img_rect(image: np.ndarray):
    
    print(f'entering small_img_rect')
    #this assumes it is receving a four channel image
    b, g, r, a = cv2.split(image)
    nzp = cv2.findNonZero(a)
    #add one pixel on 
    hmin = np.min(nzp[:, :, 1])
    hmax = np.max(nzp[:, :, 1])
    wmin = np.min(nzp[:, :, 0])
    wmax = np.max(nzp[:, :, 0])
    im = image[hmin:hmax, wmin:wmax, :]
    return im


def cropimg(imnm):
    '''This function is specific to the 'layout = MC_C3PI_REFERENCE_SEG_V1.6' images from TODO: add link
    These images are filtered for images which are only of the type mentioned above, and are all .PNG
    Given an image name, and with a df loaded in which has location information for
    the directories each image is in:
        - Attempt to write a cropped version of the image to a directory, which is created
        if necessary
    '''
    fd = df.loc[ df['Name'] == imnm]
    idir = fd['Source'].to_list()
    idir = idir[0]
    root = '/home/ngs/pillproject'
    ipath = f'{root}/{idir}/images/{imnm}'
    odir = f'{root}/{idir}/images/cropped/'
    opath = f'{root}/{idir}/images/cropped/{imnm}'
    skipped = []
    outcome = []
    l = testdir(odir)
    '''
    if(l) and os.access(ipath, os.F_OK):
        im = cv2.imread(ipath, cv2.IMREAD_UNCHANGED)
        #first crop the NIH/NLM banner at the bottom
        im = im[0:1600, :, :]
        out = small_img_rect(im)
        cv2.imwrite(opath, out)
    else:
        skipped.append(imnm)
        print(f"skipping {imnm}")

    '''
    if(l == None):
        skipped.append(imnm)
    else:
        initcwd = os.getcwd()
        if(os.access(ipath, os.F_OK)):
            im = cv2.imread(ipath, cv2.IMREAD_UNCHANGED)
            #first crop the NIH/NLM banner at the bottom
            im = im[0:1600, :, :]
            out = small_img_rect(im)
            cv2.imwrite(opath, out)
    return(outcome, l)

res = list(
            map(cropimg, imglist)
           )

# ~109s for 399 images in PillProjectDisc1 on 2020 MBP w/ Intel Quad Core i5 @ 2GHz
