import os
import numpy as np
import cv2
from typing import Tuple

def testdir(path):
    if(os.access(path, os.W_OK)):
        return True
    else:
        os.mkdir(path)
    if not(os.access(path, os.W_OK)):
        print(f"could not create {path}")
        return None

def small_img_rect(image: np.ndarray):
    #this assumes it is receving a four channel image
    b, g, r, a = cv2.split(image)
    nzp = cv2.findNonZero(a)
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


def bytescaling(data, cmin=None, cmax=None, high=255, low=0):
    """
    Converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255). If the input image already has 
    dtype uint8, no scaling is done.
    :param data: 16-bit image data array
    :param cmin: bias scaling of small values (def: data.min())
    :param cmax: bias scaling of large values (def: data.max())
    :param high: scale max value to high. (def: 255)
    :param low: scale min value to low. (def: 0)
    :return: 8-bit image data array
    """
    if data.dtype == np.uint8:
        return data
    
    if high > 255:
        high = 255
    if low < 0:
        low = 0
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")
    
    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()
        
    cscale = cmax - cmin
    if cscale == 0:
        cscale = 1
        
    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


'''Superimpose one image on another
import numpy as np
import cv2
This will really only work for source (foreground) images that have transparency 
set everywhere outside the main area / region of interest. 
- tests: 
    - reject if the background image is smaller than the foreground image
    - check for same bit depth?
    - Check that if an offset is allowed, it doesn't attempt to set pixels that don't exist
- x_offset, y_offset = COME BACK TO THIS, FOR NOW JUST CENTER ON THE BACKGROUND
- introduce some kind of blurring for the background? (GREAT IDEA BUT NOT A TEST)
'''
def superimpose(fore: np.ndarray, back: np.ndarray) -> np.ndarray:
    # Found on stackexchange and originally from skimage(?)
    # TODO check bit depths
    # TODO check that back has four channels - and create one if it doesn't.
    if(back.shape < fore.shape):
        print(f"Foreground is larger than background and it must be at most the same size")
        return None
    else:
        #Assuming centering for now
        x_offset = int(back.shape[0] / 2)
        y_offset = int(back.shape[1] / 2)
        back = cv2.GaussianBlur(back, (15, 15), 0)
        fb, fg, fr, fa = cv2.split(fore)
        sa_nz = np.nonzero(fa)
        back[sa_nz[0] + x_offset, sa_nz[1] + y_offset, :] = fore[sa_nz[0], sa_nz[1], :]
        return back


# make the input for this a string so the greyscale image can be picked up by the gc as soon as this fn exits
def find_axis_cuts(img: str) -> tuple:
    # read image in as greyscale
    im = cv2.imread(img, 0)
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(im)[0]
    #f2_im_lines = lsd.drawSegments(f2_im, lines)
    lines = np.round(lines)
    linediff_h = np.abs(lines[:, :, 1] - lines[:, :, 3])
    linediff_w = np.abs(lines[:, :, 0] - lines[:, :, 2])
    
    #index of the longest line segment on the horizontal axis
    h_idx = np.argmax(linediff_h)
    w_idx = np.argmax(linediff_w)
    
    #the cut point is the index on the original array associated with opposite index 
    h_cut_after_here = np.max(lines[h_idx][0][[0,2]]).astype("int")
    
    w_cut_before_here = np.max(lines[w_idx][0][[1,3]]).astype("int")
    
    return(w_cut_before_here, h_cut_after_here)


#for type MC_API_NLMIMAGE_V1.3  -> 4708 images
#import cv2
def cut_img_axes(cuts: Tuple, imgs: str) -> np.ndarray:
    img = cv2.imread(imgs, cv2.IMREAD_UNCHANGED)
    h_cut, w_cut = cuts
    img = img[:h_cut, w_cut:, :]
    return img

def random_bg() -> np.ndarray:
    max_size = 224
    bg_dir = '/home/izzy/projects/vision/pilldata/backgrounds'
    backgrounds = os.listdir(bg_dir)
    n_bg = len(backgrounds)
    
    rng = np.random.default_rng()
    
    r_bg = rng.integers(0, n_bg).item()
    #print('RIP RBG')
    
    bg = cv2.imread(f'{bg_dir}/{backgrounds[r_bg]}', cv2.IMREAD_UNCHANGED)
    h, w = bg.shape[:2]
    
    h_min = rng.integers(0, h - max_size)
    h_max = h_min + max_size
    
    w_min = rng.integers(0, w - max_size)
    w_max = w_min + max_size
    
    new_bg = bg[h_min:h_max, w_min:w_max, :]
    return new_bg