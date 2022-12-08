import numpy as np
import cv2

def small_img_rect(image: np.ndarray):
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