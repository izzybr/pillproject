from typing import tuple
import numpy as np
import cv2

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