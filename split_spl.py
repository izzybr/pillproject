from typing import tuple
import numpy as np
import cv2
##

def split_spl(im: np.ndarray, output_dir: str = './output') -> tuple(np.ndarray, np.ndarray):     
    h_mid = int(im.shape[0]/2)
    v_mid = int(im.shape[1]/2)
    # padding on either side of the center to check for the mean
    # basic idea: look at 25 rows/columns on either side of the axis center
    # find the row/column with the lowest mean - should be close to zero after substracting the
    # grey background (118,118,118, dtype=np.uint8 below)
    offset = 25
    # create mask to target grey background, goal to get ~0
    # Keep mask  as uint8 and take advantage of integer underflow so that 
    # color values less than the grey background will wrap around
    mask = np.full_like(im, 118, dtype=np.uint8)
    imm = im - mask
    a = imm[h_mid - offset: h_mid + offset, :, :]
    b = imm[:, v_mid - offset:v_mid + offset, :]
    # This isn't perfect by any means BUT the area being targeted is 
    # essentially a row (or column) of values that are RGB (118, 118, 118)
    # 
    path = output_dir
    if(np.mean(a) < np.mean(b)):
        cut = h_mid - offset + int(np.argmin(np.mean(a, axis = 1))/3)
        img1 = im[0:cut,:, :]
        img2 = im[cut:, :, :]
        cv2.imwrite(f'{path}/{self.ndcp}/SF_{self.name}', img1)
        cv2.imwrite(f'{path}/{self.ndcp}/SB_{self.name}', img2)
    else:
        cut = v_mid - offset + int(np.argmin(np.mean(b, axis = 0))/3)
        img1 = im[:, 0:cut, :]
        img2 = im[:, cut:, :]
        cv2.imwrite(f'{path}/{self.ndcp}/SF_{self.name}', img1)
        cv2.imwrite(f'{path}/{self.ndcp}/SB_{self.name}', img2)
    return (img1, img2)