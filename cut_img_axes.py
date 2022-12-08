import cv2
from typing import tuple
import numpy as np
#for type MC_API_NLMIMAGE_V1.3  -> 4708 images
def cut_img_axes(cuts: tuple, imgs: str) -> np.ndarray:
    img = cv2.imread(imgs, cv2.IMREAD_UNCHANGED)
    h_cut, w_cut = cuts
    img = img[:h_cut, w_cut:, :]
    return img