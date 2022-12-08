import cv2 
import numpy as np
import os
from background import random_bg
from resize import sane_resizing
from rotation import arbitrary_rotation
from superimpose import superimpose
from smallest_rect import small_img_rect
from image_mask import bettercrop
from iz_ip import bytescaling
from skimage import io
import time
from skimage import transform
import iz_val

'''
Used to create a novel image from an input file
- image: path to image
- n_augs: number of augmentations per image (integer)
- output_dir: directory to write images to, default is ./output
'''

def make_image3(image: str, n_augs: int, output_dir = './output'):
    t1 = time.time()
    rng = np.random.default_rng()
    angles = rng.integers(1, 360, size = n_augs)
    bgs = []
    for i in range(0, n_augs):
        bgs.append(random_bg())
    for angle, bg in list(zip(angles, bgs)):
        im, mask = superimpose(
                        sane_resizing(
                            small_img_rect(
                                arbitrary_rotation(
                                    bytescaling(
                                        bettercrop(image)
                                    )
                                 , angle)
                            )
                        )
                     , bg)
        cv2.imwrite(f'{output_dir}/images/{image}', im)
        cv2.imwrite(f'{output_dir}/masks/{image}', mask)

    print(time.time() - t1)
    print(f'\nn_augs = {n_augs}')
    #print(t2)    
    return True
