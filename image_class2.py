import numpy as np
import pandas as pd
import re
import requests
import cv2
from background import random_bg
from resize import sane_resizing
from rotation import arbitrary_rotation
from superimpose import superimpose
from smallest_rect import small_img_rect
from image_mask import bettercrop
from iz_ip import bytescaling
from iz_val import testdir
from data_gen2 import read_nih_data
import time
import shutil as sh

class PillProject:
    def __init__(self,
                 ndc11,
                 part,
                 location,
                 type,
                 med_name,
                 ):
        self.ndc11 = ndc11
        self.part = part
        self.ndcp = ndc11 + part
        #self.root = root
        self.location = 'https://data.lhncbc.nlm.nih.gov/public/Pills/' + location
        self.name = re.split(r'/images/', location)[1]
        self.img = np.ndarray
        self.type = type
        self.med_name = med_name
        
    def get(self):
        r = requests.get(self.location)
        if r.ok:
            img = np.asarray(bytearray(r.content), dtype='uint8')
            self.img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            return self.img

class SplImage(PillProject):
    def __init__(self,
                 ndc11,
                 part,
                 location,
                 type,
                 med_name,
                 ):
        PillProject.__init__(self, 
                             ndc11,
                             part,
                             location,
                             type,
                             med_name,
                 )
        self.ndc11 = ndc11
        self.part = part
        self.ndcp = ndc11 + part
        self.location = 'https://data.lhncbc.nlm.nih.gov/public/Pills/' + location
        self.name = re.split(r'/images/', location)[1]
        self.img = np.ndarray
        self.type = type
        self.med_name = med_name
        self.axis_cuts = tuple
        self.rm_ax = False
    
    def find_axis_cuts(self) -> tuple:
        if not isinstance(self.img, np.ndarray):
            self.get()
            
        # read image in as greyscale
        grey = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        lsd = cv2.createLineSegmentDetector(0)
        lines = lsd.detect(grey)[0]
        #im_lines = lsd.drawSegments(f2_im, lines)
        lines = np.round(lines)
        linediff_h = np.abs(lines[:, :, 1] - lines[:, :, 3])
        linediff_w = np.abs(lines[:, :, 0] - lines[:, :, 2])
        
        #index of the longest line segment on the horizontal axis
        h_idx = np.argmax(linediff_h)
        w_idx = np.argmax(linediff_w)
        
        #the cut point is the index on the original array associated with opposite index 
        h_cut_after_here = np.max(lines[h_idx][0][[0,2]]).astype("int")
        
        w_cut_before_here = np.max(lines[w_idx][0][[1,3]]).astype("int")
        self.axis_cuts = (w_cut_before_here, h_cut_after_here)
        return self.axis_cuts
    
    def cut_img_axes(self) -> np.ndarray:
        if not isinstance(self.axis_cuts, tuple):
            self.find_axis_cuts()
        h_cut, w_cut = self.axis_cuts
        self.img = self.img[:h_cut, w_cut:, :]
        self.rm_ax = True
        return self.img
    
    def split_spl(self) -> tuple():
        if self.rm_ax is False:
            self.cut_img_axes()
        h_mid = int(self.img.shape[0]/2)
        v_mid = int(self.img.shape[1]/2)
        # padding on either side of the center to check for the mean
        # basic idea: look at 25 rows/columns on either side of the axis center
        # find the row/column with the lowest mean - should be close to zero after substracting the
        # grey background (118,118,118, dtype=np.uint8 below)
        offset = 25
        # create mask to target grey background, goal to get ~0
        # Keep mask  as uint8 and take advantage of integer underflow so that 
        # color values less than the grey background will wrap around
        mask = np.full_like(self.img, 118, dtype=np.uint8)
        imm = self.img - mask
        a = imm[h_mid - offset: h_mid + offset, :, :]
        b = imm[:, v_mid - offset:v_mid + offset, :]
        # This isn't perfect by any means BUT the area being targeted is 
        # essentially a row (or column) of values that are RGB (118, 118, 118)
        # 
        if(np.mean(a) < np.mean(b)):
            cut = h_mid - offset + int(np.argmin(np.mean(a, axis = 1))/3)
            img1 = self.img[0:cut,:, :]
            img2 = self.img[cut:, :, :]
            #cv2.imwrite(f'{path}/{self.ndcp}/SF_{self.name}', img1)
            #cv2.imwrite(f'{path}/{self.ndcp}/SB_{self.name}', img2)
        else:
            cut = v_mid - offset + int(np.argmin(np.mean(b, axis = 0))/3)
            img1 = self.img[:, 0:cut, :]
            img2 = self.img[:, cut:, :]
            #cv2.imwrite(f'{path}/{self.ndcp}/SF_{self.name}', img1)
            #cv2.imwrite(f'{path}/{self.ndcp}/SB_{self.name}', img2)
        return (img1, img2)
    
class C3PI_Ref(PillProject):
    
    def get(self):
        r = requests.get(self.location, stream = True)
        if r.ok:
            try:
                with open(f'./data/{self.name}', 'wb') as f:
                    r.raw.decode_content = True
                    sh.copyfileobj(r.raw, f)
                    f.close()
                
                #img = np.frombuffer(r.content, dtype='uint16')
                self.img = cv2.imread(f'./data/{self.name}', cv2.IMREAD_UNCHANGED)
                return self.img
            finally:
                pass    
        else:
            print(f'Could not load {self.name}')
    
    def make_image3(self, n_augs: int, max_size: int = 224, output_dir: str = './output'):
        if not isinstance(self.img, np.ndarray):
            self.get()
        
        if(n_augs <= 0):
            raise ValueError('n_augs must be at least 1')
        
        if not isinstance(n_augs, int):
            raise TypeError('n_augs must be an integer >= 1')
        
        t1 = time.time()
        rng = np.random.default_rng()
        angles = rng.integers(1, 360, size = n_augs)
        augs = [i for i in range(0, n_augs)]
        bgs = []
        
        for aug in augs:
            bgs.append(random_bg())
        
        for angle, bg, aug in list(zip(angles, bgs, augs)):
            im, mask = superimpose(
                            sane_resizing(
                                small_img_rect(
                                    arbitrary_rotation(
                                        bytescaling(
                                            bettercrop(self.img)
                                        )
                                    , angle)
                                )
                            , max_size)
                        , bg)
            testdir(f'{output_dir}/images/{self.ndcp}/')
            testdir(f'{output_dir}/masks/{self.ndcp}/')
            cv2.imwrite(f'{output_dir}/images/{self.ndcp}/{aug}-{self.name}', im)
            cv2.imwrite(f'{output_dir}/masks/{self.ndcp}/{aug}-{self.name}', mask)
            
        print(time.time() - t1)
        print(f'\nn_augs = {n_augs}')
        return True

