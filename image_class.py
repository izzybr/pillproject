import requests
import cv2
import numpy as np
from typing import Tuple
from data_gen2 import read_nih_data
import re

def get_image(location):
        r = requests.get(location)
        if r.ok:
            img = np.asarray(bytearray(r.content), dtype='uint8')
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            return img
        
class SplImage(PillProject):
    kind = 'spl_image'
    
    def __init__(self,
                 ndc11,
                 part,
                 location,
                 type,
                 med_name,
                 #root = 'https://data.lhncbc.nlm.nih.gov/public/Pills/'
                 ):
        
        self.ndc11 = ndc11
        self.part = part
        self.ndcp = ndc11 + part
        #self.root = root
        #This would need to include something to indicate that the images are local (if in fact they are)
        self.location = 'https://data.lhncbc.nlm.nih.gov/public/Pills/' + location
        self.name = re.split(r'/images/', location)[1]
        self.img = np.ndarray
        self.axis_cuts = tuple
        self.type = type
        self.med_name = med_name
        self.rm_ax = False
    
    def get(self):
        r = requests.get(self.location)
        if r.ok:
            self.img = np.asarray(bytearray(r.content), dtype='uint8')
            self.img = cv2.imdecode(self.img, cv2.IMREAD_COLOR)
            self.name = re.split(r'/images/', self.location)[1]
            return self.img
    
    def find_axis_cuts(self) -> tuple:
        if not isinstance(self.img, np.ndarray):
            self.get()
            
        # read image in as greyscale
        grey = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        lsd = cv2.createLineSegmentDetector(0)
        lines = lsd.detect(grey)[0]
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
    
class C3PI_Ref_Image:    
    

df = read_nih_data()
A = df.loc[133773]
xxx = SplImage(A.ndc11, A.part, A.location, A.imgclass, A.med_name)

for ndc11, part, location, imgclass, med_name in list(zip(df2['ndc11'], df2['part'], df2['location'], df2['imgclass'], df2['med_name'])):
    xxx = SplImage(ndc11, part, location, imgclass, med_name)
    testdir(f'./output/{xxx.ndcp}')
    cv2.imwrite(f'./output/{xxx.ndcp}/SF_{xxx.name}', img1)
    cv2.imwrite(f'./output/{xxx.ndcp}/SB_{xxx.name}', img2)



#cv2.cvtColor(img, cv.BGR_TO)
    
class ImageProps:
    def __init__(self,
                 ndcp, 
                 ndc11,
                 type,
                 size,  
                 color,
                 imprint,
                 name,
                 source
                 ):
        self.ndcp = ndcp
        self.ndc11 = ndc11
        self.type = type
        self.size = size
        self.color = color
        self.imprint = imprint
        self.name = name
        self.source = source