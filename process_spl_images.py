from iz_val import read_pill_data_n, testdir
from iz_ip import find_axis_cuts, cut_img_axes
import numpy as np
import cv2
import os
from get_spl import get_spl
'''
This was made to facilitate the cleaning of images with Layout = 'MC_SPL_SPLIMAGE_V3.0'
Specifically, this removes the axes/scale on the left hand side and bottom of the image,
and then separates the image into two. The separating is done by creating a mask with the same shape
as the input image, with the mask valued at RGB 118,118,118(uint8) to target the grey background of the input. After
substracting, 25 rows/columns on either side of the axis center are evaluated to find the lowest row or
columnwise mean, the idea being that color values less than this which are actually the pill will underflow, 
and the grey (approx. 118,118,118) background will be approximately zero.  
The image is cut on the axis, and at the index with the lowest mean.  
'''
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


    

path = '/home/ngs/pillproject/final_dataset'
'''
NDC11 is only unique when combined with Part so trying to make directories for NDC11 alone will have dupes
and these pills actually have different imprints at times
This is something that should probably be done on the main df level
'''
spl = get_spl()

#spl_dupes = spl[spl.duplicated(subset=['NDC11'], keep = False)]
spl['ndc11_part'] = spl['NDC11'].astype(str) + spl['Part'].astype(str)
spl['ndc11_part'] = spl['ndc11_part'].astype('string')

splnames = [name for name in spl['Name']]
dlfiles = os.listdir('/home/ngs/download_images_only_non_sorted')
dr = '/home/ngs/download_images_only_non_sorted'
missing = [name for name in splnames if not name in dlfiles]
spl_to_process = [name for name in splnames if name in dlfiles]

'''
Splitting images by shape is not very reliable. 
if(img.shape == 'CAPSULE'):
    a = np.mean(im[middle - x: middle + x, :, :], axis = 1)
    img1 = im[0:cut, :, :]
    img2 = im[cut:im.shape[0], :, :]
    cv2.imwrite(f'{path}/{img.ndcp}/SF_{img.name})
else:
    a = np.mean(im[middle - x: middle + x, :, :], axis = 0)
    img1 = im[:, 0:cut, :]
    img2 = im[:, cut:im.shape[1], :]
    cv2.imwrite(f'{path}/{img.ndcp}/SF_{img.name})
    cv2.imwrite(f'{path}/{img.ndcp}/SB_{img.name})
'''

for name in spl_to_process:
    x = spl.loc[spl['Name'] == name]
    img = ImageProps(x[['ndc11_part']].squeeze(),
                     x[['NDC11']].squeeze(), 
                     x[['Shape']].squeeze(), 
                     x[['Size']].squeeze(), 
                     x[['Color']].squeeze(), 
                     x[['Imprint']].squeeze(), 
                     x[['Name']].squeeze(), 
                     x[['Source']].squeeze()
                     )
    testdir(f'{path}/{img.ndcp}')
    cuts = find_axis_cuts(f'{dr}/{img.name}')
    im = cut_img_axes(cuts, f'{dr}/{img.name}')
    h_mid = int(im.shape[0]/2)
    v_mid = int(im.shape[1]/2)
    # padding on either side of the center to check for the mean
    # basic idea: look at 25 rows/columns on either side of the axis center
    # find the row/column with the lowest mean - should be close to zero after substracting the
    # grey background (118,118,118, dtype=np.uint8 below)
    offset = 25
    # create mask to target grey background, goal to get ~0
    mask = np.full_like(im, 118, dtype=np.uint8)
    imm = im - mask
    h_slice = imm[h_mid - offset: h_mid + offset, :, :]
    w_slice = imm[:, v_mid - offset:v_mid + offset, :]
    # If the average value of h_slice is less than the average value of w_slice:
    #   - the split - i.e. the grey background between the pills is along the 
    #   - 
    #   - that section likely contained grey originally (~RGB 118,118,118)  
    #   - that shows the cut needs to be made horizontally 
    #   - 
    if(np.mean(h_slice) < np.mean(w_slice)):
        cut = h_mid - offset + int(np.argmin(np.mean(h_slice, axis = 1))/3)
        img1 = im[0:cut,:, :]
        img2 = im[cut:, :, :]
        front = path + '/' + img.ndcp + '/' + 'SF_' + img.name
        back = path + '/' + img.ndcp + '/' + 'SB_' + img.name
        cv2.imwrite(front, img1)
        cv2.imwrite(back, img2)
    else:
        cut = v_mid - offset + int(np.argmin(np.mean(w_slice, axis = 0))/3)
        img1 = im[:, 0:cut, :]
        img2 = im[:, cut:, :]
        front = path + '/' + img.ndcp + '/' + 'SF_' + img.name
        back = path + '/' + img.ndcp + '/' + 'SB_' + img.name
        cv2.imwrite(front, img1)
        cv2.imwrite(back, img2)
    

for x,y in list(zip(df2['Name'], df2['Source'])):
    r = requests.get(f'{root}/{y}/images/{x}')
    open(x, 'wb').write(r.content)