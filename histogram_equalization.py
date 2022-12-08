import numpy as np

from skimage import data, io, exposure, img_as_float
from skimage.color import rgb2gray
# contrast stretching
root = '/home/ngs/pillproject/PillProjectDisc1/images/'
ims = '!0X7C0H7OX3J65RAO-1-1HH7NPH6K2.JPG'
im = io.imread(f'{root}{ims}')

im = img_as_float(im)
im = rgb2gray(im)
p2, p98 = np.percentile(im, (2, 98))
im_rescale = exposure.rescale_intensity(im, in_range=(p2, p98))

#######
# histogram equalization

im_eq = exposure.equalize_hist(im)


###
# adaptive equalization

img_adapteq = exposure.equalize_adapthist(im, clip_limit=0.03)