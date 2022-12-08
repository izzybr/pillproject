import numpy as np
import cv2
import argparse
import os
from data_gen2 import read_nih_data
from image_class2 import C3PI_Ref, SplImage
from iz_val import testdir


ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dir', type=str, required=False, default = './sample_images', 
                help="path to local images")

ap.add_argument('-n', '--num', type = int, default = 5, required=False, 
                help="Number of unique images to augment")

ap.add_argument('-n_a', '--n_augs', type = int, default = 10, required = False, 
                help="Number or augmentations per unique image")

ap.add_argument('-o', '--output_dir', type = str, default = './output', required = False, 
                help = f"Directory to write images to. Attempts to write to {os.getcwd()} if none specified.\n")

ap.add_argument('-l', '--local_metadata', type = str, default = './data', required = False,
                help = 'Directory with local metadata. Default ./data\n')

ap.add_argument('-f', '--from', type = str, default = None, required = False,
                help= 'Process images from this directory\n')
#ap.add_argument('-a', '--augment', type = str, )

args = vars(ap.parse_args())

df = read_nih_data()
fd = df.loc[df['imgclass'] == 'MC_C3PI_REFERENCE_SEG_V1.6']
rng = np.random.default_rng()
mask = rng.choice(len(fd), args['num'], replace = False)
fd = fd.iloc[mask]

print('Processing MC_C3PI_REFERENCE_SEG_V1.6 images\n')
for ndc11, part, location, imgclass, med_name in list(zip(fd['ndc11'], fd['part'], fd['location'], fd['imgclass'], fd['med_name'])):
     xxx = C3PI_Ref(ndc11, part, location, imgclass, med_name)
     xxx.make_image3(args['num'])

fd = df.loc[df['imgclass'] == 'MC_SPL_SPLIMAGE_V3.0']
rng = np.random.default_rng()
mask = rng.choice(len(fd), args['num'], replace = False)
fd = fd.iloc[mask]

path = args['output_dir']
print('Processing SPL Images')
for ndc11, part, location, imgclass, med_name in list(zip(fd['ndc11'], fd['part'], fd['location'], fd['imgclass'], fd['med_name'])):
     xxx = SplImage(ndc11, part, location, imgclass, med_name)
     img1, img2 = xxx.split_spl()
     testdir(f'{path}/{xxx.ndcp}')
     print(f'Processed {xxx.name}, writing to {path}/{xxx.ndcp}\n')
     cv2.imwrite(f'{path}/{xxx.ndcp}/SF_{xxx.name}', img1)
     cv2.imwrite(f'{path}/{xxx.ndcp}/SB_{xxx.name}', img2)