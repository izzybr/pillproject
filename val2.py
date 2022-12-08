import numpy as np
import shutil
import os
from iz_val import testdir
'''
This was written to move a set of images grouped only by name.
'''


rng = np.random.default_rng()
root_dir = '/home/ngs/pillproject/augmented/Color'
val_dir = '/home/ngs/pillproject/augmented/validation'

val_fraction = 0.2
test = False
if(test):
    vals = ['TESTCOLOR']
else:
    vals = os.listdir(root_dir)

doublecheck = []

for val in vals:
    for root, subdirs, files in os.walk(f'{root_dir}/{val}'):
        dlist = []
        for file in files:
            dlist.append(
                        os.path.join(root_dir, val, file)
                        )
        ld = len(dlist)
        idx = rng.choice(ld, 
                         int(
                             np.floor(ld * val_fraction)
                             ), 
                        replace=False)
        testdir(f'{val_dir}')
        testdir(f'{val_dir}/{val}')
        local_val_dir = f'{val_dir}/{val}/'
        for x in idx:
            doublecheck.append(dlist[x])
            print(f'Moving {dlist[x]} to {local_val_dir}')
            shutil.move(dlist[x], local_val_dir)
    

            