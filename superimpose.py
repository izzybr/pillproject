import numpy as np
import cv2

def superimpose(fore: np.ndarray, back: np.ndarray, bool_mask: bool = True) -> np.ndarray:
    '''Superimpose one image on another
- fore: foreground image (pill)
- back: background image
- return_sem: Return a boolean mask with the new image
This will really only work for source (foreground) images that have transparency 
set everywhere outside the main area / region of interest. 
- tests: 
    - reject if the background image is smaller than the foreground image
    - check for same bit depth?
'''
        # TODO check that back has four channels - and create one if it doesn't.
    try:    
        if(back.shape < fore.shape):
            raise ValueError('Foreground is larger than background and it must be at most the same size')
        elif(back.shape[::-1][0] != 4):
            raise ValueError('No alpha channel present')
        else:
            rng = np.random.default_rng()
            
            max_offset_x = back.shape[0] - fore.shape[0] - 1
            max_offset_y = back.shape[1] - fore.shape[1] - 1
            
            x_offset = rng.integers(0, max_offset_x).item()
            y_offset = rng.integers(0, max_offset_y).item()
            
            b, g, r, fa = cv2.split(fore)
            fa_nz = np.nonzero(fa)
            back[fa_nz[0] + x_offset, fa_nz[1] + y_offset, :] = fore[fa_nz[0], fa_nz[1], :]
            
            if(bool_mask):
                bool_mask = np.ones_like(back[...,3], dtype=np.uint8)
                bool_mask[fa_nz[0] + x_offset, fa_nz[1] + y_offset] = 0
            else:
                bool_mask = None
        return (back, bool_mask)
    finally:
        pass