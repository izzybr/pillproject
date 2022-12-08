from skimage import transform
import numpy as np

def arbitrary_rotation(im: np.ndarray, angle: float) -> np.ndarray:
    print(f'entering rotation')
    '''
    Arbitrary rotation of an array
    '''
    rotated_image = transform.rotate(im, angle, preserve_range=True).astype(np.uint8)
    return rotated_image
