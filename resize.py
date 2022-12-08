import cv2
import numpy as np

def sane_resizing(image, max_size: int = 224, inter=cv2.INTER_LANCZOS4):
    print(f'Entering resize')
    print(f'{image.shape}')
    (h, w) = image.shape[:2]
    dim = None
    
    # Randomly resize the foreground to somewhere between 150 and 223
    # based on the current model
    # rng.integers returns one larger than the max
    max_size = max_size - 1
    min_size = int(np.floor(max_size/2))
    
    rng = np.random.default_rng()
    size = rng.integers(min_size, max_size).item()
    
    if h == w:
        #width, height = size
        dim = (size, size)
    elif h > w:
        width, height = None, size
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        width, height = size, None
        r = width / float(w)
        dim = (width, int(h * r))
        
    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    
    # return the resized image
    return resized


