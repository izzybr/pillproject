import numpy as np

def bettercrop(im) -> np.ndarray:
    #im = cv2.imread(im, cv2.IMREAD_UNCHANGED)
    im = im[:1600, :, :]
    #b,g,r,a = cv2.split(im)
    #x = np.max([b,g,r])
    #b, g, r = np.clip([b,g,r], x, 0)
    #mask = cv2.merge([b,g,r,a])
    #im = cv2.bitwise_or(im, mask)
    return im
