import numpy as np


def hogtt(window: np.ndarray):
    '''
    Separating the HOG function out for timing - want to test if 
    '''
    
    coords = []
    #features = []
    
    hIm,wIm = image.shape[:2] 
    
    # W1 will start from 0 to end of image - window size
    # W2 will start from window size to end of image
    # We'll use step (stride) like convolution kernels.
    for w1,w2 in zip(range(0,wIm-windowSize[0],step),range(windowSize[0],wIm,step)):
        for h1,h2 in zip(range(0,hIm-windowSize[1],step),range(windowSize[1],hIm,step)):
            window = img[h1:h2,w1:w2]
            hg = hog(window,
                     block_norm='L2-Hys',
                     transform_sqrt=True
                     ).reshape(1, -1)
            
            cls = model.predict(hg)
            
            probA = model.predict_proba(hg)
            #features_of_window = hog(window,orientations=9,pixels_per_cell=(16,16),
            #                         cells_per_block=(2,2)
            #                        )
            
            coords.append((w1,w2,h1,h2, cls, probA[0][0], probA[0][1]))