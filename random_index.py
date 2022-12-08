import numpy as np

def random_index(min, max):
    '''
    This is a simple wrapper for np.random.default_rng().choice
    '''
    rng = np.random.default_rng()
    return(rng.choice(min,
                        max
                    )
    ) 
    