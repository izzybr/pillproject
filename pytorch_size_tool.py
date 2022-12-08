import numpy as np
def output_size(input_size, padding, k_size, stride = 1):
    return np.floor((input_size + 2 * padding - (k_size - 1) - 1)/stride) + 1
