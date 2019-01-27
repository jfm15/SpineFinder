import numpy as np


# assumes indexes is as [transverse, sagittal, vertical]
def indexes_to_real(indexes, scales):
    return indexes * scales


# assumes real is as [transverse, sagittal, vertical]
def real_to_indexes(real, scales):
    return np.around(real / scales).astype(int)