import numpy as np
from utility_functions import opening_files
import matplotlib.pyplot as plt
import matplotlib.cm as cm

volume = opening_files.read_nii("datasets/spine-1/patient0023/4542094/4542094.nii.gz")

histogram = np.load("histogram.npy")

slice = volume[252, :, :]
#histogram_slice = np.sum(histogram, axis=0)
histogram_slice = histogram[250, :, :]

print(histogram_slice)

masked_data = np.ma.masked_where(histogram_slice == 0, histogram_slice)

#plt.imshow(histogram_slice.T, interpolation="none", aspect=8, origin='lower')
plt.imshow(slice.T, interpolation="none", aspect=8, origin='lower')
plt.imshow(masked_data.T, interpolation="none", aspect=8, origin='lower', cmap=cm.jet, alpha=1)
plt.show()