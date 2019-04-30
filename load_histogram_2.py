import numpy as np
from utility_functions import opening_files
import matplotlib.pyplot as plt
import matplotlib.cm as cm

volume = opening_files.read_nii("datasets/spine-1/patient0023/4542094/4542094.nii.gz")

output = np.load("histogram.npy")

_, width, height, depth, channels = output.shape

output = output.reshape(width, height, depth, channels)

histogram = np.empty([512, 512, 186])

for x in range(width):
    for y in range(height):
        for z in range(depth):
            displacement = output[x, y, z]
            pos = np.array([x, y, z]) * 16.0 + displacement
            pos = np.clip(pos, np.zeros(3), np.ones(3) * 511)
            pos = list(map(int, pos))
            histogram[pos[0], pos[1], pos[2]] += 1

slice = volume[252, :, :]
histogram_slice = np.sum(histogram, axis=0)

masked_data = np.ma.masked_where(histogram_slice == 0, histogram_slice)

plt.imshow(slice.T, interpolation="none", aspect=8, origin='lower')
plt.imshow(masked_data.T, interpolation="none", aspect=8, origin='lower', cmap=cm.jet, alpha=1)
plt.show()