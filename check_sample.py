import numpy as np
import matplotlib.pyplot as plt

sample_path = "samples/2551924-0-T1"

sample = np.load(sample_path + ".npy")

metadata_string = open(sample_path + ".txt", "r").read()
metadata_split = metadata_string.split(" ")
label = metadata_split[0]
centroid_coords = list(map(int, metadata_split[1:]))

slice = sample[centroid_coords[0], :, :]

fig, ax = plt.subplots(1)
ax.imshow(slice.T, interpolation="none", aspect=8, origin='lower')
ax.annotate(label, centroid_coords[1:3], color="red")
ax.scatter(centroid_coords[1], centroid_coords[2], color="red")
plt.show()