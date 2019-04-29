from keras.models import load_model
import numpy as np
from utility_functions import opening_files
import matplotlib.pyplot as plt

model = load_model('main-model.h5')

volume = opening_files.read_nii("datasets/spine-1/patient0023/4542094/4542094.nii.gz")

# apply model to each block
patch_size = np.array([256, 256, 32])
volume_size = volume.shape
collect = volume_size - patch_size

histogram = np.zeros(volume.shape)

print(collect)
for i in range(0, collect[0], 32):
    for j in range(0, collect[1], 32):
        for k in range(0, collect[2], 8):
            corner_a = np.array([i, j, k])
            corner_b = corner_a + patch_size
            sample = volume[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1], corner_a[2]:corner_b[2]]
            sample = sample.reshape(1, 256, 256, 32, 1)
            result = model.predict(sample).reshape(3)
            z = corner_a + result.astype(int)
            histogram[z[0], z[1], z[2]] += 1
            print(z)

np.save("histogram", histogram)

'''
metadata_string = open('samples/2804506-L5.txt', "r").read()
metadata_split = metadata_string.split(" ")
centroid_coords = list(map(int, metadata_split[1:]))

print(centroid_coords)

plt.show()
'''