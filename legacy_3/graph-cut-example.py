import numpy as np
import imcut.pycut as pspc
import matplotlib.pyplot as plt
from utility_functions import opening_files

spacing = (1.0, 1.0, 1.0)
volume = opening_files.read_nii("datasets/spine-1/patient0088/2684937/2684937.nii.gz", spacing=spacing)
labels, centroids = opening_files.extract_centroid_info_from_lml("datasets/spine-1/patient0088/2684937/2684937.lml")
centroids = centroids / np.array(spacing)
chosen_centroid = np.round(centroids[10]).astype(int)

# create data
data = volume * 100
data = data.astype(np.int16)

data_slice = data[chosen_centroid[0], :, :]

# Make seeds
seeds = np.ones(data.shape) * 2

for centroid in centroids:
    corner_a = centroid - 25
    corner_a = np.round(np.clip(corner_a, a_min=np.zeros(3), a_max=data.shape - np.ones(3))).astype(int)
    corner_b = centroid + 25
    corner_b = np.round(np.clip(corner_b, a_min=np.zeros(3), a_max=data.shape - np.ones(3))).astype(int)
    seeds[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1], corner_a[2]:corner_b[2]] = 0

for i in range(centroids.shape[0] - 1):
    dist = np.linalg.norm(centroids[i + 1] - centroids[i])
    spline = np.round(np.linspace(centroids[i], centroids[i + 1], num=np.round(dist).astype(int) * 2)).astype(int)
    for point in spline:
        u, v, w = point
        seeds[u-4:u+4, v-4:v+4, w-4:w+4] = 1
'''
seeds[0:6, 0:30, 0:11] = 2

# Run
'''
igc = pspc.ImageGraphCut(data, voxelsize=[1, 1, 1])
igc.set_seeds(seeds)
igc.run()


# Show results
colormap = plt.cm.get_cmap('brg')
colormap._init()
colormap._lut[:1:,3]=0


plt.imshow(data_slice.T, cmap='gray', origin="lower")
#plt.scatter(chosen_centroid[1], chosen_centroid[2], color="red", s=8)
#plt.imshow(data[:, :, 10], cmap='gray')
seg = igc.segmentation[chosen_centroid[0], :, :]
plt.imshow(seg.T, alpha=0.4)
#plt.imshow(seeds[chosen_centroid[0], :, :].T, cmap=colormap, interpolation='none', alpha=0.4)
plt.show()