import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from utility_functions import opening_files
from utility_functions import processing
import matplotlib.cm as cm

img_T1 = sitk.ReadImage('datasets/spine-1/patient0088/2684937/2684937.nii.gz')
img_T1 = processing.resample_image(img_T1, out_spacing=(2.0, 2.0, 2.0))

labels, centroids = opening_files.extract_centroid_info_from_lml('datasets/spine-1/patient0088/2684937/2684937.lml')
centroid_indexes = np.round(centroids / np.array((2.0, 2.0, 2.0))).astype(int)

seeds = [tuple(centroid_indexes[8, :])]
print(type(seeds))
# seeds = [(40, 44, 100)]
print(type(seeds))

seg_con = sitk.ConfidenceConnected(img_T1,
                                   seedList=seeds,
                                   numberOfIterations=1,
                                   multiplier=1.0,
                                   initialNeighborhoodRadius=3,
                                   replaceValue=1)

img_T1_np = sitk.GetArrayFromImage(img_T1).T
seg_con_np = sitk.GetArrayFromImage(seg_con).T


'''
labelling = np.zeros(volume.shape)

for label, centroid_idx in zip(labels, centroid_indexes):
    print("-----")
    i, j, k = centroid_idx
    for i_add in range(-1, 2):
        for j_add in range(-1, 2):
            for k_add in range(-1, 2):
                # next_point = (i + i_add, j + j_add, k + k_add)
                print(volume[i + i_add, j + j_add, k + k_add])

'''
# cut = np.round(np.mean(centroid_indexes[:, 0])).astype(int)
cut = centroid_indexes[8, 0]

volume_slice = img_T1_np[cut, :, :]
labelling_slice = seg_con_np[cut, :, :]

plt.imshow(volume_slice.T)
# plt.scatter([centroid_indexes[8, 1]], [centroid_indexes[8, 2]], color="red")
plt.imshow(labelling_slice.T, alpha=0.5)
plt.show()