import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utility_functions import opening_files as of, dense_labeler as dl

"""EXTRACT INFORMATION FROM FILES"""
dir = './spine-1/patient0013/4511471/4511471'
scan = of.read_nii(dir + '.nii')
labels, centroids = of.extract_centroid_info_from_lml(dir + '.lml')

"""DEFINE CONSTANTS"""
scales = np.array([0.3125, 0.3125, 2.5])
radii = 13

"""GET DENSE LABELLING"""
dense_labelling = dl.generate_dense_labelling_3D(scan, centroids, radii, scales)

"""SLICE SCAN AND LABELLING"""
scaled_centroids = centroids / scales
best_transverse_cut = int(round(np.mean(scaled_centroids[:, 0])))

scan_slice = scan[best_transverse_cut, :, :]
dense_labelling_slice = dense_labelling[best_transverse_cut, :, :]

"""GENERATE MASK"""
masked_data = np.ma.masked_where(dense_labelling_slice == 0, dense_labelling_slice)

"""SHOW RESULTS"""
# imshow takes a matrix n x m and plots n up the y axis so we transpose it
plt.imshow(scan_slice.T, interpolation="none", aspect=8, origin='lower')
plt.imshow(masked_data.T, interpolation="none", aspect=8, origin='lower', cmap=cm.jet, alpha=1)
plt.show()