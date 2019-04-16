import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utility_functions import dense_labelling as dl, opening_files as of

"""EXTRACT INFORMATION FROM FILES"""
dir = '/vol/bitbucket2/jfm15/spine-1/patient0013/4511471/4511471'
scan = of.read_nii(dir + '.nii')
labels, centroids = of.extract_centroid_info_from_lml(dir + '.lml')

"""DEFINE CONSTANTS"""
scales = np.array([0.3125, 0.3125, 2.5])
radii = 13

"""SLICE SCAN AND LABELLING"""
scaled_centroids = centroids / scales
best_transverse_cut = int(round(np.mean(scaled_centroids[:, 0])))

scan_slice = scan[best_transverse_cut, :, :]

"""GET DENSE LABELLING"""
dense_labelling = dl.generate_dense_labelling_2D(scan, best_transverse_cut, centroids, radii, scales)

"""GENERATE MASK"""
masked_data = np.ma.masked_where(dense_labelling == 0, dense_labelling)

"""SHOW RESULTS"""
# imshow takes a matrix n x m and plots n up the y axis so we transpose it
plt.imshow(scan_slice.T, interpolation="none", aspect=8, origin='lower')
plt.imshow(masked_data.T, interpolation="none", aspect=8, origin='lower', cmap=cm.jet, alpha=1)
plt.show()