import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utility_functions import dense_labelling as dl, opening_files as of
import os

"""DEFINE CONSTANTS"""
scales = np.array([0.3125, 0.3125, 2.5])
radii = 13

"""EXTRACT INFORMATION FROM FILES"""
dir = './spine-1'
i = 0
for patient_file in next(os.walk(dir))[1]:
    for scan_number in next(os.walk(dir + '/' + patient_file))[1]:
        i = i + 1
        if i > 5:
            break
        full_dir = dir + '/' + patient_file + '/' + scan_number + '/' + scan_number
        scan = of.read_nii(full_dir + '.nii')
        labels, centroids = of.extract_centroid_info_from_lml(full_dir + '.lml')

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
        plt.subplot(1,5,i)
        plt.imshow(scan_slice.T, interpolation="none", aspect=8, origin='lower')
        plt.imshow(masked_data.T, interpolation="none", aspect=8, origin='lower', cmap=cm.jet, alpha=1)
        print(i, "complete")

plt.show()
