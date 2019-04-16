# The aim of this script is to produce files which contain the dense labelling associated with the scans and save them

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
sys.path.append('/homes/jfm15/SpineFinder/')
from utility_functions import opening_files, dense_labelling

"""EXTRACT INFORMATION FROM FILES"""
scans_directory = '/vol/bitbucket2/jfm15/spine-1'

"""DEFINE CONSTANTS"""
scales = np.array([0.3125, 0.3125, 2.5])
radii = 13

i = 0
#for patient_file in next(os.walk(scans_directory))[1]:
for patient_file in ['patient0013']:
    for scan_number in next(os.walk(scans_directory + '/' + patient_file))[1]:
        if i > 0:
            break
        print(patient_file)
        i += 1
        full_directory = scans_directory + '/' + patient_file + '/' + scan_number + '/' + scan_number

        """GET SCAN AND LABELS"""
        scan = opening_files.read_nii(full_directory + '.nii')
        labels, centroids = opening_files.extract_centroid_info_from_lml(full_directory + '.lml')

        """SLICE SCAN AND LABELLING"""
        scaled_centroids = centroids / scales
        best_transverse_cut = int(round(np.mean(scaled_centroids[:, 0])))
        #best_transverse_cut = int(round(scaled_centroids[5, 0]))

        scan_slice = scan[best_transverse_cut, :, :]

        """GET DENSE LABELLING"""
        dense_labelling = dense_labelling.generate_dense_labelling_2D(scan, best_transverse_cut, centroids, radii, scales)

        """GENERATE MASK"""
        masked_data = np.ma.masked_where(dense_labelling == 0, dense_labelling)

        """SHOW RESULTS"""
        # imshow takes a matrix n x m and plots n up the y axis so we transpose it
        plt.imshow(scan_slice.T, interpolation="none", aspect=8, origin='lower')
        plt.imshow(masked_data.T, interpolation="none", aspect=8, origin='lower', cmap=cm.jet, alpha=1)

plt.show()