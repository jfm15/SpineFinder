# The aim of this script is to produce files which contain the dense labelling associated with the scans and save them

import os
import numpy as np

print("hello")

base_directory = '/homes/jfm15/SpineFinder/'
mask_directory = base_directory + 'masks/'

import sys
sys.path.append(base_directory)

import global_variables
from utility_functions import opening_files, dense_labeler, labels_to_file

"""EXTRACT INFORMATION FROM FILES"""
scans_directory = global_variables.spine_dataset_path

"""DEFINE CONSTANTS"""
scales = np.array([0.3125, 0.3125, 2.5])
radii = 13

limit = 1

for patient_file in (next(os.walk(scans_directory))[1])[:limit]:
    for scan_number in next(os.walk(scans_directory + '/' + patient_file))[1]:
        full_directory = scans_directory + '/' + patient_file + '/' + scan_number + '/' + scan_number

        """GET SCAN AND LABELS"""
        scan = opening_files.read_nii(full_directory + '.nii')
        labels, centroids = opening_files.extract_centroid_info_from_lml(full_directory + '.lml')

        """SLICE SCAN AND LABELLING"""
        scaled_centroids = centroids / scales
        cut_indices = []
        dense_labels = []

        for i in range(0, len(labels)):
            best_transverse_cut = int(round(scaled_centroids[i, 0]))

            scan_slice = scan[best_transverse_cut, :, :]

            """GET DENSE LABELLING"""
            cut_indices.append(best_transverse_cut)
            dense_labels.append(dense_labeler.generate_dense_labelling_2D(scan, best_transverse_cut, centroids, radii, scales, False))

        patient_folder_path = mask_directory + patient_file

        if not os.path.exists(patient_folder_path):
            os.mkdir(patient_folder_path)

        scan_folder_path = patient_folder_path + "/" + str(scan_number)

        if not os.path.exists(scan_folder_path):
            os.mkdir(scan_folder_path)

        file_path = scan_folder_path + "/" + str(scan_number) + "-sagittal-slices"
        labels_to_file.save_2d_dense_labelling(file_path, cut_indices, dense_labels)
