import numpy as np
from utility_functions import opening_files as of, dense_labeler as dl
import os

"""DEFINE CONSTANTS"""
scales = np.array([0.3125, 0.3125, 2.5])
radii = 13

"""DEFINE TRAINING AND TEST"""
training_data = np.empty([1,1,1])
print(training_data.shape)
training_data_labels = np.empty(1)
test_data = np.empty(1)
test_data_labels = np.empty(1)

dir = './spine-1'
i = 0
for patient_file in next(os.walk(dir))[1]:
    for scan_number in next(os.walk(dir + '/' + patient_file))[1]:
        full_dir = dir + '/' + patient_file + '/' + scan_number + '/' + scan_number
        scan = of.read_nii(full_dir + '.nii')
        labels, centroids = of.extract_centroid_info_from_lml(full_dir + '.lml')

        for centroid in centroids:
            scaled_centroid = centroid / scales
            cut = int(round(scaled_centroid[0]))
            dense_labelling = dl.generate_dense_labelling_2D(scan, cut, centroids, radii, scales)
            training_data = np.append(training_data, [scan[cut, :, :]], axis=0)
            training_data_labels = np.append(training_data_labels, dense_labelling, axis=0)
            i += 1
            print(i)