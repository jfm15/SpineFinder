import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utility_functions import dense_labelling as dl, opening_files as of
from utility_functions import helper_functions as hf
import os

"""DEFINE CONSTANTS"""
scales = np.array([0.3125, 0.3125, 2.5])
# block size in mm
block_size = np.array([96, 112, 32])

training_data = np.empty(np.append(0, block_size))
training_data_labels = np.empty(0)

dir = './spine-1'
for patient_file in next(os.walk(dir))[1]:
    for scan_number in next(os.walk(dir + '/' + patient_file))[1]:
        if patient_file == 'patient0030':
            continue

        full_dir = dir + '/' + patient_file + '/' + scan_number + '/' + scan_number
        scan = of.read_nii(full_dir + '.nii')
        labels, centroids = of.extract_centroid_info_from_lml(full_dir + '.lml')

        random_range = scan.shape - block_size

        x = 0
        for i in range(10):
            x += 1
            print(x)
            point_1 = np.round(np.random.rand(3) * random_range).astype(int)
            point_2 = point_1 + block_size

            sample = scan[point_1[0]:point_2[0], point_1[1]:point_2[1], point_1[2]:point_2[2]]
            training_data = np.append(training_data, [sample], axis=0)

            for idx, centroid in enumerate(centroids):
                centroid_indices = hf.real_to_indexes(centroid, scales)
                if (centroid_indices >= point_1).all() and (centroid_indices <= point_2).all():
                    training_data_labels = np.append(training_data_labels, 1)
                else:
                    training_data_labels = np.append(training_data_labels, 0)


