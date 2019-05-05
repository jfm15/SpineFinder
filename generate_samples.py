import glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches
from utility_functions import opening_files, dense_labeler
import SimpleITK as sitk

root_path = os.getcwd()
dataset_dir = "/".join([root_path, "datasets", "spine-1"])
sample_dir = "/".join([root_path, "samples"])
scales = np.array([0.3125, 0.3125, 2.5])
radii = 13
sample_size = np.array([128, 128, 32])


def generate_samples(dataset_dir, sample_dir, scales, sample_size, no_of_samples):

    ext_len = len(".nii.gz")
    prefix_len = len(dataset_dir)
    # get paths of all volumes
    for volume_path in glob.glob(dataset_dir + "/**/*.nii.gz", recursive=True):

        # get path to corresponding metadata
        volume_path_without_ext = volume_path[:-ext_len]
        volume_metadata_path = volume_path_without_ext + ".lml"

        # get real data from paths
        volume = opening_files.read_nii(volume_path, spacing=(2.0, 2.0, 2.0))
        labels, centroids = opening_files.extract_centroid_info_from_lml(volume_metadata_path)

        dense_labels = dense_labeler.generate_dense_labelling_3D(volume, centroids, radii, scales)

        # make directory to put samples in
        volume_path_suffix = volume_path_without_ext[prefix_len:]
        # volume_in_samples_dir = (sample_dir + volume_path_suffix).rsplit('/', 1)[0]
        # os.makedirs(volume_in_samples_dir)

        random_area = volume.shape - sample_size

        for i in range (0, no_of_samples):

            random_factor = np.random.rand(3)
            random_position = np.round(random_area * random_factor).astype(int)
            corner_a = random_position
            corner_b = random_position + sample_size

            sample = volume[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1], corner_a[2]:corner_b[2]]
            labelling = dense_labels[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1], corner_a[2]:corner_b[2]]

            name = volume_path_suffix.rsplit('/', 1)[1] + '-' + str(i)
            sample_id = name + '-sample'
            sample_path = "/".join([sample_dir, sample_id])
            np.save(sample_path, sample)

            labelling_id = name + '-labelling'
            labelling_path = "/".join([sample_dir, labelling_id])
            np.save(labelling_path, labelling)

            print(name, np.unique(labelling))


generate_samples(dataset_dir, sample_dir, scales, sample_size, 10)