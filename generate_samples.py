import glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches
from utility_functions import opening_files

root_path = os.getcwd()
dataset_dir = "/".join([root_path, "datasets", "spine-1"])
sample_dir = "/".join([root_path, "samples"])
scales = np.array([0.3125, 0.3125, 2.5])
sample_size = np.array([128, 128, 32])


def generate_samples(dataset_dir, sample_dir, scales, sample_size, no_of_samples):

    ext_len = len(".nii.gz")
    prefix_len = len(dataset_dir)
    # get paths of all volumes
    for volume_path in glob.glob(dataset_dir + "/**/*.nii.gz", recursive=True)[:5]:

        # get path to corresponding metadata
        volume_path_without_ext = volume_path[:-ext_len]
        volume_metadata_path = volume_path_without_ext + ".lml"

        # get real data from paths
        volume = opening_files.read_nii(volume_path)
        labels, centroids = opening_files.extract_centroid_info_from_lml(volume_metadata_path)
        scaled_centroids = np.round(centroids / scales).astype(int)

        # pad volume so we can sample from it
        half_sample_size = (sample_size / 2).astype(int)
        volume_paddings = np.array(list(zip(sample_size, sample_size)))
        padded_volume = np.pad(volume, volume_paddings, "constant", constant_values=0)

        # make directory to put samples in
        volume_path_suffix = volume_path_without_ext[prefix_len:]
        # volume_in_samples_dir = (sample_dir + volume_path_suffix).rsplit('/', 1)[0]
        # os.makedirs(volume_in_samples_dir)

        for label, scaled_centroid in zip(labels, scaled_centroids):

            for i in range (0, no_of_samples):

                # random element
                random_component = np.round(np.random.rand(3) * sample_size * 0.5 - half_sample_size * 0.5).astype(int)

                # remember this takes into account padding
                corner_a = scaled_centroid + half_sample_size + random_component
                corner_b = scaled_centroid + half_sample_size + sample_size + random_component

                centroid_position = half_sample_size - random_component

                sample = padded_volume[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1], corner_a[2]:corner_b[2]]

                sample_id = volume_path_suffix.rsplit('/', 1)[1] + '-' + str(i) + "-" + label
                sample_path = "/".join([sample_dir, sample_id])
                np.save(sample_path, sample)

                annotation_file = open(sample_path + ".txt", "w+")
                annotation_file.write(" ".join([label] + list(map(str, centroid_position))))
                annotation_file.close()

                print(sample_id)

            """
            fig, ax = plt.subplots(1)
            ax.imshow(sample[128, :, :].T, interpolation="none", aspect=8, origin='lower')
            ax.annotate(label, centroid_position[1:3], color="red")
            ax.scatter(centroid_position[1], centroid_position[2], color="red")
            plt.show()
            """

            """
            best_transverse_cut = scaled_centroid[0]

            centroid_position = scaled_centroid[1:3]
            bottom_left = centroid_position - sample_size[1:3] / 2.0

            fig, ax = plt.subplots(1)

            ax.imshow(volume[best_transverse_cut, :, :].T, interpolation="none", aspect=8, origin='lower')
            ax.annotate(label, centroid_position, color="red")
            ax.scatter(centroid_position[0], centroid_position[1], color="red")
            ax.add_patch(plt_patches.Rectangle(bottom_left, sample_size[1], sample_size[2], linewidth=1, edgecolor='r', facecolor='none'))
            plt.show()
            """


generate_samples(dataset_dir, sample_dir, scales, sample_size, 4)