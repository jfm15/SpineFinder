import glob
from utility_functions import opening_files, processing
import numpy as np
import scipy.ndimage
from utility_functions.labels import LABELS
from utility_functions.sampling_helper_functions import densely_label, pre_compute_disks


def generate_samples(dataset_dir, sample_dir,
                     spacing, X_size, y_size,
                     no_of_samples, no_of_zero_samples,
                     file_ext=".nii.gz"):

    # numpy these so they can be divided later on
    X_size = np.array(X_size)
    y_size = np.array(y_size)
    offset = ((X_size - y_size) / 2.0).astype(int)

    ext_len = len(file_ext)

    disk_indices = pre_compute_disks(spacing)

    paths = glob.glob(dataset_dir + "/**/*" + file_ext, recursive=True)

    for cnt, data_path in enumerate(paths):

        print(str(cnt) + " / " + str(len(paths)))

        # get path to corresponding metadata
        data_path_without_ext = data_path[:-ext_len]
        metadata_path = data_path_without_ext + ".lml"

        # get image, resample it and scale centroids accordingly
        labels, centroids = opening_files.extract_centroid_info_from_lml(metadata_path)
        centroid_indexes = np.round(centroids / np.array(spacing)).astype(int)

        volume = opening_files.read_nii(data_path, spacing=spacing)

        # densely populate
        dense_labelling = densely_label(volume.shape, disk_indices, labels, centroid_indexes, use_labels=False)
        X_size_in_pixels = (X_size / np.array(spacing)).astype(int)
        y_size_in_pixels = (y_size / np.array(spacing)).astype(int)

        # crop or pad depending on what is necessary
        if volume.shape[0] < X_size_in_pixels[0]:
            dif = X_size_in_pixels[0] - volume.shape[0]
            volume = np.pad(volume, ((0, dif), (0, 0), (0, 0)),
                                  mode="constant", constant_values=-5)
            dense_labelling = np.pad(dense_labelling, ((0, dif), (0, 0), (0, 0)),
                                         mode="constant")

        if volume.shape[1] < X_size_in_pixels[1]:
            dif = X_size_in_pixels[1] - volume.shape[1]
            volume = np.pad(volume, ((0, 0), (0, dif), (0, 0)),
                                  mode="constant", constant_values=-5)
            dense_labelling = np.pad(dense_labelling, ((0, 0), (0, dif), (0, 0)),
                                         mode="constant")

        if volume.shape[2] < X_size_in_pixels[2]:
            dif = X_size_in_pixels[2] - volume.shape[2]
            volume = np.pad(volume, ((0, 0), (0, 0), (0, dif)),
                                  mode="constant", constant_values=-5)
            dense_labelling = np.pad(dense_labelling, ((0, 0), (0, 0), (0, dif)),
                                         mode="constant")

        random_area = volume.shape - X_size_in_pixels

        name = (data_path.rsplit('/', 1)[-1])[:-ext_len]
        i = 0
        j = 0
        while i < no_of_samples:

            random_factor = np.random.rand(3)
            random_position = np.round(random_area * random_factor).astype(int)
            corner_a = random_position
            corner_b = random_position + X_size_in_pixels

            corner_c = random_position + offset
            corner_d = corner_c + y_size_in_pixels

            sample = volume[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1], corner_a[2]:corner_b[2]]
            labelling = dense_labelling[corner_c[0]:corner_d[0], corner_c[1]:corner_d[1], corner_c[2]:corner_d[2]]

            # if a centroid is contained
            unique_labels = np.unique(labelling).shape[0]
            if unique_labels > 1 or j < no_of_zero_samples:
                if unique_labels == 1:
                    j += 1
                i += 1

                # save file
                name_plus_id = name + "-" + str(i)
                path = '/'.join([sample_dir, name_plus_id])
                sample_path = path + "-sample"
                labelling_path = path + "-labelling"
                if np.all(sample.shape == X_size_in_pixels):
                    np.save(sample_path, sample)
                    np.save(labelling_path, labelling)
                else:
                    print(data_path, volume.shape)


generate_samples(dataset_dir="datasets",
                 sample_dir="samples/two_class",
                 spacing=(1.0, 1.0, 1.0),
                 X_size=(68.0, 68.0, 84.0),
                 y_size=(28.0, 28.0, 44.0),
                 no_of_samples=60,
                 no_of_zero_samples=6)
