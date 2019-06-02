import glob
from utility_functions import opening_files, processing
import numpy as np
import scipy.ndimage
from utility_functions.labels import LABELS
from utility_functions.sampling_helper_functions import densely_label


def generate_samples(dataset_dir, sample_dir,
                     spacing, sample_size,
                     no_of_samples, no_of_zero_samples,
                     use_labels=False, file_ext=".nii.gz"):

    # numpy these so they can be divided later on
    sample_size = np.array(sample_size)

    ext_len = len(file_ext)

    # track quantities of various labels using this histogram
    no_of_values = 2
    if use_labels:
        no_of_values = len(LABELS)
    values_histogram = np.zeros(no_of_values)

    for data_path in glob.glob(dataset_dir + "/**/*" + file_ext, recursive=True):

        # get path to corresponding metadata
        data_path_without_ext = data_path[:-ext_len]
        metadata_path = data_path_without_ext + ".lml"

        # get image, resample it and scale centroids accordingly
        labels, centroids = opening_files.extract_centroid_info_from_lml(metadata_path)
        centroid_indexes = np.round(centroids / np.array(spacing)).astype(int)

        volume = opening_files.read_nii(data_path, spacing=spacing)

        # densely populate
        dense_labelling = densely_label(labels=labels,
                                        volume_shape=volume.shape,
                                        centroids=centroid_indexes,
                                        use_labels=use_labels)

        sample_size_in_pixels = (sample_size / np.array(spacing)).astype(int)

        # crop or pad depending on what is necessary
        if volume.shape[0] < sample_size_in_pixels[0]:
            dif = sample_size_in_pixels[0] - volume.shape[0]
            volume = np.pad(volume, ((0, dif), (0, 0), (0, 0)),
                                  mode="constant", constant_values=-5)
            dense_labelling = np.pad(dense_labelling, ((0, dif), (0, 0), (0, 0)),
                                         mode="constant")

        if volume.shape[1] < sample_size_in_pixels[1]:
            dif = sample_size_in_pixels[1] - volume.shape[1]
            volume = np.pad(volume, ((0, 0), (0, dif), (0, 0)),
                                  mode="constant", constant_values=-5)
            dense_labelling = np.pad(dense_labelling, ((0, 0), (0, dif), (0, 0)),
                                         mode="constant")

        if volume.shape[2] < sample_size_in_pixels[2]:
            dif = sample_size_in_pixels[2] - volume.shape[2]
            volume = np.pad(volume, ((0, 0), (0, 0), (0, dif)),
                                  mode="constant", constant_values=-5)
            dense_labelling = np.pad(dense_labelling, ((0, 0), (0, 0), (0, dif)),
                                         mode="constant")

        random_area = volume.shape - sample_size_in_pixels

        name = (data_path.rsplit('/', 1)[-1])[:-ext_len]
        print(name)
        i = 0
        j = 0
        while i < no_of_samples:

            random_factor = np.random.rand(3)
            random_position = np.round(random_area * random_factor).astype(int)
            corner_a = random_position
            corner_b = random_position + sample_size_in_pixels

            sample = volume[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1], corner_a[2]:corner_b[2]]
            labelling = dense_labelling[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1], corner_a[2]:corner_b[2]]

            # if a centroid is contained
            unique_labels = np.unique(labelling).shape[0]
            if unique_labels > 1 or j < no_of_zero_samples:
                if unique_labels == 1:
                    j += 1
                i += 1

                # update histogram
                values_histogram += np.bincount(labelling.reshape(-1).astype(int), minlength=no_of_values)

                # save file
                name_plus_id = name + "-" + str(i)
                path = '/'.join([sample_dir, name_plus_id])
                sample_path = path + "-sample"
                labelling_path = path + "-labelling"
                if np.all(sample.shape == sample_size_in_pixels):
                    np.save(sample_path, sample)
                    np.save(labelling_path, labelling)
                else:
                    print(data_path, volume.shape)

    values_histogram = np.sum(values_histogram) - values_histogram
    values_histogram /= np.sum(values_histogram)
    values_histogram = np.around(values_histogram, decimals=4)
    print(values_histogram)


generate_samples(dataset_dir="datasets/",
                 sample_dir="samples/two_class/training",
                 spacing=(1.0, 1.0, 1.0),
                 sample_size=(64.0, 64.0, 80.0),
                 no_of_samples=20,
                 no_of_zero_samples=2,
                 use_labels=False)