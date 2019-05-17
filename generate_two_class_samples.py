import glob
from utility_functions import opening_files, processing
import numpy as np
import scipy.ndimage
from utility_functions.labels import LABELS
from utility_functions.sampling_helper_functions import densely_label


def generate_samples(dataset_dir, sample_dir,
                     spacing, radius, sample_size,
                     no_of_samples, no_of_zero_samples,
                     use_labels=False, rotate=False, file_ext=".nii.gz"):

    # numpy these so they can be divided later on
    radius = np.array(radius)
    sample_size = np.array(sample_size)
    cut_size = sample_size + np.array([10, 10, 10])

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

        volume = opening_files.read_nii(data_path)

        # densely populate
        dense_labelling = densely_label(labels=labels,
                                        volume_shape=volume.shape,
                                        centroid_indexes=centroid_indexes,
                                        spacing=spacing,
                                        radius=radius,
                                        use_labels=use_labels)

        sample_size_in_pixels = (sample_size / np.array(spacing)).astype(int)
        cut_size_in_pixels = (cut_size / np.array(spacing)).astype(int)
        cut_size_in_pixels = np.clip(cut_size_in_pixels, a_min=np.zeros(3), a_max=volume.shape - np.ones(3)).astype(int)

        if rotate:
            random_area = volume.shape - cut_size_in_pixels
        else:
            random_area = volume.shape - sample_size_in_pixels

        name = (data_path.rsplit('/', 1)[-1])[:-ext_len]
        print(name)
        i = 0
        j = 0
        while i < no_of_samples:

            random_factor = np.random.rand(3)
            random_position = np.round(random_area * random_factor).astype(int)
            corner_a = random_position
            if rotate:
                corner_b = random_position + cut_size_in_pixels
            else:
                corner_b = random_position + sample_size_in_pixels

            sample = volume[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1], corner_a[2]:corner_b[2]]
            labelling = dense_labelling[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1], corner_a[2]:corner_b[2]]

            # randomly rotate
            if rotate:
                angle = np.random.rand() * 20.0 - 10.0
                sample = scipy.ndimage.interpolation.rotate(sample, angle, axes=(-1, 1), mode="constant")
                labelling = scipy.ndimage.interpolation.rotate(labelling, angle, axes=(-1, 1), mode="constant")
                labelling = np.round(labelling).astype(int)

                padding_corner_a = np.round((sample.shape - sample_size_in_pixels) / 2.0).astype(int)
                padding_corner_b = padding_corner_a + sample_size_in_pixels
                sample = sample[padding_corner_a[0]:padding_corner_b[0], padding_corner_a[1]:padding_corner_b[1],
                         padding_corner_a[2]:padding_corner_b[2]]
                labelling = labelling[padding_corner_a[0]:padding_corner_b[0], padding_corner_a[1]:padding_corner_b[1],
                         padding_corner_a[2]:padding_corner_b[2]]

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
                 sample_dir="samples/two_class",
                 spacing=(2.0, 2.0, 2.0),
                 radius=(28.0, 28.0, 28.0),
                 sample_size=(60.0, 60.0, 72.0),
                 no_of_samples=60,
                 no_of_zero_samples=6,
                 use_labels=False,
                 rotate=False)
