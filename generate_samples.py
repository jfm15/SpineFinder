import glob
from utility_functions import opening_files, processing
import SimpleITK as sitk
import numpy as np


def densely_label(labels, volume_shape, centroid_indexes, spacing, radius, use_labels, label_translation):

    diameter_in_pixels = radius / np.array(spacing)
    radius_in_pixels = ((diameter_in_pixels - np.ones(3)) / 2.0).astype(int)

    dense_labelling = np.zeros(volume_shape)

    upper_clip = volume_shape - np.ones(3)

    for label, centroid_idx in zip(labels, centroid_indexes):

        corner_a = centroid_idx - radius_in_pixels
        corner_a = np.clip(corner_a, a_min=np.zeros(3), a_max=upper_clip).astype(int)
        corner_b = centroid_idx + radius_in_pixels
        corner_b = np.clip(corner_b, a_min=np.zeros(3), a_max=upper_clip).astype(int)

        label_value = 1
        if use_labels:
            label_value = label_translation.index(label)

        dense_labelling[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1], corner_a[2]:corner_b[2]] = label_value

    return dense_labelling


def generate_samples(dataset_dir, sample_dir,
                     spacing, radius, sample_size,
                     no_of_samples, no_of_zero_samples,
                     use_labels=False, label_translation=[], file_ext=".nii.gz"):

    # numpy these so they can be divided later on
    radius = np.array(radius)
    sample_size = np.array(sample_size)

    ext_len = len(file_ext)

    # track quantities of various labels using this histogram
    no_of_values = 2
    if use_labels:
        no_of_values = len(label_translation)
    values_histogram = np.zeros(no_of_values)

    for data_path in glob.glob(dataset_dir + "/**/*" + file_ext, recursive=True):

        # get path to corresponding metadata
        data_path_without_ext = data_path[:-ext_len]
        metadata_path = data_path_without_ext + ".lml"

        # get image, resample it and scale centroids accordingly
        image = sitk.ReadImage(data_path)
        image = processing.resample_image(image, out_spacing=spacing)
        labels, centroids = opening_files.extract_centroid_info_from_lml(metadata_path)
        centroid_indexes = np.round(centroids / np.array(spacing)).astype(int)

        volume = sitk.GetArrayFromImage(image).T

        # densely populate
        dense_labelling = densely_label(labels=labels,
                                        volume_shape=volume.shape,
                                        centroid_indexes=centroid_indexes,
                                        spacing=spacing,
                                        radius=radius,
                                        use_labels=use_labels,
                                        label_translation=label_translation)

        sample_size_in_pixels = (sample_size / np.array(spacing)).astype(int)
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
                np.save(sample_path, sample)
                np.save(labelling_path, labelling)

    values_histogram = np.sum(values_histogram) - values_histogram
    values_histogram /= np.sum(values_histogram)
    values_histogram = np.around(values_histogram, decimals=4)
    print(values_histogram)


label_translation = ["B", "C1", "C2", "C3", "C4", "C5", "C6", "C7",
                     "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9",
                     "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5", "L6"
                     "S1", "S2"]

generate_samples(dataset_dir="datasets/",
                 sample_dir="samples/multi_class",
                 spacing=(2.0, 2.0, 2.0),
                 radius=(28.0, 28.0, 28.0),
                 sample_size=(56.0, 56.0, 56.0),
                 no_of_samples=120,
                 no_of_zero_samples=20,
                 use_labels=True,
                 label_translation=label_translation)
