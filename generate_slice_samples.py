import glob
import numpy as np
from utility_functions import opening_files
from utility_functions.sampling_helper_functions import densely_label


def generate_slice_samples(dataset_dir, sample_dir, sample_size=(40, 160), diameter=(28.0, 28.0, 28.0), spacing=(2.0, 2.0, 2.0),
                           no_of_samples=5, no_of_vertebrae_in_each=2, file_ext=".nii.gz"):
    sample_size = np.array(sample_size)
    ext_len = len(file_ext)

    for data_path in glob.glob(dataset_dir + "/**/*" + file_ext, recursive=True):

        # get path to corresponding metadata
        data_path_without_ext = data_path[:-ext_len]
        metadata_path = data_path_without_ext + ".lml"

        volume = opening_files.read_nii(data_path, spacing=spacing)
        # print(volume.shape)
        labels, centroids = opening_files.extract_centroid_info_from_lml(metadata_path)
        centroid_indexes = np.round(centroids / np.array(spacing)).astype(int)

        dense_labelling = densely_label(labels, volume.shape, centroid_indexes, spacing, diameter, use_labels=True)

        radius = (np.array(diameter) / np.array(spacing)) / 2.0
        lower_i = np.min(centroid_indexes[:, 0])
        lower_i = np.max([lower_i - radius[0], 0]).astype(int)
        upper_i = np.max(centroid_indexes[:, 0])
        upper_i = np.min([upper_i + radius[0], volume.shape[0] - 1]).astype(int)

        '''
        lower_j = np.min(centroid_indexes[:, 1])
        lower_j = np.max([lower_j - radius[1], 0]).astype(int)
        upper_j = np.max(centroid_indexes[:, 1])
        upper_j = np.min([upper_j + radius[1], volume.shape[1] - 1]).astype(int)

        lower_k = np.min(centroid_indexes[:, 2])
        lower_k = np.max([lower_k - radius[2], 0]).astype(int)
        upper_k = np.max(centroid_indexes[:, 2])
        upper_k = np.min([upper_k + radius[2], volume.shape[2] - 1]).astype(int)
        '''

        cuts = []
        while len(cuts) < no_of_samples:
            cut = np.random.randint(lower_i, high=upper_i)
            sample_labels_slice = dense_labelling[cut, :, :]
            if np.unique(sample_labels_slice).shape[0] > no_of_vertebrae_in_each:
                cuts.append(cut)

        name = (data_path.rsplit('/', 1)[-1])[:-ext_len]
        print(name)

        for idx, i in enumerate(cuts):

            volume_slice = volume[i, :, :]
            sample_labels_slice = dense_labelling[i, :, :]

            # crop or pad depending on what is necessary
            if volume_slice.shape[0] < sample_size[0]:
                dif = sample_size[0] - volume_slice.shape[0]
                volume_slice = np.pad(volume_slice, ((0, dif), (0, 0)), mode="constant")
                sample_labels_slice = np.pad(sample_labels_slice, ((0, dif), (0, 0)), mode="constant")

            if volume_slice.shape[1] < sample_size[1]:
                dif = sample_size[1] - volume_slice.shape[1]
                volume_slice = np.pad(volume_slice, ((0, 0), (0, dif)), mode="constant")
                sample_labels_slice = np.pad(sample_labels_slice, ((0, 0), (0, dif)), mode="constant")

            while True:
                random_area = volume_slice.shape - sample_size
                random_factor = np.random.rand(2)
                random_position = np.round(random_area * random_factor).astype(int)
                corner_a = random_position
                corner_b = corner_a + sample_size

                cropped_volume_slice = volume_slice[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1]]
                cropped_sample_labels_slice = sample_labels_slice[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1]]

                if np.unique(cropped_sample_labels_slice).shape[0] > no_of_vertebrae_in_each:
                    break

            # save file
            name_plus_id = name + "-" + str(idx)
            path = '/'.join([sample_dir, name_plus_id])
            sample_path = path + "-sample"
            labelling_path = path + "-labelling"
            np.save(sample_path, cropped_volume_slice)
            np.save(labelling_path, cropped_sample_labels_slice)


generate_slice_samples(dataset_dir="datasets/",
                       sample_dir="samples/slices",
                       sample_size=(40, 160),
                       diameter=(28.0, 28.0, 28.0),
                       no_of_samples=20,
                       no_of_vertebrae_in_each=2)