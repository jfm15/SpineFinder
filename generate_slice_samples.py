import glob
import numpy as np
import elasticdeform
from utility_functions import opening_files
from utility_functions.sampling_helper_functions import densely_label


def generate_slice_samples(dataset_dir, sample_dir, sample_size=(40, 160), spacing=(2.0, 2.0, 2.0),
                           no_of_samples=5, no_of_vertebrae_in_each=2, file_ext=".nii.gz"):
    sample_size = np.array(sample_size)
    ext_len = len(file_ext)

    paths = glob.glob(dataset_dir + "/**/*" + file_ext, recursive=True)

    for cnt, data_path in enumerate(paths):

        print(str(cnt) + " / " + str(len(paths)))
        # get path to corresponding metadata
        data_path_without_ext = data_path[:-ext_len]
        metadata_path = data_path_without_ext + ".lml"

        volume = opening_files.read_nii(data_path, spacing=spacing)

        # print(volume.shape)
        labels, centroids = opening_files.extract_centroid_info_from_lml(metadata_path)
        centroid_indexes = np.round(centroids / np.array(spacing)).astype(int)

        dense_labelling = densely_label(volume.shape, labels, centroid_indexes, spacing, use_labels=True)

        # dense_labelling_squashed = np.any(dense_labelling, axis=(1, 2))
        # lower_i = np.min(np.where(dense_labelling_squashed == 1))
        # upper_i = np.max(np.where(dense_labelling_squashed == 1))
        lower_i = np.min(centroid_indexes[:, 0])
        lower_i = np.max([lower_i - 10, 0]).astype(int)
        upper_i = np.max(centroid_indexes[:, 0])
        upper_i = np.min([upper_i + 10, volume.shape[0] - 1]).astype(int)

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

        j = 0
        cuts = []
        while len(cuts) < no_of_samples:
            cut = np.random.randint(lower_i, high=upper_i)
            sample_labels_slice = dense_labelling[cut, :, :]
            j += 1
            care_about = np.count_nonzero(sample_labels_slice)
            if care_about > 5000 or (care_about > 2500 and j > 50) or j > 100:
                cuts.append(cut)

        name = (data_path.rsplit('/', 1)[-1])[:-ext_len]

        for idx, i in enumerate(cuts):

            volume_slice = volume[i, :, :]
            sample_labels_slice = dense_labelling[i, :, :]

            [volume_slice, sample_labels_slice] = elasticdeform.deform_random_grid(
                [volume_slice, sample_labels_slice], sigma=7, points=3, order=0)

            # crop or pad depending on what is necessary
            if volume_slice.shape[0] < sample_size[0]:
                dif = sample_size[0] - volume_slice.shape[0]
                volume_slice = np.pad(volume_slice, ((0, dif), (0, 0)),
                                      mode="constant", constant_values=-5)
                sample_labels_slice = np.pad(sample_labels_slice, ((0, dif), (0, 0)),
                                             mode="constant")

            if volume_slice.shape[1] < sample_size[1]:
                dif = sample_size[1] - volume_slice.shape[1]
                volume_slice = np.pad(volume_slice, ((0, 0), (0, dif)),
                                      mode="constant", constant_values=-5)
                sample_labels_slice = np.pad(sample_labels_slice, ((0, 0), (0, dif)),
                                             mode="constant")

            '''
            j = 0
            while True:
                random_area = volume_slice.shape - sample_size
                random_factor = np.random.rand(2)
                random_position = np.round(random_area * random_factor).astype(int)
                corner_a = random_position
                corner_b = corner_a + sample_size

                cropped_volume_slice = volume_slice[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1]]
                cropped_sample_labels_slice = sample_labels_slice[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1]]

                care_about_labels = np.count_nonzero(cropped_sample_labels_slice)
                j += 1
                if care_about_labels > 5000 or j > 100:
                    print(j)
                    break
            '''

            # save file
            name_plus_id = name + "-" + str(idx)
            path = '/'.join([sample_dir, name_plus_id])
            sample_path = path + "-sample"
            labelling_path = path + "-labelling"
            np.save(sample_path, volume_slice)
            np.save(labelling_path, sample_labels_slice)


generate_slice_samples(dataset_dir="datasets",
                       sample_dir="samples/slices",
                       sample_size=(272, 720),
                       no_of_samples=20,
                       spacing=(1.0, 1.0, 1.0),
                       no_of_vertebrae_in_each=1)