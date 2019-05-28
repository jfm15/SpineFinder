import glob
import numpy as np
import elasticdeform
from utility_functions import opening_files
from utility_functions.sampling_helper_functions import densely_label, pre_compute_disks


def generate_slice_samples(dataset_dir, sample_dir, spacing, no_of_samples, make_multiple=16, file_ext=".nii.gz"):

    ext_len = len(file_ext)

    paths = glob.glob(dataset_dir + "/**/*" + file_ext, recursive=True)

    for cnt, data_path in enumerate(paths):

        print(str(cnt) + " / " + str(len(paths)))
        # get path to corresponding metadata
        data_path_without_ext = data_path[:-ext_len]
        metadata_path = data_path_without_ext + ".lml"

        volume = opening_files.read_nii(data_path, spacing=spacing)

        # get cropping amount
        padding = make_multiple - np.mod(volume.shape, make_multiple)
        padding = np.mod(padding, make_multiple)
        padding_list = list(zip(np.zeros(3).astype(int), padding))[1:3]

        # print(volume.shape)
        labels, centroids = opening_files.extract_centroid_info_from_lml(metadata_path)
        centroid_indexes = np.round(centroids / np.array(spacing)).astype(int)

        disk_indices = pre_compute_disks(spacing)
        dense_labelling = densely_label(volume.shape, disk_indices, labels, centroid_indexes, use_labels=True)

        cuts = []
        while len(cuts) < no_of_samples:
            cut = np.random.randint(0, high=volume.shape[0])
            sample_labels_slice = dense_labelling[cut, :, :]
            if np.unique(sample_labels_slice).shape[0] > 1:
                cuts.append(cut)

        name = (data_path.rsplit('/', 1)[-1])[:-ext_len]

        count = 0
        for i in cuts:

            volume_slice = volume[i, :, :]
            sample_labels_slice = dense_labelling[i, :, :]

            # get vertebrae identification map
            # detection_slice = (sample_labels_slice > 0).astype(int)

            [volume_slice, sample_labels_slice] = elasticdeform.deform_random_grid(
                [volume_slice, sample_labels_slice], sigma=7, points=3, order=0)

            volume_slice = np.pad(volume_slice, padding_list, mode="constant")
            sample_labels_slice = np.pad(sample_labels_slice, padding_list, mode="constant")

            # save file
            count += 1
            name_plus_id = name + "-" + str(count)
            path = '/'.join([sample_dir, name_plus_id])
            sample_path = path + "-sample"
            labelling_path = path + "-labelling"
            np.save(sample_path, volume_slice)
            np.save(labelling_path, sample_labels_slice)


generate_slice_samples(dataset_dir="datasets/spine-1",
                       sample_dir="samples/slices",
                       spacing=(1.0, 1.0, 1.0),
                       no_of_samples=64,
                       make_multiple=16)