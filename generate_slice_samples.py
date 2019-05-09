import glob
import numpy as np
from utility_functions import opening_files
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def generate_slice_samples(dataset_dir, sample_dir, no_of_samples=5, label_translation=[], file_ext=".nii.gz"):

    ext_len = len(file_ext)

    for data_path in glob.glob(dataset_dir + "/**/*" + file_ext, recursive=True):

        # get path to corresponding metadata
        data_path_without_ext = data_path[:-ext_len]
        metadata_path = data_path_without_ext + ".lml"

        volume = opening_files.read_nii(data_path)
        labels, centroids = opening_files.extract_centroid_info_from_lml(metadata_path)
        centroid_indexes = np.round(centroids / np.array((2.0, 2.0, 2.0))).astype(int)

        lower = np.min(centroid_indexes[:, 0])
        lower = np.max([lower - 4, 0])
        upper = np.max(centroid_indexes[:, 0])
        upper = np.min([upper + 4, volume.shape[0] - 1])

        cuts = np.round(np.linspace(lower, upper, no_of_samples)).astype(int)

        sample_labels = np.zeros(volume.shape)

        name = (data_path.rsplit('/', 1)[-1])[:-ext_len]
        print(name)

        for idx, i in enumerate(cuts):
            for j in range(0, volume.shape[1]):
                for k in range(0, volume.shape[2]):
                    label = -1
                    min_distance = 1000
                    for label_name, centroid_idx in zip(labels, centroid_indexes):
                        dist = np.linalg.norm((i, j, k) - centroid_idx)
                        if dist < min_distance:
                            min_distance = dist
                            label = label_translation.index(label_name)
                        sample_labels[i, j, k] = label

            volume_slice = volume[i, :, :]
            sample_labels_slice = sample_labels[i, :, :]

            # save file
            name_plus_id = name + "-" + str(idx)
            path = '/'.join([sample_dir, name_plus_id])
            sample_path = path + "-sample"
            labelling_path = path + "-labelling"
            np.save(sample_path, volume_slice)
            np.save(labelling_path, sample_labels_slice)


label_translation = ["B", "C1", "C2", "C3", "C4", "C5", "C6", "C7",
                     "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9",
                     "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5", "L6",
                     "S1", "S2"]

generate_slice_samples(dataset_dir="datasets/spine-1",
                       sample_dir="samples/slices",
                       no_of_samples=10,
                       label_translation=label_translation)