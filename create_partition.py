import glob, os
import numpy as np

def create_partition_and_labels(samples_dir, training_percentage, randomise=True):

    partition = {}
    training_labels = []
    validation_labels = []
    labels = {}

    ext_len = len(".npy")
    paths = glob.glob(samples_dir + "/**/*.txt", recursive=True)
    no_of_training = round(len(paths) * training_percentage)

    if randomise:
        np.random.shuffle(paths)

    for i, sample_path in enumerate(paths):

        sample_path_without_ext = sample_path[:-ext_len]
        label = sample_path_without_ext.rsplit('/', 1)[1]

        # assign to lists for partition
        if i < no_of_training:
            training_labels.append(label)
        else:
            validation_labels.append(label)

        # read file and assign to labels
        metadata_string = open(sample_path, "r").read()
        metadata_split = metadata_string.split(" ")
        centroid_coords = list(map(int, metadata_split[1:]))
        labels[label] = centroid_coords

    partition["training"] = training_labels
    partition["validation"] = validation_labels
    return partition, labels
