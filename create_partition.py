import glob
import numpy as np


def create_partition_and_labels(training_samples_dir, val_samples_dir):

    partition = {}
    training_labels = []
    validation_labels = []
    labels = {}

    ext_len = len("-labelling.npy")
    training_paths = glob.glob(training_samples_dir + "/**/*labelling.npy", recursive=True)
    val_paths = glob.glob(val_samples_dir + "/**/*labelling.npy", recursive=True)

    for training_path in training_paths:

        sample_path_without_ext = training_path[:-ext_len]
        label = sample_path_without_ext.rsplit('/', 1)[1]

        training_labels.append(label)

        # read file and assign to labels
        labels[label] = label + "-labelling"

    for val_path in val_paths:
        sample_path_without_ext = val_path[:-ext_len]
        label = sample_path_without_ext.rsplit('/', 1)[1]

        training_labels.append(label)

        # read file and assign to labels
        labels[label] = label + "-labelling"

    partition["train"] = training_labels
    partition["validation"] = validation_labels
    return partition, labels
