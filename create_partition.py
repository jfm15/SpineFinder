import glob
import numpy as np


def create_partition_and_labels(samples_dir, training_percentage, randomise=True):

    partition = {}
    training_labels = []
    validation_labels = []
    labels = {}

    ext_len = len("-labelling.npy")
    paths = glob.glob(samples_dir + "/**/*labelling.npy", recursive=True)
    scans = list(map(lambda x: x.split('-')[0], paths))
    scans = np.unique(np.array(scans))
    no_of_training_scans = round(scans.shape[0] * training_percentage)

    if randomise:
        np.random.shuffle(scans)

    add_to_training = True

    for i, scan_path in enumerate(scans):

        if i >= no_of_training_scans:
            add_to_training = False

        for label_path in glob.glob(scan_path + "*-labelling.npy", recursive=True):

            sample_path_without_ext = label_path[:-ext_len]
            label = sample_path_without_ext.rsplit('/', 1)[1]

            # assign to lists for partition
            if add_to_training:
                training_labels.append(label)
            else:
                validation_labels.append(label)

            # read file and assign to labels
            labels[label] = label + "-labelling"

    partition["train"] = training_labels
    partition["validation"] = validation_labels
    return partition, labels
