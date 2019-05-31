import glob
import numpy as np
import matplotlib.pyplot as plt
from utility_functions import opening_files
from utility_functions.labels import LABELS_NO_B, LABELS_NO_B_OR_L6, LABELS_NO_L6


def vertebrae_frequencies(dataset_dir, file_ext=".lml"):
    paths = glob.glob(dataset_dir + "/**/*" + file_ext, recursive=True)

    frequencies = np.zeros(len(LABELS_NO_B))

    for labelling_path in paths:
        labels, centroids = opening_files.extract_centroid_info_from_lml(labelling_path)
        for label in labels:
            idx = LABELS_NO_B.index(label)
            frequencies[idx] += 1

    x = np.arange(len(LABELS_NO_B))
    plt.bar(x, frequencies, 0.5)
    plt.xticks(x, LABELS_NO_B)
    plt.show()


def vertebrae_frequencies_in_samples(samples_dir, plot_path, file_ext="-labelling.npy"):
    paths = glob.glob(samples_dir + "/**/*" + file_ext, recursive=True)

    frequencies = np.zeros(len(LABELS_NO_B_OR_L6))

    for labelling_path in paths:
        labelling = np.load(labelling_path)
        unique_labels_idx = np.round(np.unique(labelling)).astype(int)
        for label_idx in unique_labels_idx:
            if label_idx > 0:
                frequencies[label_idx - 1] += 1

    x = np.arange(len(LABELS_NO_B_OR_L6))
    plt.bar(x, frequencies, 0.5)
    plt.xticks(x, LABELS_NO_B_OR_L6)
    plt.savefig(plot_path + '/vertebrae_frequencies_in_samples.png')


def vertebrae_pixel_frequencies_in_samples(samples_dir, plot_path, file_ext="-labelling.npy"):
    paths = glob.glob(samples_dir + "/**/*" + file_ext, recursive=True)

    frequencies = np.zeros(len(LABELS_NO_B_OR_L6))

    for labelling_path in paths:
        labelling = np.load(labelling_path)
        bincounts = np.bincount(labelling.reshape(-1).astype(int), minlength=len(LABELS_NO_L6))
        frequencies += bincounts[1:]

    x = np.arange(len(LABELS_NO_B_OR_L6))
    plt.bar(x, frequencies, 0.5)
    plt.xticks(x, LABELS_NO_B_OR_L6)
    plt.savefig(plot_path + '/vertebrae_pixel_frequencies_in_samples.png')


# vertebrae_frequencies('datasets')
# vertebrae_frequencies_in_samples('samples/slices', 'plots')
vertebrae_pixel_frequencies_in_samples('samples/slices', 'plots')