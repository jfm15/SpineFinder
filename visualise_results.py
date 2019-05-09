import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utility_functions import opening_files, sampling_helper_functions


def visualise_results(data_path, centroids_path, prediction_path, plot_dir, cut_range=(0.2, 0.8), rows=1, cols=6, file_ext="-prediction.npy"):

    volume = opening_files.read_nii(data_path)
    prediction = np.load(prediction_path)

    prediction = sampling_helper_functions.labelling(prediction)

    labels, centroids = opening_files.extract_centroid_info_from_lml(centroids_path)
    spacing = (2.0, 2.0, 2.0)
    centroid_indexes = np.round(centroids / np.array(spacing)).astype(int)

    label_translation = ["B", "C1", "C2", "C3", "C4", "C5", "C6", "C7",
                         "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9",
                         "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5", "L6",
                         "S1", "S2"]

    for i in range(0, volume.shape[0]):
        for j in range(0, volume.shape[1]):
            for k in range(0, volume.shape[2]):
                if prediction[i, j, k] != 0:
                    label = -1
                    min_distance = 1000
                    for label_name, centroid_idx in zip(labels, centroid_indexes):
                        dist = np.linalg.norm((i, j, k) - centroid_idx)
                        if dist < min_distance:
                            min_distance = dist
                            label = label_translation.index(label_name)
                    prediction[i, j, k] = label

    # centroid_x = centroid_indexes[:, 1]
    # centroid_y = centroid_indexes[:, 2]

    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(9.3, 6))

    fig.subplots_adjust(left=0.03, right=0.97, hspace=0.3, wspace=0.05)

    lower = volume.shape[0] * cut_range[0]
    upper = volume.shape[0] * cut_range[1]
    cut_indices = np.round(np.linspace(lower, upper, num=rows*cols)).astype(int)

    for ax, cut_idx in zip(axs.flat, cut_indices):
        volume_slice = volume[cut_idx, :, :]
        prediction_slice = prediction[cut_idx, :, :]

        ax.imshow(volume_slice.T, vmin=-1000, vmax=1000)
        # ax.scatter(centroid_x, centroid_y, color="red")

        if np.unique(prediction_slice).shape[0] > 1:
            masked_data = np.ma.masked_where(prediction_slice == 0, prediction_slice)
            ax.imshow(masked_data.T, cmap=cm.jet, alpha=0.5)

    plt.tight_layout()

    ext_len = len(file_ext)
    name = (prediction_path.rsplit('/', 1)[-1])[:-ext_len]
    name = name + "-plots"
    plot_path = '/'.join([plot_dir, name])
    plt.savefig(plot_path)


visualise_results(data_path="datasets/spine-2/patient0090/3155447/3155447.nii.gz",
                  centroids_path="datasets/spine-2/patient0090/3155447/3155447.lml",
                  prediction_path="predictions/six_conv_20_epochs/3155447-prediction.npy",
                  plot_dir="plots")